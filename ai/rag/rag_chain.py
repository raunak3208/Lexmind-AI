"""
ai/rag/rag_chain.py

Builds a LangChain RAG chain wired to:
  - Mistral LLM (free tier)
  - Retriever from ai/rag/retriever.py (MMR / similarity / multi-query)

Exposes:
  build_rag_chain()  ->  a runnable LangChain chain
  ask()              ->  convenience wrapper: query -> answer string
"""

import logging
from typing import Optional

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from ai.config import settings
from ai.rag.retriever import get_retriever
from ai.cache.cache_service import cache_get, cache_set   # dedicated retriever module

logger = logging.getLogger(__name__)


# -- System prompt ---------------------------------------------------------------
SYSTEM_PROMPT = """You are LexMind, an expert legal document analysis assistant.
You help lawyers, paralegals, and clients understand contracts.

RULES:
1. Answer ONLY from the provided contract context and knowledge graph.
2. If the answer is not in the context, say: "This information is not found in the provided contract."
3. Quote relevant clause text when helpful -- use quotation marks.
4. Be precise and concise. Avoid legal jargon unless quoting the contract.
5. If asked about risk or ambiguity, flag it clearly.
6. Use the knowledge graph section to answer relationship questions (who owes what to whom).

Contract context:
{context}

{graph_context}
"""

HUMAN_PROMPT = "{question}"


def _format_docs(docs) -> str:
    """Join retrieved chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def get_llm() -> ChatMistralAI:
    """
    Mistral LLM -- mistral-small-latest is FREE on Mistral free tier.
    temperature=0 -> deterministic answers for legal use.
    """
    return ChatMistralAI(
        model=settings.mistral_llm_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0,
        max_retries=3,
    )


def build_rag_chain(
    document_id: Optional[str] = None,
    k: int = None,
    strategy: str = "mmr",
    use_graph: bool = True,
):
    """
    Build and return a LangChain RAG chain.

    Args:
        document_id: restrict retrieval to one document (per-doc chat).
                     Pass None to search all documents (global search).
        k:           number of context chunks to retrieve.
        strategy:    retrieval strategy from retriever.py --
                       "mmr"         -> balanced relevance + diversity (default)
                       "similarity"  -> pure cosine, best for search
                       "multi_query" -> best recall, 1 extra Mistral call
                       "reranked"    -> two-stage: fetch wide, rerank precise (best quality)

    Returns:
        A LangChain Runnable that accepts {"question": str}
        and returns a string answer.
    """
    retriever = get_retriever(strategy=strategy, document_id=document_id, k=k)
    llm       = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )

    # -- Chain assembly (LangChain Expression Language) --------------------------
    #
    #  question --+-> retriever -> format_docs -> context -+
    #             +-------------------------------------->  prompt -> llm -> parser
    #
    def _get_graph_context(inputs):
        if not use_graph:
            return ""
        try:
            from ai.knowledge_graph.graph_service import query_graph
            ctx = query_graph(inputs, document_id=document_id)
            if ctx:
                return f"Knowledge Graph Context:\n{ctx}"
            return ""
        except Exception:
            return ""

    rag_chain = (
        RunnableParallel(
            {
                "context":       retriever | _format_docs,
                "question":      RunnablePassthrough(),
                "graph_context": _get_graph_context,
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info(
        f"RAG chain built  strategy={strategy}  "
        f"document_id={document_id or 'ALL'}  k={k or settings.retriever_k}"
    )
    return rag_chain


async def ask(
    question: str,
    document_id: Optional[str] = None,
    k: int = None,
    strategy: str = "mmr",
    use_graph: bool = True,
) -> dict:
    """
    High-level async function: ask a question, get an answer.

    Args:
        question:    the user's question about the contract
        document_id: optional -- restrict to one document
        k:           number of context chunks
        strategy:    retrieval strategy ("mmr" | "similarity" | "multi_query")

    Returns:
        {
          "answer": str,
          "document_id": str | None,
          "question": str,
          "strategy": str
        }
    """
    # Check semantic cache first — zero Mistral calls on hit
    cached = cache_get(question, document_id=document_id, strategy=strategy)
    if cached:
        logger.info(
            f"RAG cache HIT  sim={cached.get('similarity')}  "
            f"doc={document_id or 'ALL'}  q='{question[:60]}'"
        )
        return cached

    chain = build_rag_chain(document_id=document_id, k=k, strategy=strategy, use_graph=use_graph)

    logger.info(
        f"RAG query (cache miss)  strategy={strategy}  "
        f"doc={document_id or 'ALL'}  q='{question[:80]}'"
    )
    answer = await chain.ainvoke(question)

    # Store in cache for future requests
    cache_set(
        question=question,
        answer=answer,
        document_id=document_id,
        strategy=strategy,
    )

    return {
        "answer":      answer,
        "document_id": document_id,
        "question":    question,
        "strategy":    strategy,
        "cache_hit":   False,
    }
