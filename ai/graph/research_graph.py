"""
ai/graph/nodes.py

Every agent and tool step wrapped as a LangGraph node function.

Each node:
  - Receives the full ContractAnalysisState or ResearchState
  - Runs its logic
  - Returns a partial dict with only the fields it updates
  - Sets current_node for progress tracking
  - Catches exceptions and sets status=failed + error message

Contract nodes:
  pre_analysis_node     run all tools against vector store (no LLM)
  extractor_node        extract clauses from contract text
  classifier_node       verify and correct clause types
  risk_node             score risk and flag dangerous clauses
  summarizer_node       generate plain-English summary
  augment_risk_node     merge tool findings into risk report
  finalize_node         assemble FullAnalysisResult

Research nodes:
  search_node           Tavily web search
  reader_node           scrape best URL
  writer_node           synthesize research report
  critic_node           review and score the report
"""

import json
import logging
import time

logger = logging.getLogger(__name__)


# ── CONTRACT ANALYSIS NODES ───────────────────────────────────────────────────

async def pre_analysis_node(state: dict) -> dict:
    """Run all vector-store tools before any LLM call. Fast, no API cost."""
    from ai.tools.risk_scorer_tool import (
        check_missing_clauses, search_risky_patterns, check_ambiguous_language
    )
    from ai.tools.document_summary_tool import (
        extract_key_dates, extract_governing_law,
        extract_payment_terms, extract_obligations
    )

    document_id = state["document_id"]
    logger.info(f"[Node: pre_analysis] doc={document_id}")

    findings = {}
    tool_map = {
        "missing_clauses":    (check_missing_clauses,    document_id),
        "risky_patterns":     (search_risky_patterns,    document_id),
        "ambiguous_language": (check_ambiguous_language, document_id),
        "key_dates":          (extract_key_dates,        document_id),
        "governing_law":      (extract_governing_law,    document_id),
        "payment_terms":      (extract_payment_terms,    document_id),
        "obligations":        (extract_obligations,      document_id),
    }

    for key, (tool_fn, arg) in tool_map.items():
        try:
            findings[key] = json.loads(tool_fn.invoke(arg))
            logger.info(f"  {key}: ok")
        except Exception as e:
            logger.warning(f"  {key}: failed — {e}")
            findings[key] = {}

    return {
        "tool_findings": findings,
        "current_node":  "pre_analysis",
        "status":        "running",
    }


async def extractor_node(state: dict) -> dict:
    """Extract all clauses from raw contract text using Mistral."""
    from ai.agents.extractor_agent import run_extractor
    from ai.guardrails.guardrail_service import protect_contract

    logger.info(f"[Node: extractor] doc={state['document_id']}")

    # Guard and sanitise contract text before sending to LLM
    contract_guard = protect_contract(state["contract_text"])
    safe_text = contract_guard["text"]
    if contract_guard.get("warnings"):
        logger.warning(f"  contract warnings: {contract_guard['warnings']}")

    try:
        extraction = await run_extractor(
            contract_text=safe_text,
            document_id=state["document_id"],
            filename=state["filename"],
        )
        logger.info(f"  clauses={extraction.total_clauses}")
        return {
            "extraction":   extraction,
            "current_node": "extractor",
            "status":       "running",
        }
    except Exception as e:
        logger.error(f"  extractor failed: {e}")
        return {
            "current_node": "extractor",
            "status":       "failed",
            "error":        f"Extractor failed: {str(e)}",
        }


async def classifier_node(state: dict) -> dict:
    """Verify and correct clause type labels."""
    from ai.agents.classifier_agent import run_classifier

    logger.info(f"[Node: classifier] doc={state['document_id']}")

    try:
        extraction = await run_classifier(state["extraction"])
        return {
            "extraction":   extraction,
            "current_node": "classifier",
            "status":       "running",
        }
    except Exception as e:
        logger.warning(f"  classifier failed: {e} — keeping original types")
        return {
            "current_node": "classifier",
            "status":       "running",   # non-fatal — continue with original
        }


async def risk_node(state: dict) -> dict:
    """Score contract risk 0-100 and flag dangerous clauses."""
    from ai.agents.risk_agent import run_risk_agent

    logger.info(f"[Node: risk] doc={state['document_id']}")

    try:
        risk_report = await run_risk_agent(state["extraction"])
        logger.info(
            f"  score={risk_report.risk_score} "
            f"flags={risk_report.total_flags} "
            f"overall={risk_report.overall_risk}"
        )
        return {
            "risk_report":  risk_report,
            "current_node": "risk",
            "status":       "running",
        }
    except Exception as e:
        logger.error(f"  risk agent failed: {e}")
        return {
            "current_node": "risk",
            "status":       "failed",
            "error":        f"Risk agent failed: {str(e)}",
        }


async def augment_risk_node(state: dict) -> dict:
    """Merge tool findings into the risk report — adds missing clause flags."""
    from ai.schemas.risk_schema import RiskFlag, RiskLevel
    from ai.tools.risk_scorer_tool import calculate_risk_score

    logger.info(f"[Node: augment_risk] doc={state['document_id']}")

    risk_report  = state["risk_report"]
    tool_findings = state.get("tool_findings", {})

    existing_count = len(risk_report.flags)
    new_flags = []
    counter = existing_count + 1

    missing = tool_findings.get("missing_clauses", {}).get("missing", [])
    for clause_name in missing:
        new_flags.append(RiskFlag(
            flag_id=f"R-T{counter:03d}",
            clause_id="N/A",
            risk_level=RiskLevel.MEDIUM,
            category="missing-clause",
            description=f"Standard '{clause_name}' clause not found.",
            suggestion=f"Add a {clause_name} clause to protect both parties.",
            flagged_text=None,
        ))
        counter += 1

    patterns = tool_findings.get("risky_patterns", {}).get("patterns_found", [])
    for pattern in patterns:
        try:
            level = RiskLevel(pattern.get("risk_level", "medium"))
        except ValueError:
            level = RiskLevel.MEDIUM
        new_flags.append(RiskFlag(
            flag_id=f"R-T{counter:03d}",
            clause_id="N/A",
            risk_level=level,
            category=pattern.get("category", "other"),
            description=pattern.get("description", ""),
            suggestion=f"Review '{pattern.get('pattern')}' language carefully.",
            flagged_text=pattern.get("context", "")[:200],
        ))
        counter += 1

    if new_flags:
        risk_report.flags.extend(new_flags)
        risk_report.total_flags = len(risk_report.flags)

        flags_input = json.dumps({
            "flags": [{"risk_level": f.risk_level.value} for f in risk_report.flags]
        })
        score_result = json.loads(calculate_risk_score.invoke(flags_input))
        risk_report.risk_score = score_result.get("risk_score", risk_report.risk_score)
        try:
            risk_report.overall_risk = RiskLevel(
                score_result.get("overall_risk", risk_report.overall_risk.value)
            )
        except ValueError:
            pass

        logger.info(
            f"  added {len(new_flags)} tool flags — "
            f"new_score={risk_report.risk_score} total={risk_report.total_flags}"
        )

    return {
        "risk_report":  risk_report,
        "current_node": "augment_risk",
        "status":       "running",
    }


async def summarizer_node(state: dict) -> dict:
    """Generate plain-English contract summary."""
    from ai.agents.summarizer_agent import run_summarizer

    logger.info(f"[Node: summarizer] doc={state['document_id']}")

    try:
        summary = await run_summarizer(state["extraction"], state["risk_report"])

        # Patch governing law from tools if agent missed it
        tool_gov = state.get("tool_findings", {}).get("governing_law", {})
        if not summary.governing_law and tool_gov.get("found"):
            summary.governing_law = (tool_gov.get("context", "") or "")[:100]

        logger.info(f"  type='{summary.contract_type}' parties={summary.parties}")
        return {
            "summary":      summary,
            "current_node": "summarizer",
            "status":       "running",
        }
    except Exception as e:
        logger.error(f"  summarizer failed: {e}")
        return {
            "current_node": "summarizer",
            "status":       "failed",
            "error":        f"Summarizer failed: {str(e)}",
        }


async def finalize_node(state: dict) -> dict:
    """Assemble final FullAnalysisResult and mark pipeline complete."""
    from ai.schemas.document_schema import FullAnalysisResult

    elapsed = round(time.time() - state.get("started_at", time.time()), 2)
    logger.info(f"[Node: finalize] doc={state['document_id']} elapsed={elapsed}s")

    result = FullAnalysisResult(
        document_id=state["document_id"],
        filename=state["filename"],
        status="completed",
        extraction=state["extraction"],
        risk_report=state["risk_report"],
        summary=state["summary"],
    )

    return {
        "result":       result,
        "current_node": "finalize",
        "status":       "completed",
        "elapsed":      elapsed,
    }


async def error_node(state: dict) -> dict:
    """Handle pipeline failures gracefully."""
    logger.error(
        f"[Node: error] doc={state.get('document_id')} "
        f"failed_at={state.get('current_node')} "
        f"error={state.get('error')}"
    )
    return {
        "current_node": "error",
        "status":       "failed",
    }



async def graph_build_node(state: dict) -> dict:
    """
    Build knowledge graph from contract text after analysis completes.
    Runs after summarizer — non-fatal, analysis result unaffected if this fails.

    Extracts entities + relationships via Mistral, builds NetworkX DiGraph,
    saves to data/knowledge_graphs/{document_id}.json
    """
    from ai.knowledge_graph.graph_service import build_document_graph

    document_id = state.get("document_id", "unknown")
    logger.info(f"[Node: graph_build] doc={document_id}")

    try:
        result = await build_document_graph(
            contract_text=state.get("contract_text", ""),
            document_id=document_id,
            filename=state.get("filename", ""),
        )
        logger.info(
            f"  graph built: nodes={result.get('total_entities')} "
            f"edges={result.get('total_relations')}"
        )
        return {
            "graph_result": result,
            "current_node": "graph_build",
            "status":       "running",
        }
    except Exception as e:
        logger.warning(f"  graph_build failed (non-fatal): {e}")
        return {
            "graph_result": {"status": "failed", "error": str(e)},
            "current_node": "graph_build",
            "status":       "running",   # non-fatal — continue to finalize
        }

# ── RESEARCH NODES ────────────────────────────────────────────────────────────

async def search_node(state: dict) -> dict:
    """Run Tavily web search for the research topic."""
    from ai.agents.research.agents import build_search_agent

    topic = state["topic"]
    logger.info(f"[Node: search] topic='{topic[:50]}'")

    try:
        agent = build_search_agent()
        result = await agent.ainvoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        search_results = result["messages"][-1].content
        logger.info(f"  search results: {len(search_results)} chars")
        return {
            "search_results": search_results,
            "current_node":   "search",
            "status":         "running",
        }
    except Exception as e:
        logger.error(f"  search failed: {e}")
        return {
            "current_node": "search",
            "status":       "failed",
            "error":        f"Search failed: {str(e)}",
        }


async def reader_node(state: dict) -> dict:
    """Scrape the most relevant URL from search results."""
    from ai.agents.research.agents import build_reader_agent

    topic          = state["topic"]
    search_results = state["search_results"]
    logger.info(f"[Node: reader] topic='{topic[:50]}'")

    try:
        agent = build_reader_agent()
        result = await agent.ainvoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it.\n\n"
                f"Search Results:\n{search_results[:800]}"
            )]
        })
        scraped = result["messages"][-1].content
        logger.info(f"  scraped: {len(scraped)} chars")
        return {
            "scraped_content": scraped,
            "current_node":    "reader",
            "status":          "running",
        }
    except Exception as e:
        logger.error(f"  reader failed: {e}")
        return {
            "current_node": "reader",
            "status":       "failed",
            "error":        f"Reader failed: {str(e)}",
        }


async def writer_node(state: dict) -> dict:
    """Synthesize research into a structured report."""
    from ai.agents.research.agents import writer_chain

    topic = state["topic"]
    logger.info(f"[Node: writer] topic='{topic[:50]}'")

    try:
        research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"SCRAPED CONTENT:\n{state['scraped_content']}"
        )
        report = await writer_chain.ainvoke({
            "topic":    topic,
            "research": research_combined,
        })
        logger.info(f"  findings={len(report.findings)}")
        return {
            "report":       report,
            "current_node": "writer",
            "status":       "running",
        }
    except Exception as e:
        logger.error(f"  writer failed: {e}")
        return {
            "current_node": "writer",
            "status":       "failed",
            "error":        f"Writer failed: {str(e)}",
        }


async def critic_node(state: dict) -> dict:
    """Review and score the research report."""
    from ai.agents.research.agents import critic_chain

    report = state["report"]
    logger.info(f"[Node: critic] title='{report.title[:50]}'")

    try:
        report_text = (
            f"Title: {report.title}\nSummary: {report.summary}\n"
            f"Findings: {report.findings}\nAnalysis: {report.analysis}"
        )
        feedback = await critic_chain.ainvoke({"report": report_text})
        elapsed = round(time.time() - state.get("started_at", time.time()), 2)
        logger.info(f"  score={feedback.score}/10 elapsed={elapsed}s")
        return {
            "feedback":     feedback,
            "current_node": "critic",
            "status":       "completed",
            "elapsed":      elapsed,
        }
    except Exception as e:
        logger.error(f"  critic failed: {e}")
        return {
            "current_node": "critic",
            "status":       "failed",
            "error":        f"Critic failed: {str(e)}",
        }