from ai.tools.clause_extractor_tool import (
    find_clauses_by_type,
    find_parties_in_document,
    get_clause_text,
    get_all_clause_type_tools,
)
from ai.tools.risk_scorer_tool import (
    check_missing_clauses,
    search_risky_patterns,
    check_ambiguous_language,
    calculate_risk_score,
    get_all_risk_tools,
)
from ai.tools.document_summary_tool import (
    extract_key_dates,
    extract_governing_law,
    extract_payment_terms,
    extract_obligations,
    get_all_summary_tools,
)
from ai.tools.comparison_tool import compare_contracts


def get_all_tools() -> list:
    """Return every tool registered in LexMind."""
    return (
        get_all_clause_type_tools()
        + get_all_risk_tools()
        + get_all_summary_tools()
        + [compare_contracts]
    )