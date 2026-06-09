"""
evaluation/report_generator.py

Generates visual HTML + JSON reports from evaluation results.

Input:  result dict from ragas_evaluator.py
Output: evaluation/results/report_<timestamp>.html
        evaluation/results/report_<timestamp>.json

Location: evaluation/report_generator.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _score_color(score: float) -> str:
    """Return color based on score value."""
    if score >= 0.8:
        return "#22c55e"   # green
    elif score >= 0.6:
        return "#f59e0b"   # amber
    elif score >= 0.4:
        return "#f97316"   # orange
    return "#ef4444"       # red


def _score_label(score: float) -> str:
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    return "Poor"


def _metric_description(metric: str) -> str:
    descriptions = {
        "faithfulness":        "Is the answer grounded in retrieved context? High = no hallucination.",
        "answer_relevancy":    "Does the answer directly address the question asked?",
        "context_recall":      "Did retrieval find chunks containing the correct answer?",
        "context_precision":   "What fraction of retrieved chunks were actually useful?",
        "composite_score":     "Weighted average of all metrics.",
        "composite":           "Weighted average of all metrics.",
    }
    return descriptions.get(metric, "")


def generate_html_report(result: dict, output_path: str = None) -> str:
    """
    Generate an HTML evaluation report from a single strategy result
    or a comparison result.

    Args:
        result:       output from run_local_evaluation, run_ragas_evaluation,
                      or compare_strategies
        output_path:  where to save the HTML file

    Returns:
        path to the saved HTML file
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    is_comparison = "comparison" in result

    if not output_path:
        output_path = str(RESULTS_DIR / f"report_{ts}.html")

    if is_comparison:
        html = _build_comparison_html(result, ts)
    else:
        html = _build_single_html(result, ts)

    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"HTML report saved: {output_path}")
    return output_path


def _build_single_html(result: dict, ts: str) -> str:
    """Build HTML for single strategy evaluation."""
    scores    = result.get("scores", {})
    strategy  = result.get("strategy", "unknown")
    mode      = result.get("mode", "local")
    n_samples = result.get("n_samples", 0)
    doc_id    = result.get("document_id", "all documents")
    per_sample = result.get("per_sample", [])

    metric_rows = ""
    for metric, score in scores.items():
        color = _score_color(score)
        label = _score_label(score)
        desc  = _metric_description(metric)
        bar_w = int(score * 100)
        metric_rows += f"""
        <tr>
          <td class="metric-name">{metric.replace('_', ' ').title()}</td>
          <td>
            <div class="bar-bg">
              <div class="bar-fill" style="width:{bar_w}%; background:{color}"></div>
            </div>
          </td>
          <td><span class="score-badge" style="background:{color}">{score:.4f}</span></td>
          <td><span class="label" style="color:{color}">{label}</span></td>
          <td class="desc">{desc}</td>
        </tr>"""

    sample_rows = ""
    for i, s in enumerate(per_sample):
        m = s.get("metrics", {})
        composite = m.get("composite_score", 0)
        color = _score_color(composite)
        sample_rows += f"""
        <tr>
          <td>{i+1}</td>
          <td class="question">{s['question']}</td>
          <td class="answer">{s['answer'][:150]}{'...' if len(s['answer']) > 150 else ''}</td>
          <td><span class="score-badge" style="background:{color}">{composite:.3f}</span></td>
          <td>{m.get('context_recall', 0):.3f}</td>
          <td>{m.get('context_precision', 0):.3f}</td>
          <td>{m.get('answer_relevancy', 0):.3f}</td>
          <td>{m.get('faithfulness', 0):.3f}</td>
        </tr>"""

    composite = scores.get("composite_score", scores.get("composite", 0))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LexMind RAG Evaluation — {strategy}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #818cf8; font-size: 28px; margin-bottom: 4px; }}
  .subtitle {{ color: #64748b; font-size: 14px; margin-bottom: 32px; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }}
  .meta-card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
  .meta-card .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }}
  .meta-card .value {{ font-size: 22px; font-weight: 700; color: #818cf8; margin-top: 4px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }}
  h2 {{ color: #cbd5e1; font-size: 18px; margin: 0 0 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; color: #64748b; font-weight: 600; padding: 10px 12px; border-bottom: 1px solid #334155; }}
  td {{ padding: 12px; border-bottom: 1px solid #1e293b; vertical-align: middle; }}
  tr:hover td {{ background: #0f172a; }}
  .bar-bg {{ background: #0f172a; border-radius: 4px; height: 10px; width: 160px; }}
  .bar-fill {{ height: 10px; border-radius: 4px; transition: width 0.3s; }}
  .score-badge {{ color: white; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 13px; }}
  .label {{ font-weight: 600; }}
  .desc {{ color: #64748b; font-size: 12px; max-width: 280px; }}
  .metric-name {{ font-weight: 600; color: #cbd5e1; }}
  .question {{ color: #94a3b8; max-width: 200px; }}
  .answer {{ color: #64748b; max-width: 220px; font-size: 12px; }}
  .composite-ring {{ text-align: center; padding: 20px; }}
  .big-score {{ font-size: 64px; font-weight: 800; }}
</style>
</head>
<body>
<div class="container">
  <h1>LexMind RAG Evaluation Report</h1>
  <p class="subtitle">Strategy: <strong>{strategy}</strong> &nbsp;|&nbsp; Mode: <strong>{mode}</strong> &nbsp;|&nbsp; Generated: {ts}</p>

  <div class="meta-grid">
    <div class="meta-card">
      <div class="label">Strategy</div>
      <div class="value" style="font-size:16px">{strategy}</div>
    </div>
    <div class="meta-card">
      <div class="label">Samples Evaluated</div>
      <div class="value">{n_samples}</div>
    </div>
    <div class="meta-card">
      <div class="label">Document</div>
      <div class="value" style="font-size:13px">{str(doc_id)[:20]}</div>
    </div>
    <div class="meta-card">
      <div class="label">Composite Score</div>
      <div class="value" style="color:{_score_color(composite)}">{composite:.4f}</div>
    </div>
  </div>

  <div class="card">
    <h2>Metric Scores</h2>
    <table>
      <tr>
        <th>Metric</th><th>Score Bar</th><th>Score</th><th>Rating</th><th>What it measures</th>
      </tr>
      {metric_rows}
    </table>
  </div>

  {'<div class="card"><h2>Per-Sample Results</h2><table><tr><th>#</th><th>Question</th><th>Answer</th><th>Composite</th><th>Recall</th><th>Precision</th><th>Relevancy</th><th>Faithful</th></tr>' + sample_rows + '</table></div>' if sample_rows else ''}

</div>
</body>
</html>"""


def _build_comparison_html(result: dict, ts: str) -> str:
    """Build HTML for multi-strategy comparison."""
    comparison = result.get("comparison", {})
    winner     = result.get("winner", "")

    strategy_cards = ""
    for strategy, r in comparison.items():
        scores    = r.get("scores", {})
        composite = scores.get("composite_score", scores.get("composite", 0))
        color     = _score_color(composite)
        is_winner = strategy == winner
        border    = "border: 2px solid #818cf8;" if is_winner else ""

        metric_lines = ""
        for metric, score in scores.items():
            m_color = _score_color(score)
            metric_lines += f"""
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
              <span style="color:#94a3b8; font-size:13px">{metric.replace('_',' ').title()}</span>
              <span style="color:{m_color}; font-weight:700">{score:.4f}</span>
            </div>"""

        winner_badge = '<span style="background:#818cf8; color:white; padding:2px 10px; border-radius:20px; font-size:12px; margin-left:8px">WINNER</span>' if is_winner else ""

        strategy_cards += f"""
        <div style="background:#1e293b; border-radius:12px; padding:24px; border:1px solid #334155; {border}">
          <h3 style="color:#cbd5e1; margin:0 0 8px">{strategy} {winner_badge}</h3>
          <div style="font-size:42px; font-weight:800; color:{color}; margin-bottom:20px">{composite:.4f}</div>
          {metric_lines}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LexMind Strategy Comparison</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #818cf8; font-size: 28px; margin-bottom: 4px; }}
  .subtitle {{ color: #64748b; font-size: 14px; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
  h3 {{ font-size: 18px; }}
</style>
</head>
<body>
<div class="container">
  <h1>LexMind Strategy Comparison</h1>
  <p class="subtitle">Winner: <strong style="color:#818cf8">{winner}</strong> &nbsp;|&nbsp; Generated: {ts}</p>
  <div class="grid">{strategy_cards}</div>
</div>
</body>
</html>"""


def generate_json_report(result: dict, output_path: str = None) -> str:
    """Save evaluation result as a clean JSON report."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if not output_path:
        output_path = str(RESULTS_DIR / f"report_{ts}.json")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"JSON report saved: {output_path}")
    return output_path