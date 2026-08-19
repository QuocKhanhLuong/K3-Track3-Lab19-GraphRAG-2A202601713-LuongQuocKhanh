"""Run the official first-5000 / 50-question Golden benchmark.

This script is intentionally strict: it only runs after colab_solution.py has
successfully built the Flat FAISS index, entity matcher and Neo4j graph in the
same runtime. It never tries to create a second Neo4j connection from secrets.
"""

from pathlib import Path
import shutil

import pandas as pd


# -----------------------------------------------------------------------------
# Fail fast if the full solution did not finish.
# -----------------------------------------------------------------------------
for required in (
    "run_evaluation",
    "comparison_table",
    "validate_golden",
    "run_cypher",
):
    if required not in globals():
        raise RuntimeError(
            f"Missing `{required}`. colab_solution.py must finish successfully "
            "before the official Golden evaluation."
        )

if globals().get("flat_index") is None:
    raise RuntimeError(
        "Flat FAISS index is not built. colab_solution.py did not reach [5/8]."
    )
if int(getattr(flat_index, "ntotal", 0) or 0) <= 0:
    raise RuntimeError("Flat FAISS index is empty; refusing to run Golden 50.")

if globals().get("entity_match_index") is None:
    raise RuntimeError(
        "Entity matcher is not built. colab_solution.py did not finish indexing."
    )
if int(getattr(entity_match_index, "ntotal", 0) or 0) <= 0:
    raise RuntimeError("Entity matcher is empty; refusing to run Golden 50.")

try:
    neo4j_probe = run_cypher("RETURN 1 AS ok")
    if not neo4j_probe or int(neo4j_probe[0].get("ok", 0)) != 1:
        raise RuntimeError("Neo4j probe returned an unexpected result.")
except Exception as exc:
    raise RuntimeError(
        "Neo4j graph is not available in this runtime. "
        "colab_solution.py must finish successfully before Golden 50."
    ) from exc

print(
    f"Runtime ready: flat_index={flat_index.ntotal:,} vectors | "
    f"entity_index={entity_match_index.ntotal:,} entities | Neo4j=OK"
)


ROOT = Path("/content/lab19_submission")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
for directory in (DATA_DIR, OUT_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LOCAL_CANDIDATES = [
    Path("/content/lab19/data/graphrag_golden_50_first5000.csv"),
    Path("data/graphrag_golden_50_first5000.csv"),
]
GOLDEN_URL = (
    "https://raw.githubusercontent.com/QuocKhanhLuong/"
    "K3-Track3-Lab19-GraphRAG-2A202601713-LuongQuocKhanh/main/"
    "data/graphrag_golden_50_first5000.csv"
)

golden_path = next((p for p in LOCAL_CANDIDATES if p.exists()), None)
if golden_path is not None:
    official_golden_df = pd.read_csv(golden_path)
    source_desc = str(golden_path)
else:
    official_golden_df = pd.read_csv(GOLDEN_URL)
    source_desc = GOLDEN_URL

required_columns = {
    "id",
    "group",
    "question",
    "reference_answer",
    "reference_evidence",
}
missing = required_columns - set(official_golden_df.columns)
if missing:
    raise RuntimeError(f"Official Golden file is missing columns: {sorted(missing)}")
if len(official_golden_df) != 50:
    raise RuntimeError(
        f"Expected 50 official Golden questions, found {len(official_golden_df)}."
    )

validate_golden(official_golden_df, require_answers=True)
print("=" * 80)
print("OFFICIAL GOLDEN EVALUATION — FIRST 5,000 ROWS / 50 QUESTIONS")
print("source:", source_desc)
print("groups:", official_golden_df["group"].value_counts().to_dict())
print("=" * 80)

# Make the official set the canonical Golden artifact for submission.
golden_df = official_golden_df.copy()
golden_df.to_csv(DATA_DIR / "golden_dataset.csv", index=False)
golden_df.to_csv(DATA_DIR / "graphrag_golden_50_first5000.csv", index=False)

print("\nRunning Flat RAG vs GraphRAG on all 50 official questions...")
eval_results_df = run_evaluation(golden_df)
comparison_df = comparison_table(eval_results_df)

# Canonical rubric filenames now point to the official benchmark.
eval_results_df.to_csv(OUT_DIR / "graphrag_eval_results.csv", index=False)
comparison_df.to_csv(OUT_DIR / "graphrag_vs_flatrag_summary.csv", index=False)

# Explicit copies make final-result provenance obvious.
eval_results_df.to_csv(OUT_DIR / "graphrag_eval_results_official50.csv", index=False)
comparison_df.to_csv(OUT_DIR / "graphrag_vs_flatrag_summary_official50.csv", index=False)


def _mean(column):
    if column not in eval_results_df.columns:
        return float("nan")
    return float(pd.to_numeric(eval_results_df[column], errors="coerce").mean())


def _fmt(value, digits=3):
    return "N/A" if pd.isna(value) else f"{value:.{digits}f}"


flat_comp = _mean("flat_comprehensiveness")
graph_comp = _mean("graph_comprehensiveness")
flat_faith = _mean("flat_faithfulness")
graph_faith = _mean("graph_faithfulness")
flat_hop = _mean("flat_multi_hop_reasoning")
graph_hop = _mean("graph_multi_hop_reasoning")
flat_latency = _mean("flat_latency_s")
graph_latency = _mean("graph_latency_s")
flat_tokens = _mean("flat_total_tokens")
graph_tokens = _mean("graph_total_tokens")

summary_md = f"""# Official Golden-50 Evaluation

**Scope:** HackerNoon first 5,000 rows  
**Questions:** {len(golden_df)}  
**Groups:** {golden_df['group'].value_counts().to_dict()}  
**Golden source:** `data/graphrag_golden_50_first5000.csv`

| Metric | Flat RAG | GraphRAG | Delta |
|---|---:|---:|---:|
| Comprehensiveness | {_fmt(flat_comp)} | {_fmt(graph_comp)} | {_fmt(graph_comp-flat_comp)} |
| Faithfulness | {_fmt(flat_faith)} | {_fmt(graph_faith)} | {_fmt(graph_faith-flat_faith)} |
| Multi-hop reasoning | {_fmt(flat_hop)} | {_fmt(graph_hop)} | {_fmt(graph_hop-flat_hop)} |
| Mean latency (s) | {_fmt(flat_latency)} | {_fmt(graph_latency)} | {_fmt(graph_latency-flat_latency)} |
| Mean total tokens | {_fmt(flat_tokens, 1)} | {_fmt(graph_tokens, 1)} | {_fmt(graph_tokens-flat_tokens, 1)} |

The canonical submission CSVs in `outputs/` are generated from this official
50-question benchmark. The earlier five-question data-grounded benchmark from
`colab_solution.py` is only a pipeline sanity check.
"""

(REPORT_DIR / "official_golden_50.md").write_text(summary_md, encoding="utf-8")

# Append/replace one clearly delimited official section in the main report.
lab_report = REPORT_DIR / "lab_report.md"
start_marker = "\n<!-- OFFICIAL_GOLDEN_50_START -->\n"
end_marker = "\n<!-- OFFICIAL_GOLDEN_50_END -->\n"
section = start_marker + summary_md + end_marker
if lab_report.exists():
    text = lab_report.read_text(encoding="utf-8")
    if start_marker in text and end_marker in text:
        before = text.split(start_marker, 1)[0]
        after = text.split(end_marker, 1)[1]
        text = before + section + after
    else:
        text = text.rstrip() + "\n" + section
    lab_report.write_text(text, encoding="utf-8")
else:
    lab_report.write_text(summary_md, encoding="utf-8")

print("\nOfficial evaluation complete.")
print("Saved:")
print(" -", OUT_DIR / "graphrag_eval_results.csv")
print(" -", OUT_DIR / "graphrag_vs_flatrag_summary.csv")
print(" -", REPORT_DIR / "official_golden_50.md")

display(eval_results_df)
display(comparison_df)

zip_path = Path("/content/lab19_submission_official50.zip")
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix("")), "zip", ROOT)
print("\nZIP:", zip_path)

try:
    from google.colab import files
    files.download(str(zip_path))
except Exception:
    pass
