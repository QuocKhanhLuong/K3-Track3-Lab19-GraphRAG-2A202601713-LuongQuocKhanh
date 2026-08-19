# Lab 19 — One-click Colab Runbook

Use only:

`Day19_OpenAI_Colab_Run.ipynb`

The intended workflow is:

**fresh Colab runtime → add Secrets → Runtime > Run all → wait for the final ZIP**

No manual smoke-test cell, resume cell, Neo4j patch cell, or separate Golden cell is required.

## Required Colab Secrets

Add:

- `OPENAI_API_KEY`
- `HF_TOKEN`
- `NEO4J_URI`
- `NEO4J_USERNAME` **or** `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

Aura credential downloads commonly use `NEO4J_USERNAME`. The runtime supports both names and maps them to the starter notebook's `NEO4J_USER` variable automatically.

The one-click parser is intentionally tolerant. For Neo4j values it accepts:

- a plain value such as `neo4j+s://<instance>.databases.neo4j.io`;
- `KEY=value`;
- `KEY = value`;
- an Aura hostname without a scheme;
- or a full credential block copied from the Aura download.

So existing Colab Secrets do not need to be reformatted just to satisfy the runner.

Optional settings:

- `LLM_MODEL=gpt-4.1-mini` (default)
- `JUDGE_MODEL=gpt-4.1-mini` (default)
- `LAB_MAX_ARTICLES=5000` (default)
- `LAB_MAX_CHUNKS=12000` (default)
- `EXTRACTION_MAX_CHUNKS=12000` (default; extracts every retained first-5000 chunk)
- `COREF_BATCH_SIZE=16` (default)
- `EXTRACT_BATCH_SIZE=16` (default)
- `LAB_RESET_GRAPH=1` only when the Aura database is dedicated to this lab and deleting existing `:Entity` nodes is safe.

The one-click runner forces OpenAI for coreference, extraction, generation and judging; stale Groq/provider secrets are ignored.

Never hard-code secrets in the notebook or commit them to GitHub.

## What Run all does

1. Safely `cd /content`, removes any stale `/content/lab19`, and clones the latest `main`.
2. Loads the instructor notebook definitions and streams the official first 5,000 HackerNoon source rows.
3. Applies `openai_runtime_patch.py`:
   - accepts both `NEO4J_USERNAME` and `NEO4J_USER`;
   - normalizes common Aura URI copy/paste formats;
   - fixes HackerNoon `description` as the article-text column;
   - forces the legacy Groq wrappers to OpenAI;
   - adds exponential retry for LLM requests;
   - expands extraction to all retained chunks from the first-5000 corpus.
4. Automatically preflights Neo4j Aura and OpenAI **before** expensive coreference/extraction starts.
5. Runs `colab_solution.py` end-to-end and builds the Neo4j graph, Flat FAISS index, entity matcher, artifacts and reports.
6. Only if the full solution succeeds, runs `official_golden_eval.py` on the official 50-question Golden set.
7. The official evaluator checks that FAISS, the entity matcher and Neo4j are actually ready before evaluating.
8. Downloads `/content/lab19_submission_official50.zip`.

All Python runner files in the final execution cell are loaded with `exec(compile(...))`. Therefore a failed patch/preflight stops that cell immediately; it cannot continue into the solution or Golden benchmark with partial stale state.

## Final output

```text
lab19_submission/
├── data/
│   ├── golden_dataset.csv
│   └── graphrag_golden_50_first5000.csv
├── outputs/
│   ├── graphrag_eval_results.csv
│   ├── graphrag_vs_flatrag_summary.csv
│   ├── graphrag_eval_results_official50.csv
│   ├── graphrag_vs_flatrag_summary_official50.csv
│   ├── entity_resolution_audit.csv
│   ├── guard_probe_audit.csv
│   ├── top_degree_entities.csv
│   └── extraction_errors.csv
└── reports/
    ├── lab_report.md
    ├── official_golden_50.md
    ├── technical_defense.md
    ├── failure_analysis.md
    └── reflection_LuongQuocKhanh.md
```

The canonical rubric CSVs are generated from the official 50-question benchmark, not the five-question internal smoke benchmark.
