# Lab 19 — Colab Runbook (OpenAI + Official Golden 50)

The easiest path is now the ready-to-run notebook:

`Day19_OpenAI_Colab_Run.ipynb`

It keeps the instructor reference notebook untouched, executes its definition cells in the same Colab runtime, patches the LLM backend to OpenAI, runs the full solution, then runs the official 50-question Golden benchmark from the first 5,000 source rows.

## 1. Required Colab Secrets

Add these in the Colab Secrets panel:

- `OPENAI_API_KEY`
- `HF_TOKEN`
- `NEO4J_URI`
- `NEO4J_USER` = `neo4j`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` = `neo4j`

Optional:

- `LLM_PROVIDER` = `openai` (default)
- `LLM_MODEL` = `gpt-4.1-mini` (default)
- `JUDGE_PROVIDER` = `openai` (default)
- `JUDGE_MODEL` = `gpt-4.1-mini` (default)
- `LAB_MAX_ARTICLES` = `5000`
- `LAB_MAX_CHUNKS` = `10000`
- `EXTRACTION_MAX_CHUNKS` = `5000`
- `COREF_BATCH_SIZE` = `12`
- `EXTRACT_BATCH_SIZE` = `12`
- `LAB_RESET_GRAPH` = `1` only if the Neo4j database is dedicated to this lab and it is safe to delete the existing graph.

Never hard-code secrets in the notebook.

## 2. Run the ready notebook

Open `Day19_OpenAI_Colab_Run.ipynb` in Google Colab and run the cells from top to bottom.

The notebook does four things:

1. clones the latest `main` branch;
2. executes the instructor notebook definitions while limiting the streamed corpus to the first 5,000 rows;
3. runs `openai_runtime_patch.py` and `colab_solution.py`;
4. runs `official_golden_eval.py` on `data/graphrag_golden_50_first5000.csv`.

The OpenAI patch deliberately preserves the legacy names `groq_client`, `GROQ_MODEL`, `groq_chat`, and `groq_json`, because the reference notebook looks them up dynamically. Under the default configuration those wrappers actually call the OpenAI client.

It also fixes the HackerNoon schema mismatch by accepting `description` as the article text column.

## 3. If running inside the original notebook manually

After all definition cells are loaded, run this cell:

```python
!wget -q "https://raw.githubusercontent.com/QuocKhanhLuong/K3-Track3-Lab19-GraphRAG-2A202601713-LuongQuocKhanh/main/openai_runtime_patch.py" -O /content/openai_runtime_patch.py
!wget -q "https://raw.githubusercontent.com/QuocKhanhLuong/K3-Track3-Lab19-GraphRAG-2A202601713-LuongQuocKhanh/main/colab_solution.py" -O /content/colab_solution.py

%run -i /content/openai_runtime_patch.py
%run -i /content/colab_solution.py
```

Then run the official Golden-50 cell:

```python
!wget -q "https://raw.githubusercontent.com/QuocKhanhLuong/K3-Track3-Lab19-GraphRAG-2A202601713-LuongQuocKhanh/main/official_golden_eval.py" -O /content/official_golden_eval.py

%run -i /content/official_golden_eval.py
```

Do not restart the runtime between the solution cell and the Golden-50 cell: the evaluation reuses the in-memory FAISS index, entity matcher, retrieval functions, and the Neo4j graph built by the solution.

## 4. Output

The baseline runner writes `/content/lab19_submission/`.

The official evaluation makes the official 50-question dataset the canonical submission Golden set and overwrites the two rubric CSVs with the official benchmark results:

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

The final Golden cell also downloads:

`/content/lab19_submission_official50.zip`

## 5. Before submission

Check that `outputs/extraction_errors.csv` does not show a systematic API failure, inspect a few rows in the official Golden results, and confirm the Neo4j provenance check passed with `invalid_provenance_edges == 0`.

Then save the executed Colab notebook to GitHub and commit the generated `data/`, `outputs/`, and `reports/` artifacts.
