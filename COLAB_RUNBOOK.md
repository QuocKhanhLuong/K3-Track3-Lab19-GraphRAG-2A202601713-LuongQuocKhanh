# Lab 19 — Colab Runbook

Branch prepared by AI helper: `agent/lab19-colab-ready`.

The original notebook already contains nearly all implementation functions, but the execution calls are commented out and the Golden Dataset/report are incomplete. `colab_solution.py` turns the notebook into a reproducible end-to-end run without inventing empirical metrics.

## 1. Open the notebook in Google Colab

Open `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` from this branch in Colab. T4 GPU is recommended but the embedding model/FAISS pipeline can also run on CPU.

## 2. Add Colab Secrets

Required:

- `NEO4J_URI`
- `NEO4J_USER` = `neo4j`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` = `neo4j`
- `HF_TOKEN`
- `GROQ_API_KEY`
- `GROQ_MODEL` = `llama-3.3-70b-versatile` (or another currently available Groq chat model)
- `JUDGE_PROVIDER` = `groq` or `openai`
- `JUDGE_MODEL`
- `OPENAI_API_KEY` only when `JUDGE_PROVIDER=openai`

Optional:

- `STUDENT_NAME` = `Lương Quốc Khánh`
- `LAB_PROJECT_NAME` = project name used in the reflection
- `COREF_BATCH_SIZE` = `8`
- `EXTRACT_BATCH_SIZE` = `6`
- `LAB_RESET_GRAPH` = `1` only when the Neo4j database is dedicated to this lab and it is safe to delete existing `:Entity` nodes. Otherwise keep `0`.

Never hard-code keys into the notebook.

## 3. Run the notebook definition cells

Use **Runtime → Run all**. The reference notebook intentionally leaves the heavy pipeline calls commented, so this stage mainly installs dependencies, downloads the HackerNoon subset, and defines all required functions.

If the 300 MB streaming cell is too slow for the lab session, change the two variables in cell 1.3 before running:

```python
LIMIT_ROWS = 30000
LIMIT_MB = 80
PRIORITIZE_MB = False
```

The later `LAB_MAX_ARTICLES=1500` and `LAB_MAX_CHUNKS=3000` guards still bound the actual processing workload.

## 4. Add ONE final Colab cell

```python
!wget -q "https://raw.githubusercontent.com/QuocKhanhLuong/K3-Track3-Lab19-GraphRAG-2A202601713-LuongQuocKhanh/agent/lab19-colab-ready/colab_solution.py" -O /content/colab_solution.py
%run -i /content/colab_solution.py
```

The runner will:

1. load → exact-dedup → chunk the dataset;
2. run conservative coreference resolution;
3. run schema-constrained NER/RE;
4. run Entity Resolution with cosine threshold `0.90`, lexical guard, Union-Find, and audit logging;
5. bulk-ingest nodes/edges into Neo4j using the notebook's `UNWIND` functions;
6. assert `invalid_provenance_edges == 0`;
7. build Flat RAG + Hybrid GraphRAG;
8. construct a 5-question data-grounded Golden Dataset with `factoid`, `multi-hop`, and `cross-doc` groups from extracted evidence;
9. run LLM-as-a-Judge on both methods;
10. run super-node checks and the existing community/self-correction bonus scaffolds when possible;
11. generate the required CSVs and reports from actual metrics;
12. download `/content/lab19_submission.zip`.

## 5. What is inside the ZIP

```text
lab19_submission/
├── data/
│   └── golden_dataset.csv
├── outputs/
│   ├── graphrag_eval_results.csv
│   ├── graphrag_vs_flatrag_summary.csv
│   ├── entity_resolution_audit.csv
│   ├── guard_probe_audit.csv
│   ├── top_degree_entities.csv
│   └── extraction_errors.csv
└── reports/
    ├── lab_report.md
    ├── technical_defense.md
    ├── failure_analysis.md
    └── reflection_LuongQuocKhanh.md
```

`lab_report.md` is the single-file format described by README/ASSIGNMENT. The three split report files are generated as well because `RUBRIC.md` also names them explicitly. Keeping both formats avoids losing procedural points because the starter materials are inconsistent about report layout.

## 6. Final submission steps

1. Inspect `data/golden_dataset.csv` and spot-check the 5 reference answers/evidence against the cited chunks. The runner deliberately derives them from real extracted provenance rather than fabricating answers.
2. Inspect `outputs/extraction_errors.csv`. A few retry/rate-limit errors can be acceptable if enough triples remain, but a large failure rate should be rerun.
3. Confirm `outputs/entity_resolution_audit.csv` has useful audit rows and inspect `guard_probe_audit.csv` for false-merge protection examples.
4. In Colab choose **File → Save a copy in GitHub** so the submitted notebook contains execution outputs.
5. Add the generated `data/`, `outputs/`, and `reports/` files to the repository and commit them.

## Expected defense points

- Flat RAG is the cheaper/faster baseline and often wins on simple factoids.
- GraphRAG should be strongest when the answer requires explicit A → B → C reasoning across chunks.
- False coreference and false entity merges are more dangerous than a retrieval miss because they create confident but structurally wrong graph evidence.
- Super-node pruning controls token/context explosion but creates recency bias; production ranking should combine relation relevance, confidence, time, and query intent.
- The rejected AI-agent approach is all-pairs `O(N²)` entity similarity/near-dedup; the lab uses ANN candidates + guards instead.
