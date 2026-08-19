# Lab 19: Production-Grade GraphRAG vs Flat RAG

**AICB-K34 · Ngày 19 · Track 3: GraphRAG**  
**Môi trường:** Google Colab + Neo4j AuraDB  
**Dữ liệu:** HackerNoon Tech Company News Data Dump (`HackerNoon/tech-company-news-data-dump`)

Repo này giữ nguyên notebook reference của đề và bổ sung một runner Colab để chạy end-to-end bằng OpenAI + Neo4j Aura + official Golden 50.

## Mục tiêu

Pipeline so sánh:

```text
HackerNoon first 5,000 rows
        ↓
Dedup + Chunking
        ↓
Conservative Coreference
        ↓
NER + Relation Extraction
        ↓
Entity Resolution
        ↓
Neo4j Knowledge Graph ─────┐
                           ├─→ GraphRAG / Hybrid Retrieval
FAISS Flat Index ──────────┘
        ↓
Official Golden 50
        ↓
LLM-as-a-Judge + latency + token usage
```

Xem `ASSIGNMENT.md` cho yêu cầu module và `RUBRIC.md` cho thang điểm.

---

## Quick Start — cách duy nhất khuyến nghị

Mở:

`Day19_OpenAI_Colab_Run.ipynb`

trên Google Colab, thêm Secrets rồi chọn:

**Runtime → Run all**

Runner được thiết kế để chạy từ fresh runtime, không cần thêm cell test/resume thủ công.

### Colab Secrets bắt buộc

```text
OPENAI_API_KEY
HF_TOKEN
NEO4J_URI
NEO4J_USERNAME   # hoặc NEO4J_USER
NEO4J_PASSWORD
NEO4J_DATABASE
```

Với Neo4j Aura, dùng đúng giá trị trong credential file tải khi tạo instance. Aura thường đặt tên username là `NEO4J_USERNAME`; runner tự map sang tên `NEO4J_USER` mà notebook reference dùng.

Mỗi Secret chỉ nên chứa **value**, ví dụ:

```text
NEO4J_URI -> neo4j+s://<instance>.databases.neo4j.io
```

Không commit API key/password vào repo.

### Optional

```text
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-4.1-mini
LAB_MAX_ARTICLES=5000
LAB_MAX_CHUNKS=12000
EXTRACTION_MAX_CHUNKS=12000
COREF_BATCH_SIZE=16
EXTRACT_BATCH_SIZE=16
LAB_RESET_GRAPH=0
```

Chỉ đặt `LAB_RESET_GRAPH=1` nếu database Aura chỉ dùng cho lab này và an toàn để xoá graph cũ.

---

## One-click runner làm gì

`Day19_OpenAI_Colab_Run.ipynb`:

1. `cd /content` trước khi xoá clone cũ, nên rerun không còn lỗi `getcwd`.
2. Clone latest `main` vào `/content/lab19`.
3. Chạy definitions từ notebook reference với đúng first 5,000 source rows.
4. Chạy `openai_runtime_patch.py` để:
   - đọc cả `NEO4J_USERNAME` và `NEO4J_USER`;
   - validate Neo4j URI;
   - dùng `description` làm HackerNoon article text fallback;
   - dùng OpenAI cho coref/extraction/seed/generation mặc định;
   - retry exponential khi LLM bị transient/rate-limit error;
   - extract toàn bộ chunk được giữ lại từ first-5000 corpus.
5. Tự preflight Neo4j + OpenAI trước khi bắt đầu hàng trăm LLM calls.
6. Chạy `colab_solution.py` để build graph, FAISS indexes, audit artifacts và report.
7. Chỉ khi solution hoàn tất mới chạy `official_golden_eval.py`.
8. Official evaluator xác minh `flat_index`, entity matcher và Neo4j graph tồn tại trước khi benchmark 50 câu.
9. Tự tải `/content/lab19_submission_official50.zip`.

---

## Reference notebook

`Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` vẫn là notebook gốc của đề và được giữ để đối chiếu kiến trúc/prompt/rubric.

Không dùng notebook reference làm flow chạy chính nếu mục tiêu là hoàn thành official Golden 50 bằng one-click runner.

---

## Thành phần kỹ thuật

### Flat RAG

- SentenceTransformer `all-MiniLM-L6-v2`
- FAISS `IndexFlatIP`
- top-k chunk retrieval

### GraphRAG

- Node types: `Company`, `Person`, `Technology`
- Relations: `ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`
- Neo4j bulk write bằng `UNWIND`
- provenance trên edge: `source_chunk_id`, `published_date`, `evidence`, `confidence`
- entity resolution: embedding candidate + lexical guard + Union-Find
- traversal: seed extraction + exact/fuzzy match + BFS
- super-node mitigation + global edge/context cap

### Evaluation

Official file:

`data/graphrag_golden_50_first5000.csv`

Schema:

```text
id, group, question, reference_answer, reference_evidence
```

Groups gồm factoid, multi-hop và cross-doc.

Metrics xuất ra gồm LLM Judge scores, latency và token usage cho Flat RAG vs GraphRAG.

---

## Output cuối

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

ZIP cuối:

`/content/lab19_submission_official50.zip`

Hai file rubric chính:

- `outputs/graphrag_eval_results.csv`
- `outputs/graphrag_vs_flatrag_summary.csv`

sẽ được official evaluator ghi bằng kết quả Golden 50.

---

## Files quan trọng

```text
Day19_OpenAI_Colab_Run.ipynb                    # runner khuyến nghị
Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb  # reference đề
openai_runtime_patch.py                          # provider + schema + Aura + preflight
colab_solution.py                                # full pipeline
 official_golden_eval.py                         # official 50-question evaluation
COLAB_RUNBOOK.md                                 # hướng dẫn chạy
ASSIGNMENT.md
RUBRIC.md
```

## Submission check

Trước khi nộp, kiểm tra:

- pipeline chạy hết tới `[8/8]`;
- official Golden chạy đủ 50 câu;
- `invalid_provenance_edges == 0`;
- `outputs/extraction_errors.csv` không có systematic API failure;
- ZIP cuối đã được tải;
- notebook executed + output/report CSV được commit nếu rubric yêu cầu.
