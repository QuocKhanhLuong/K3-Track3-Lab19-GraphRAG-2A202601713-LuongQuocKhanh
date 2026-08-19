# Lab 19: Production-Grade GraphRAG vs Flat RAG

**AICB-K34 · Ngày 19 · Track 3: GraphRAG**  
**Môi trường:** Google Colab + Neo4j AuraDB  
**Dữ liệu:** HackerNoon Tech Company News Data Dump (`HackerNoon/tech-company-news-data-dump`)

Repo này dùng **notebook gốc của BTC** làm flow chạy chính. Không dùng custom OpenAI one-click runner hay preflight wrapper nữa.

## Chạy lab

Mở trực tiếp notebook:

`Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`

trên Google Colab và chạy theo đúng thứ tự cell trong notebook.

Các API key / Neo4j credentials cần thiết thì khai báo theo đúng tên biến/secret mà notebook gốc yêu cầu. Không commit secret vào GitHub.

Nếu có lỗi, debug ngay tại cell gốc đang lỗi; không chèn runtime patch/preflight riêng trước khi xác nhận notebook BTC thật sự cần sửa.

## Mục tiêu lab

Pipeline so sánh:

```text
HackerNoon data
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
Golden Evaluation
      ↓
LLM-as-a-Judge + latency + token usage
```

## Tài liệu chính

- `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` — notebook gốc BTC, source of truth để chạy lab.
- `ASSIGNMENT.md` — yêu cầu bài làm.
- `RUBRIC.md` — tiêu chí chấm.
- `data/` — Golden/evaluation data đã có trong repo.

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
- super-node mitigation + context cap

### Evaluation

Theo flow/rubric của notebook BTC và các file Golden/evaluation có sẵn trong repo.

## Nguyên tắc hiện tại

Ưu tiên chạy **nguyên bản notebook BTC trước**. Chỉ sửa đúng lỗi xuất hiện trong notebook gốc sau khi có traceback cụ thể; không thêm wrapper riêng chỉ để dự đoán lỗi trước.