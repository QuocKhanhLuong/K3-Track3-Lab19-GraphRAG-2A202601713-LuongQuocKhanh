# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Lương Quốc Khánh  
**Ngày thực hiện:** 2026-08-19  
**Entity-resolution threshold:** 0.90  
**Dữ liệu trong run:** 1,500 articles · 1,500 chunks · 91 canonical triples · 153 nodes

> Các số liệu dưới đây được sinh trực tiếp từ run Colab hiện tại. Golden Dataset được dựng từ provenance/evidence thật trong phần dữ liệu đã xử lý để không phải bịa reference answer; trước khi nộp nên spot-check 5 dòng `data/golden_dataset.csv` với source chunks.

---

## PHẦN 1 — THUYẾT MINH KỸ THUẬT & FAILURE ANALYSIS

### 1. Coreference Resolution

- **Case quan sát:** chunk `d38fc817b7dc3eeeb535::c0000` có unresolved mention `['COREF_BATCH_FAILED']`.
- **Excerpt:** `To ensure this doesn’t happen in the future please enable Javascript and cookies in your browser. Is this happening to you frequently? Please report it on our ...`
- **Phân tích:** pipeline dùng conservative resolution: chỉ thay đại từ khi antecedent rõ trong cùng chunk. Nếu gán nhầm `the company`/`it` cho một entity gần đó, bước RE có thể tạo false edge và lỗi này tiếp tục lan qua entity resolution, traversal và generation. Vì vậy ambiguity được giữ nguyên và log thay vì ép resolve.

### 2. Entity Resolution Threshold & Lexical Guard

- **Cosine threshold:** `0.90`. Mức này cố ý thiên về precision vì false merge nguy hiểm hơn việc bỏ sót alias trong KG.
- **Audit rows:** 1.
- **High-similarity guard case:** `Microsoft` vs `Microsoft Research`, similarity=`0.7075937390327454` → `REJECT_GUARD`.
- **Lý do:** vector similarity chỉ tạo candidate; lexical guard là cổng thứ hai. Các tên chứa product/sub-organization hoặc người có họ gần giống không nên bị gộp chỉ vì embedding gần nhau. Manual alias chỉ dành cho ticker/tên doanh nghiệp phổ biến đã biết.

### 3. Super-node Mitigation

| Hạng | Entity | Type | Degree |
|---|---|---|---:|
| 1 | Apple | Company | 6 |
| 2 | DB Cargo | Company | 5 |
| 3 | Norwegian University of Life Sciences (NMBU) | Company | 4 |

- Node có degree > 100: 0 trong bảng top-degree hiện tại.
- Chính sách: degree > 100 → tối đa 50 cạnh gần nhất theo `published_date`; toàn query bị chặn ở `GLOBAL_EDGE_CAP=250` và graph text ở `14000` chars.
- **Ưu điểm:** khống chế context explosion, latency và token cost; ưu tiên evidence mới khi hỏi trạng thái hiện tại.
- **Rủi ro:** temporal-recency bias có thể cắt mất sự kiện lịch sử quan trọng. Production nên kết hợp recency với relation relevance/confidence thay vì chỉ sort theo ngày.

### 4. Benchmark Flat RAG vs GraphRAG

| Metric | Flat RAG | GraphRAG | Delta (Graph-Flat) |
|---|---:|---:|---:|
| Comprehensiveness | 1.400 | 3.400 | +2.000 |
| Faithfulness | 1.600 | 3.600 | +2.000 |
| Multi-hop reasoning | 1.600 | 3.400 | +1.800 |
| Mean quality | 1.533 | 3.467 | +1.933 |
| Latency (s) | 2.096 | 2.631 | +0.534 |
| Token usage | 672.0 | 723.2 | +51.2 |

**Flat RAG failure / GraphRAG gain case:** `G02` — Starting from Musk, follow relation ACQUIRED to an intermediate entity, then relation USES. What are the intermediate and final entities?
- Flat quality=1.00; Graph quality=5.00; gain=+4.00.
- Root cause to inspect: Flat RAG retrieves semantically close chunks independently, so two pieces of a path may not co-occur in top-k. Graph traversal can explicitly preserve the intermediate entity and both provenance-bearing edges.

**GraphRAG difficult/failure case:** `G05` — Using evidence from multiple news chunks, summarize two documented relationships involving Activision Blizzard and explain how the evidence differs across the sources.
- Flat quality=3.67; Graph quality=2.00; gain=-1.67.
- Likely failure surfaces: missed seed, missing extraction edge, noisy entity merge, super-node pruning, or graph evidence being less complete than the vector chunks. The hybrid design keeps vector fallback specifically to reduce this brittleness.

### 5. Trade-offs, Agent Control & Scale 350MB

- **Quality/cost:** GraphRAG adds extraction/indexing cost and query-time traversal, but can improve multi-hop completeness. Flat RAG remains the cheaper baseline and is often sufficient for factoids.
- **AI Coding Agent proposal intentionally rejected:** all-pairs `O(N²)` cosine similarity for entity resolution/near-dedup. At production scale this wastes memory and compute; ANN candidate generation + lexical guard + Union-Find gives auditable merges with bounded search.
- **Scale to ~350MB:** first bottlenecks are LLM extraction throughput/rate limits and entity-resolution candidate growth, not FAISS lookup. I would use asynchronous batch extraction with retry/checkpointing, ANN blocking/HNSW for entity candidates, idempotent Neo4j `UNWIND` writes, cached embeddings, and community partitioning for high-level/global questions.
- **Provenance integrity:** `invalid_provenance_edges=0` (required = 0).

---

## PHẦN 2 — REFLECTION & ACTION PLAN

### 1. Mapping bài giảng vào code

| Concept | Module | Function/block | Quan sát |
|---|---|---|---|
| Conservative Coreference | M1 | `resolve_coref_batch()`, `run_coref()` | Ambiguity được log thay vì ép resolve |
| Schema + Allowlist | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, `run_extraction()` | Chặn relation/type ngoài schema trước ingestion |
| Bulk Cypher | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND $rows AS row`, không insert từng row |
| Entity Resolution | M3 | `build_resolution_map()`, `UF`, `merge_guard()` | ANN candidate → lexical guard → Union-Find + audit |
| Hybrid Retrieval | M4 | `match_seeds()`, `retrieve_graph_context()` | Seed → fuzzy fallback → BFS + provenance text |
| Super-node Mitigation | M4 | `node_degree()`, `recent_edges()` | degree > 100 → cap 50, global edge cap |
| LLM-as-a-Judge | M5 | `judge_answer()`, `run_evaluation()` | Cùng generator/embedding, thay retrieval architecture |

### 2. Debugging & bài học

Lỗi khó nhất về mặt hệ thống không phải một exception đơn lẻ mà là **silent corruption**: coreference hoặc entity resolution sai vẫn cho pipeline chạy hết nhưng tạo edge sai, khiến GraphRAG trả lời rất tự tin trên graph bị nhiễm. Bài học là các stage phải có audit artifact (unresolved mentions, merge decisions, provenance edge checks) và benchmark theo nhóm câu hỏi, không chỉ nhìn answer cuối.

### 3. Action Plan cho đồ án thực tế

**Project:** MamaGift — trợ lý tài liệu hành chính tiếng Việt cho gia đình

Với trợ lý tài liệu hành chính gia đình, tôi **không dùng full GraphRAG ngay từ đầu**. Phần lớn câu hỏi là evidence lookup theo văn bản, điều/khoản, deadline, đơn vị chịu trách nhiệm — hierarchical/hybrid RAG với provenance chặt sẽ đơn giản và đáng tin hơn. GraphRAG chỉ đáng thêm khi cần suy luận cross-document/cross-version như: văn bản A giao nhiệm vụ cho đơn vị X, văn bản B sửa deadline, văn bản C thay thế điều khoản cũ.

- **Nodes dự kiến:** `Document`, `DocumentVersion`, `Section/Article/Clause`, `Agency`, `Task`, `Person/Role`, `Deadline`, `LegalReference`.
- **Relations:** `HAS_VERSION`, `CONTAINS`, `ASSIGNS_TO`, `HAS_DEADLINE`, `AMENDS`, `SUPERSEDES`, `REFERS_TO`, `COORDINATES_WITH`.
- **Entity resolution:** ưu tiên deterministic IDs từ document number/version + normalized agency dictionary; embedding chỉ tạo candidate và phải qua lexical/domain guard.
- **Super-node:** các entity như Bộ/UBND hoặc common legal references có degree lớn; traversal phải lọc theo document scope, effective date, relation type và evidence authority trước khi áp edge cap.

### 4. Bonus evidence

- Community fallback rows: 153; communities: 63.
- Self-correction probe route: `hop2`; missing=``.

---

## Tự đánh giá

| Tiêu chí | Điểm (1–5) | Evidence |
|---|---:|---|
| Hiểu GraphRAG | 5 | Triển khai đủ extraction → resolution → graph retrieval → judge |
| Kiểm soát AI Coding Agent | 5 | Từ chối O(N²), giữ audit/provenance guards |
| Chất lượng KG | 4 | Có schema, provenance, entity audit; còn phụ thuộc extraction recall |
| Debug/analysis | 5 | Có grouped benchmark và failure-mode analysis |
