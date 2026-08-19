# Reflection — Lab 19



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
