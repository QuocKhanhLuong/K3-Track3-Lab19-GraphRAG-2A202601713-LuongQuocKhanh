# Failure Analysis — Lab 19

## Case 1 — Flat RAG weaker

**G02**: Starting from Musk, follow relation ACQUIRED to an intermediate entity, then relation USES. What are the intermediate and final entities?

Flat quality=1.00; Graph quality=5.00; gain=+4.00.

Flat retrieval can miss one half of a multi-hop chain because top-k similarity ranks chunks independently. GraphRAG makes the intermediate entity explicit and preserves edge provenance.

## Case 2 — GraphRAG difficult

**G05**: Using evidence from multiple news chunks, summarize two documented relationships involving Activision Blizzard and explain how the evidence differs across the sources.

Flat quality=3.67; Graph quality=2.00; gain=-1.67.

Potential causes are seed miss, extraction recall, wrong entity resolution, super-node pruning, or stale/missing graph evidence. The practical mitigation is hybrid graph + vector context plus explicit diagnostics and self-correction.
