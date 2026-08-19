"""Resume Lab 19 from Neo4j connection without repeating extraction/coreference.

Use in the SAME Colab runtime after [4/8] entity resolution completed and Neo4j auth failed.
Before running, update the Colab secrets with the NEW Aura instance credentials.
"""

from pathlib import Path
import urllib.request

_required = [
    "get_secret", "connect_neo4j", "setup_graph_schema", "run_cypher",
    "bulk_insert_nodes", "bulk_insert_edges", "graph_checks",
    "build_flat_index", "build_entity_matcher",
    "triples_df", "nodes_df", "chunks_df",
]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(
        "Current runtime no longer has the [4/8] state. Missing: " + ", ".join(_missing)
    )

# Reload current Colab secrets into the already-defined BTC notebook globals.
# Aura's downloaded credential file uses NEO4J_USERNAME, while the BTC notebook
# expects NEO4J_USER. Prefer the fresh Aura name and only fall back to BTC's alias.
NEO4J_URI = get_secret("NEO4J_URI", "")
NEO4J_USERNAME = get_secret("NEO4J_USERNAME", "")
NEO4J_USER = NEO4J_USERNAME or get_secret("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "")
NEO4J_DATABASE = get_secret("NEO4J_DATABASE", "neo4j")

try:
    if globals().get("driver") is not None:
        driver.close()
except Exception:
    pass
driver = None

print("[resume] Reusing completed [4/8] state:")
print(f"         triples={len(triples_df):,} | nodes={len(nodes_df):,} | chunks={len(chunks_df):,}")
print("[resume] Fresh Aura settings:")
print(f"         uri={NEO4J_URI}")
print(f"         user={NEO4J_USER}")
print(f"         database={NEO4J_DATABASE}")
print(f"         password_loaded={bool(NEO4J_PASSWORD)} (value hidden)")
print("[resume] Connecting to Neo4j...")
connect_neo4j()
setup_graph_schema()

RESET_GRAPH = str(get_secret("LAB_RESET_GRAPH", "0")).strip().lower() in {"1", "true", "yes"}
if RESET_GRAPH:
    print("LAB_RESET_GRAPH=1 -> deleting existing :Entity nodes before ingestion")
    run_cypher("MATCH (n:Entity) DETACH DELETE n")
else:
    existing = run_cypher("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"]
    if existing:
        print(f"WARNING: Neo4j already has {existing} :Entity nodes; ingestion uses MERGE and is idempotent by entity id/chunk edge key.")

bulk_insert_nodes(nodes_df)
bulk_insert_edges(triples_df)
graph_counts, top_degree_df = graph_checks()

print("\n[5/8] Building Flat RAG and entity-matching indexes...")
build_flat_index(chunks_df)
build_entity_matcher(nodes_df)

# Reuse the unchanged remainder of colab_solution.py starting at Golden construction.
_solution_url = (
    "https://raw.githubusercontent.com/QuocKhanhLuong/"
    "K3-Track3-Lab19-GraphRAG-2A202601713-LuongQuocKhanh/"
    "main/colab_solution.py"
)
_source = urllib.request.urlopen(_solution_url, timeout=30).read().decode("utf-8")
_marker = "# 4. Build a DATA-GROUNDED Golden Dataset"
_pos = _source.find(_marker)
if _pos < 0:
    raise RuntimeError("Could not find Golden continuation marker in colab_solution.py")
print("[resume] Neo4j ingestion OK. Continuing from [6/8] Golden/evaluation...")
exec(compile(_source[_pos:], _solution_url + "#resume-6", "exec"), globals(), globals())
