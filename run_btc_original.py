"""Run the original BTC Lab 19 notebook scaffold end-to-end.

Use AFTER `Runtime -> Run all` on
Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb.

This file does not replace the BTC notebook or redefine its architecture. It only:
1) fixes the HackerNoon `description` text-column compatibility gap; and
2) executes the already-defined BTC pipeline through the existing colab_solution.py.
"""

from __future__ import annotations

import urllib.request


# The BTC notebook defines these functions/classes during Run all.
_required = [
    "get_secret",
    "pick_col",
    "load_news",
    "standardize_news",
    "build_chunks",
    "run_coref",
    "run_extraction",
    "connect_neo4j",
    "run_evaluation",
]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(
        "Run Runtime -> Run all on the BTC notebook first. Missing definitions: "
        + ", ".join(_missing)
    )


# BTC's loader originally checks text/content/article/body/story, while the
# HackerNoon stream used in this lab exposes article text as `description`.
_original_pick_col = pick_col


def pick_col(df, candidates, required=True):
    try:
        return _original_pick_col(df, candidates, required=required)
    except KeyError:
        lookup = {str(c).lower(): c for c in df.columns}
        requested = {str(c).lower() for c in candidates}
        text_aliases = {"text", "content", "article", "body", "story"}
        if requested & text_aliases:
            for fallback in (
                "description",
                "articlebody",
                "article_body",
                "summary",
                "markdown",
            ):
                if fallback in lookup:
                    print(f"[schema] Using `{lookup[fallback]}` as article text column")
                    return lookup[fallback]
        raise


# Execute the existing solution helper in THIS notebook namespace so it reuses
# the BTC functions already defined above. No provider/preflight wrapper is added.
_solution_url = (
    "https://raw.githubusercontent.com/QuocKhanhLuong/"
    "K3-Track3-Lab19-GraphRAG-2A202601713-LuongQuocKhanh/"
    "main/colab_solution.py"
)
print("[run] Executing BTC GraphRAG pipeline...")
_source = urllib.request.urlopen(_solution_url, timeout=30).read().decode("utf-8")
exec(compile(_source, _solution_url, "exec"), globals(), globals())
