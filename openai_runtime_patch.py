"""Runtime compatibility patch for Lab 19.

Run this AFTER executing the definition cells from the reference notebook and
BEFORE running colab_solution.py.

It deliberately keeps the notebook's existing function names (groq_chat,
groq_json, groq_client, GROQ_MODEL) so no large notebook rewrite is needed.
When LLM_PROVIDER=openai, those wrappers transparently call OpenAI instead.
"""

import os


def _require_global(name):
    if name not in globals():
        raise RuntimeError(
            f"Missing `{name}`. Run the reference notebook definition cells first."
        )


_require_global("get_secret")

LLM_PROVIDER = str(get_secret("LLM_PROVIDER", "openai") or "openai").strip().lower()
LLM_MODEL = str(get_secret("LLM_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()

if LLM_PROVIDER == "openai":
    from openai import OpenAI

    OPENAI_API_KEY = str(get_secret("OPENAI_API_KEY", "") or "").strip()
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to Colab Secrets before running this patch."
        )

    # Keep these legacy names because groq_chat()/groq_json() look them up at call time.
    groq_client = OpenAI(api_key=OPENAI_API_KEY)
    GROQ_MODEL = LLM_MODEL

    # Use OpenAI for the judge as well unless explicitly overridden.
    JUDGE_PROVIDER = str(get_secret("JUDGE_PROVIDER", "openai") or "openai").lower()
    JUDGE_MODEL = str(get_secret("JUDGE_MODEL", LLM_MODEL) or LLM_MODEL)

elif LLM_PROVIDER == "groq":
    from groq import Groq

    GROQ_API_KEY = str(get_secret("GROQ_API_KEY", "") or "").strip()
    GROQ_MODEL = str(get_secret("GROQ_MODEL", "llama-3.3-70b-versatile") or "").strip()
    if not GROQ_API_KEY or not GROQ_MODEL:
        raise RuntimeError("LLM_PROVIDER=groq requires GROQ_API_KEY and GROQ_MODEL.")
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    raise ValueError("LLM_PROVIDER must be 'openai' or 'groq'.")


# HackerNoon currently exposes article text as `description`. The starter
# notebook did not include that name in its text-column candidates.
def pick_col(df, candidates, required=True):
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        key = str(candidate).lower()
        if key in lookup:
            return lookup[key]

    normalized = {str(c).lower() for c in candidates}
    text_candidates = {"text", "content", "article", "body", "story"}
    if normalized & text_candidates:
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

    if required:
        raise KeyError(
            f"Missing one of columns: {candidates}. Available columns: {list(df.columns)}"
        )
    return None


# The official Golden set is explicitly based on the first 5,000 rows. Override
# the starter lab guards at runtime so Flat RAG indexes the matching scope and
# GraphRAG is allowed to extract the corresponding chunk range.
LAB_MAX_ARTICLES = int(get_secret("LAB_MAX_ARTICLES", "5000"))
LAB_MAX_CHUNKS = int(get_secret("LAB_MAX_CHUNKS", "10000"))
EXTRACTION_MAX_CHUNKS = int(get_secret("EXTRACTION_MAX_CHUNKS", "5000"))

# colab_solution.py reads these through get_secret(). Larger batches reduce API
# round-trips while keeping prompts small enough for the extraction JSON schema.
os.environ.setdefault("COREF_BATCH_SIZE", "12")
os.environ.setdefault("EXTRACT_BATCH_SIZE", "12")

print("=" * 72)
print("Lab 19 runtime patch active")
print(f"LLM provider        : {LLM_PROVIDER}")
print(f"Pipeline model      : {GROQ_MODEL}")
print(f"Judge               : {JUDGE_PROVIDER} / {JUDGE_MODEL}")
print(f"LAB_MAX_ARTICLES    : {LAB_MAX_ARTICLES}")
print(f"LAB_MAX_CHUNKS      : {LAB_MAX_CHUNKS}")
print(f"EXTRACTION_MAX_CHUNKS: {EXTRACTION_MAX_CHUNKS}")
print("=" * 72)
