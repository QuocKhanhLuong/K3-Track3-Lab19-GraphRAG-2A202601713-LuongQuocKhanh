"""Resume Lab 19 after BTC's Groq extraction stage returned zero rows.

Run this in the SAME Colab runtime after run_btc_original.py failed at [3/8].
It preserves all BTC prompts/functions and only swaps the chat transport to OpenAI,
then resumes from extraction instead of repeating download/chunk/coreference.
"""

from __future__ import annotations

import random
import time
import urllib.request

import pandas as pd
from openai import OpenAI


_required = [
    "get_secret",
    "run_extraction",
    "extraction_source",
    "groq_chat",
    "groq_json",
    "parse_json_object",
    "OPENAI_API_KEY",
]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(
        "This recovery cell must run in the SAME runtime after the [3/8] failure. "
        "Missing: " + ", ".join(_missing)
    )

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from Colab Secrets.")

# Show the real BTC/Groq failure that was swallowed batch-by-batch by run_extraction().
if "extraction_errors_df" in globals() and isinstance(extraction_errors_df, pd.DataFrame) and not extraction_errors_df.empty:
    print("[diagnostic] First Groq extraction errors:")
    for err in extraction_errors_df["error"].astype(str).head(3):
        print(" -", err[:700])

_OPENAI_PIPELINE_MODEL = get_secret("OPENAI_PIPELINE_MODEL", "gpt-4.1-mini")
_openai_client = OpenAI(api_key=OPENAI_API_KEY)


def groq_chat(messages, model=None, json_mode=False, max_retries=6):
    """BTC-compatible chat function using OpenAI transport for recovery."""
    last = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": _OPENAI_PIPELINE_MODEL,
                "messages": messages,
                "temperature": 0.0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = _openai_client.chat.completions.create(**kwargs)
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            return resp.choices[0].message.content, usage
        except Exception as exc:
            last = exc
            if attempt == max_retries - 1:
                break
            time.sleep(min(20, 2 ** attempt + random.random()))
    raise RuntimeError(last)


print(f"[recovery] Reusing {len(extraction_source):,} already-coreferenced chunks.")
print(f"[recovery] Retrying BTC NER/RE with OpenAI model: {_OPENAI_PIPELINE_MODEL}")

raw_triples_df, extraction_errors_df = run_extraction(
    extraction_source,
    batch_size=EXTRACT_BATCH_SIZE,
)
if raw_triples_df.empty:
    print("[recovery] Extraction still returned 0 rows. First errors:")
    if not extraction_errors_df.empty:
        for err in extraction_errors_df["error"].astype(str).head(5):
            print(" -", err[:1000])
    raise RuntimeError("OpenAI recovery extraction returned 0 rows.")

print(
    f"[recovery] raw triples={len(raw_triples_df):,} | "
    f"failed batches={len(extraction_errors_df):,}"
)

# Continue from BTC solution's [4/8] marker, reusing all state already built before the failure.
_solution_url = (
    "https://raw.githubusercontent.com/QuocKhanhLuong/"
    "K3-Track3-Lab19-GraphRAG-2A202601713-LuongQuocKhanh/"
    "main/colab_solution.py"
)
_source = urllib.request.urlopen(_solution_url, timeout=30).read().decode("utf-8")
_marker = 'print("\\n[4/8] Entity resolution + lexical guard...")'
_pos = _source.find(_marker)
if _pos < 0:
    raise RuntimeError("Could not find [4/8] resume marker in colab_solution.py")

print("[recovery] Resuming from [4/8] without repeating download/coreference...")
exec(compile(_source[_pos:], _solution_url + "#resume-4", "exec"), globals(), globals())
