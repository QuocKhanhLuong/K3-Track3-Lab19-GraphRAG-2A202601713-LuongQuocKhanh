"""Runtime compatibility + preflight patch for the Lab 19 Colab runner.

Run after the reference notebook definition cells and before colab_solution.py.
The patch intentionally keeps the starter notebook's legacy names
(groq_client, GROQ_MODEL, groq_chat, groq_json) so the rest of the lab can run
unchanged while using OpenAI end-to-end.
"""

import json
import os
import random
import re
import time


def _require_global(name):
    if name not in globals():
        raise RuntimeError(
            f"Missing `{name}`. Run the reference notebook definition cells first."
        )


_require_global("get_secret")


def _strip_wrapping(value):
    value = str(value or "").replace("\ufeff", "").strip()
    value = value.replace("```bash", "").replace("```env", "").replace("```", "").strip()
    return value.strip().strip('"').strip("'").strip()


def _extract_named_value(raw_value, key):
    """Accept plain VALUE, `KEY=VALUE`, `KEY = VALUE`, or a full Aura credential block."""
    raw = _strip_wrapping(raw_value)
    if not raw:
        return ""

    match = re.search(
        rf"(?mi)^\s*{re.escape(key)}\s*=\s*([^\r\n#]+?)\s*$",
        raw,
    )
    if match:
        return _strip_wrapping(match.group(1))

    match = re.match(rf"(?i)^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", raw)
    if match:
        return _strip_wrapping(match.group(1))

    return raw


def _extract_block_value(raw_value, key):
    """Extract KEY from a multi-line credentials block; return empty if absent."""
    raw = _strip_wrapping(raw_value)
    if not raw:
        return ""
    match = re.search(
        rf"(?mi)^\s*{re.escape(key)}\s*=\s*([^\r\n#]+?)\s*$",
        raw,
    )
    return _strip_wrapping(match.group(1)) if match else ""


def _secret_first(*names, default=""):
    for name in names:
        raw = get_secret(name, "")
        value = _extract_named_value(raw, name)
        if value:
            return value
    return default


def _normalize_neo4j_uri(value):
    """Extract/normalize a Neo4j URI from common Colab/Aura copy-paste formats."""
    raw = _strip_wrapping(value)
    if not raw:
        return ""

    match = re.search(
        r"(?i)((?:neo4j|bolt)(?:\+s|\+ssc)?://[^\s'\"`]+)",
        raw,
    )
    if match:
        return match.group(1).rstrip(",;")

    match = re.search(r"(?i)([a-z0-9-]+\.databases\.neo4j\.io)", raw)
    if match:
        return "neo4j+s://" + match.group(1)

    return raw


# -----------------------------------------------------------------------------
# Neo4j Aura credentials
# Preferred setup: one Colab Secret named NEO4J_CREDENTIALS containing the full
# downloaded Aura credential file. This removes cross-secret stale/mismatch risk.
# Separate NEO4J_* secrets remain supported as a fallback.
# -----------------------------------------------------------------------------
_AURA_BLOCK = _strip_wrapping(get_secret("NEO4J_CREDENTIALS", ""))

if _AURA_BLOCK:
    NEO4J_URI = _normalize_neo4j_uri(_extract_block_value(_AURA_BLOCK, "NEO4J_URI"))
    NEO4J_USER = (
        _extract_block_value(_AURA_BLOCK, "NEO4J_USERNAME")
        or _extract_block_value(_AURA_BLOCK, "NEO4J_USER")
    )
    NEO4J_PASSWORD = _extract_block_value(_AURA_BLOCK, "NEO4J_PASSWORD")
    NEO4J_DATABASE = _extract_block_value(_AURA_BLOCK, "NEO4J_DATABASE") or "neo4j"
    NEO4J_CREDENTIAL_SOURCE = "NEO4J_CREDENTIALS block"
else:
    NEO4J_URI = _normalize_neo4j_uri(_secret_first("NEO4J_URI"))
    NEO4J_USER = _secret_first("NEO4J_USERNAME", "NEO4J_USER")
    NEO4J_PASSWORD = _secret_first("NEO4J_PASSWORD")
    NEO4J_DATABASE = _secret_first("NEO4J_DATABASE", default="neo4j")
    NEO4J_CREDENTIAL_SOURCE = "individual NEO4J_* secrets"

NEO4J_USERNAME = NEO4J_USER

_supported_neo4j_schemes = (
    "neo4j://",
    "neo4j+s://",
    "neo4j+ssc://",
    "bolt://",
    "bolt+s://",
    "bolt+ssc://",
)

_missing_neo4j = [
    name
    for name, value in {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME or NEO4J_USER": NEO4J_USER,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
    }.items()
    if not value
]
if _missing_neo4j:
    raise RuntimeError(
        "Missing Neo4j credentials: " + ", ".join(_missing_neo4j) + ". "
        "Recommended: create one Colab Secret named NEO4J_CREDENTIALS and paste "
        "the complete Aura credential file into it."
    )
if not NEO4J_URI.lower().startswith(_supported_neo4j_schemes):
    raise RuntimeError(
        "Could not parse NEO4J_URI. Recommended: create one Colab Secret named "
        "NEO4J_CREDENTIALS and paste the complete Aura credential file into it."
    )


# -----------------------------------------------------------------------------
# OpenAI is forced for the entire one-click run. Old GROQ/JUDGE_PROVIDER secrets
# are intentionally ignored so a stale Colab Secret cannot switch the run back.
# -----------------------------------------------------------------------------
from openai import OpenAI

LLM_PROVIDER = "openai"
LLM_MODEL = _secret_first("LLM_MODEL", default="gpt-4.1-mini")
OPENAI_API_KEY = _secret_first("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in Colab Secrets.")

groq_client = OpenAI(api_key=OPENAI_API_KEY)
GROQ_MODEL = LLM_MODEL
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = _secret_first("JUDGE_MODEL", default=LLM_MODEL)


def groq_chat(messages, model=None, json_mode=False, max_retries=8):
    model = model or GROQ_MODEL
    last_error = None

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = groq_client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            } if getattr(response, "usage", None) is not None else {}
            return text, usage
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            delay = min(30.0, (2 ** attempt) + random.random())
            print(
                f"[LLM retry {attempt + 1}/{max_retries}] "
                f"{type(exc).__name__}; sleeping {delay:.1f}s"
            )
            time.sleep(delay)

    raise RuntimeError(f"LLM request failed after {max_retries} attempts: {last_error}")


def _parse_json_object(text):
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def groq_json(system, user, model=None):
    text, usage = groq_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        json_mode=True,
    )
    return _parse_json_object(text), usage


# -----------------------------------------------------------------------------
# Dataset compatibility and official first-5000 scope.
# -----------------------------------------------------------------------------
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


LAB_MAX_ARTICLES = int(_secret_first("LAB_MAX_ARTICLES", default="5000"))
LAB_MAX_CHUNKS = int(_secret_first("LAB_MAX_CHUNKS", default="12000"))
EXTRACTION_MAX_CHUNKS = int(
    _secret_first("EXTRACTION_MAX_CHUNKS", default=str(LAB_MAX_CHUNKS))
)

os.environ["COREF_BATCH_SIZE"] = _secret_first("COREF_BATCH_SIZE", default="16")
os.environ["EXTRACT_BATCH_SIZE"] = _secret_first("EXTRACT_BATCH_SIZE", default="16")


# -----------------------------------------------------------------------------
# Automatic fail-fast preflight before any expensive LLM extraction.
# -----------------------------------------------------------------------------
def preflight_services():
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError

    print("\n[preflight] Neo4j Aura...")
    print(f"[preflight] credential source: {NEO4J_CREDENTIAL_SOURCE}")
    print(f"[preflight] uri: {NEO4J_URI}")
    print(f"[preflight] username: {NEO4J_USER}")
    print(f"[preflight] database: {NEO4J_DATABASE}")

    test_driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    try:
        try:
            test_driver.verify_connectivity()
        except AuthError as exc:
            raise RuntimeError(
                "Neo4j Aura rejected the username/password. The URI reached Aura, "
                "so this is an authentication credential mismatch, not a URI/parser "
                "error. Re-download/reset the Aura instance credentials and update "
                "NEO4J_CREDENTIALS (recommended)."
            ) from exc

        with test_driver.session(database=NEO4J_DATABASE) as session:
            ok = session.run("RETURN 1 AS ok").single()["ok"]
        if ok != 1:
            raise RuntimeError("Neo4j preflight query did not return 1.")
    finally:
        test_driver.close()
    print("[preflight] Neo4j OK")

    print("[preflight] OpenAI...")
    text, _ = groq_chat(
        [{"role": "user", "content": "Reply exactly: OK"}],
        model=GROQ_MODEL,
        max_retries=3,
    )
    if not text.strip().upper().startswith("OK"):
        raise RuntimeError(f"OpenAI preflight returned unexpected text: {text[:120]!r}")
    print("[preflight] OpenAI OK")


print("=" * 72)
print("Lab 19 runtime patch active")
print(f"LLM provider          : {LLM_PROVIDER}")
print(f"Pipeline model        : {GROQ_MODEL}")
print(f"Judge                 : {JUDGE_PROVIDER} / {JUDGE_MODEL}")
print(f"Neo4j credential src  : {NEO4J_CREDENTIAL_SOURCE}")
print(f"Neo4j URI loaded      : {bool(NEO4J_URI)}")
print(f"Neo4j user            : {NEO4J_USER}")
print(f"Neo4j database        : {NEO4J_DATABASE}")
print(f"LAB_MAX_ARTICLES      : {LAB_MAX_ARTICLES}")
print(f"LAB_MAX_CHUNKS        : {LAB_MAX_CHUNKS}")
print(f"EXTRACTION_MAX_CHUNKS : {EXTRACTION_MAX_CHUNKS}")
print("=" * 72)
