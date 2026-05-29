"""
llm_nlq.py — Optional, privacy-first LLM-assisted Natural-Language Query layer.
DataQual AI · v3.0 Phase 2 · RAK Pvt. Ltd

WHY THIS MODULE EXISTS
----------------------
The rule-based `parse_nlq` engine in app.py is fast, free, and predictable, but
combinatorially expensive to extend. This module adds an OPTIONAL LLM path that
translates a natural-language question into a constrained "query plan", which our
own deterministic code then executes. It is additive and opt-in — the app keeps
working with zero external calls when this is disabled (default).

THE SIX DESIGN GUARANTEES (Analytics-Engineering decisions)
-----------------------------------------------------------
1. FREE API ACCESS — works against any OpenAI-compatible endpoint. The genuinely
   free tiers in 2026 are Groq, Google Gemini (OpenAI-compat), and OpenRouter
   (`:free` models). The user generates their OWN key (2-min signup); we never
   ship a key. See PROVIDER_PRESETS.
2. SMALL FREE MODEL — defaults target small/fast instruct models on free tiers.
3. FUNCTION-CALLING / JSON SCHEMA — the model is forced to emit ONE structured
   object matching QUERY_PLAN_SCHEMA (via tool-calling when supported, else JSON
   mode). It cannot return prose or code.
4. FALLBACK RULE ENGINE — disabled / over-budget / errored / invalid-plan all
   transparently fall back to the caller's rule engine (`parse_nlq`).
5. COST GUARD ($0) — free-tier only + a hard per-session call cap + a character
   ceiling. When exhausted, the layer disables itself; the app keeps working.
6. SECURITY & PRIVACY — the model NEVER receives data rows or cell values. It sees
   only the question + a compact SCHEMA (column names, roles, dtypes; optional
   categorical sample values are OFF by default). Raw data never leaves the host.
   Because the model can only emit a validated plan, a hallucinating or hostile
   model cannot exfiltrate data or execute arbitrary code.

ZERO NEW DEPENDENCIES — standard library only (urllib). No `openai`, no `requests`.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from functools import reduce
from typing import Any, Callable, Optional

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Provider presets — all OpenAI-compatible chat-completions endpoints.
# The user supplies their own free key. We do not bundle keys.
# ──────────────────────────────────────────────────────────────────────────────
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "Groq (free · fast)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "signup": "https://console.groq.com/keys",
    },
    "OpenRouter (free models)": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "signup": "https://openrouter.ai/keys",
    },
    "Google Gemini (OpenAI-compat)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "signup": "https://aistudio.google.com/app/apikey",
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "model": "",
        "signup": "",
    },
}

# Operators the executor understands. The model is told to use only these.
ALLOWED_OPS = [
    "==", "!=", ">", ">=", "<", "<=",
    "between", "in", "contains", "is_null", "not_null",
]
ALLOWED_AGGS = ["count", "mean", "sum", "min", "max", "median"]

# JSON schema for the query plan (used as the tool-call parameter schema).
QUERY_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string", "enum": ["filter", "aggregate"]},
        "logic": {"type": "string", "enum": ["and", "or"]},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string", "enum": ALLOWED_OPS},
                    "value": {"type": ["string", "number", "boolean", "array", "null"]},
                    "value2": {"type": ["string", "number", "null"]},
                },
                "required": ["column", "op"],
            },
        },
        "group_by": {"type": ["string", "null"]},
        "aggregate": {"type": ["string", "null"], "enum": ALLOWED_AGGS + [None]},
        "agg_column": {"type": ["string", "null"]},
        "sort_by": {"type": ["string", "null"]},
        "sort_dir": {"type": "string", "enum": ["asc", "desc"]},
        "limit": {"type": ["integer", "null"]},
    },
    "required": ["intent"],
}

# Friendly aliases → real engineered columns produced by run_scoring().
_COLUMN_ALIASES = {
    "severity": "dq_severity",
    "grade": "dq_grade",
    "score": "dq_score",
    "quality": "dq_score",
    "quality_score": "dq_score",
    "dq": "dq_score",
    "anomaly": "is_anomaly",
    "is_anomaly": "is_anomaly",
}


class PlanError(Exception):
    """Raised when a query plan is structurally invalid or references unknown columns."""


# ──────────────────────────────────────────────────────────────────────────────
# Cost guard — UI-agnostic. app.py persists one instance in st.session_state.
# ──────────────────────────────────────────────────────────────────────────────
class CostGuard:
    """Hard $0 guard: caps calls and characters sent per session.

    Free-tier providers cost nothing, but we still bound usage so a runaway loop
    or an over-eager user cannot exhaust a free quota. When the cap is hit the
    layer reports `can_call() -> (False, reason)` and the caller uses rules.
    """

    def __init__(self, max_calls: int = 25, max_chars: int = 60_000) -> None:
        self.max_calls = int(max_calls)
        self.max_chars = int(max_chars)
        self.calls = 0
        self.chars = 0

    def can_call(self) -> tuple[bool, str]:
        if self.calls >= self.max_calls:
            return False, f"call cap reached ({self.calls}/{self.max_calls})"
        if self.chars >= self.max_chars:
            return False, f"character cap reached ({self.chars}/{self.max_chars})"
        return True, ""

    def record(self, prompt_chars: int, completion_chars: int = 0) -> None:
        self.calls += 1
        self.chars += int(prompt_chars) + int(completion_chars)

    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.calls)


# ──────────────────────────────────────────────────────────────────────────────
# Schema summary — the ONLY thing about the data the model ever sees.
# ──────────────────────────────────────────────────────────────────────────────
def build_schema_summary(
    df: pd.DataFrame,
    col_mapping: dict,
    include_samples: bool = False,
    max_samples: int = 8,
) -> dict:
    """Build a compact, privacy-safe schema description.

    NEVER includes data rows. With include_samples=True it adds up to `max_samples`
    distinct values for CATEGORICAL columns only (never IDs, numerics, or text) so
    the model can map e.g. "iOS" to the right column — this is opt-in because even
    category labels can be mildly sensitive.
    """
    role_of: dict[str, str] = {}
    for role_key, cols in (col_mapping or {}).items():
        if not isinstance(cols, (list, tuple)):
            continue
        role = role_key.replace("_columns", "").rstrip("s")
        for c in cols:
            role_of.setdefault(c, role)

    columns = []
    for c in df.columns:
        columns.append({
            "name": str(c),
            "role": role_of.get(c, "derived" if str(c).startswith(("dq_", "is_")) else "other"),
            "dtype": str(df[c].dtype),
        })

    summary: dict[str, Any] = {
        "columns": columns,
        "engineered_columns": {
            "dq_score": "overall data-quality score, integer 0-100",
            "dq_grade": "letter grade A,B,C,D,F",
            "dq_severity": "one of CLEAN,LOW,MEDIUM,HIGH,CRITICAL",
            "is_anomaly": "boolean anomaly flag (if present)",
        },
        "allowed_ops": ALLOWED_OPS,
        "allowed_aggregates": ALLOWED_AGGS,
    }

    if include_samples:
        cats = list((col_mapping or {}).get("categorical_columns", []))
        samples: dict[str, list] = {}
        for c in cats:
            if c in df.columns:
                vals = (
                    df[c].dropna().astype(str).value_counts().head(max_samples).index.tolist()
                )
                if vals:
                    samples[c] = vals
        if samples:
            summary["categorical_samples"] = samples
    return summary


_SYSTEM_PROMPT = (
    "You are a query planner for a data-quality tool. Convert the user's question "
    "into ONE JSON object describing how to filter/aggregate a table you CANNOT see. "
    "You only know the schema provided. Rules:\n"
    "- Use ONLY column names that appear in the schema. Never invent columns.\n"
    "- Use ONLY the allowed operators and aggregates.\n"
    "- intent='aggregate' ONLY when the user asks to group/break-down/rank by a column; "
    "otherwise intent='filter'.\n"
    "- Dates must be ISO 'YYYY-MM-DD'. Use op 'between' with value and value2 for ranges.\n"
    "- For 'top/worst N' set limit=N and sort by dq_score (asc=worst first).\n"
    "- Output the plan via the emit_query_plan function (or as raw JSON). No prose."
)


def _post_json(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def translate_to_plan(query: str, schema: dict, cfg: dict) -> tuple[dict, dict]:
    """Call the LLM and return (plan, usage). Raises on transport/parse failure."""
    base_url = (cfg.get("base_url") or "").rstrip("/")
    model = cfg.get("model") or ""
    api_key = cfg.get("api_key") or ""
    timeout = int(cfg.get("timeout", 20))
    if not base_url or not model or not api_key:
        raise PlanError("LLM not fully configured (base_url / model / api_key missing)")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    user_msg = (
        "SCHEMA (no data values shown):\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n\nQUESTION: "
        + query.strip()
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    prompt_chars = len(_SYSTEM_PROMPT) + len(user_msg)

    # Attempt 1: tool/function calling (preferred — strongest schema enforcement).
    tool_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 500,
        "tools": [{
            "type": "function",
            "function": {
                "name": "emit_query_plan",
                "description": "Return the structured plan for the question.",
                "parameters": QUERY_PLAN_SCHEMA,
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": "emit_query_plan"}},
    }
    data: Optional[dict] = None
    try:
        data = _post_json(url, headers, tool_payload, timeout)
    except urllib.error.HTTPError:
        # Some free endpoints reject forced tool_choice — retry with JSON mode.
        json_payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }
        data = _post_json(url, headers, json_payload, timeout)

    plan = _extract_plan(data)
    usage = data.get("usage") or {}
    completion_chars = int(usage.get("completion_tokens", 0)) * 4 or len(json.dumps(plan))
    return plan, {"prompt_chars": prompt_chars, "completion_chars": completion_chars}


def _extract_plan(data: dict) -> dict:
    """Pull the plan out of either a tool call or a JSON message body."""
    try:
        choice = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise PlanError(f"unexpected API response shape: {e}")

    tool_calls = choice.get("tool_calls") or []
    if tool_calls:
        args = tool_calls[0].get("function", {}).get("arguments", "{}")
        try:
            return json.loads(args)
        except json.JSONDecodeError as e:
            raise PlanError(f"tool arguments not valid JSON: {e}")

    content = (choice.get("content") or "").strip()
    if not content:
        raise PlanError("empty model response")
    # Strip markdown fences if present.
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"): content.rfind("}") + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise PlanError(f"model did not return JSON: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Validation + execution — the deterministic, trusted half.
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_col(name: Optional[str], df: pd.DataFrame) -> Optional[str]:
    if name is None:
        return None
    if name in df.columns:
        return name
    alias = _COLUMN_ALIASES.get(str(name).strip().lower())
    if alias and alias in df.columns:
        return alias
    # Case-insensitive direct match.
    lower = {c.lower(): c for c in df.columns}
    if str(name).strip().lower() in lower:
        return lower[str(name).strip().lower()]
    return None


def validate_plan(plan: dict, df: pd.DataFrame, col_mapping: dict) -> dict:
    """Structurally validate and normalise a plan; raise PlanError if unusable.

    Returns a normalised copy with columns resolved to real df columns.
    """
    if not isinstance(plan, dict):
        raise PlanError("plan is not an object")
    intent = plan.get("intent", "filter")
    if intent not in ("filter", "aggregate"):
        raise PlanError(f"bad intent: {intent}")

    norm = {
        "intent": intent,
        "logic": plan.get("logic", "and") if plan.get("logic") in ("and", "or") else "and",
        "filters": [],
        "group_by": None,
        "aggregate": plan.get("aggregate") if plan.get("aggregate") in ALLOWED_AGGS else None,
        "agg_column": None,
        "sort_by": None,
        "sort_dir": "asc" if plan.get("sort_dir") == "asc" else ("desc" if plan.get("sort_dir") == "desc" else "asc"),
        "limit": None,
    }

    for f in plan.get("filters") or []:
        if not isinstance(f, dict):
            continue
        col = _resolve_col(f.get("column"), df)
        op = f.get("op")
        if col is None:
            raise PlanError(f"unknown column in filter: {f.get('column')!r}")
        if op not in ALLOWED_OPS:
            raise PlanError(f"unknown op: {op!r}")
        norm["filters"].append({"column": col, "op": op, "value": f.get("value"), "value2": f.get("value2")})

    if intent == "aggregate":
        gb = _resolve_col(plan.get("group_by"), df)
        if gb is None:
            raise PlanError("aggregate plan needs a valid group_by column")
        norm["group_by"] = gb

    norm["agg_column"] = _resolve_col(plan.get("agg_column"), df)
    norm["sort_by"] = _resolve_col(plan.get("sort_by"), df)

    lim = plan.get("limit")
    if isinstance(lim, (int, float)) and lim and lim > 0:
        norm["limit"] = int(lim)

    if not norm["filters"] and intent == "filter" and norm["group_by"] is None:
        # A plan that filters nothing and groups nothing is useless — let rules try.
        if norm["limit"] is None and norm["sort_by"] is None:
            raise PlanError("empty plan (no filters, no grouping, no sort/limit)")
    return norm


def _is_date_col(col: str, col_mapping: dict) -> bool:
    return col in set((col_mapping or {}).get("date_columns", []))


def _is_numeric_col(col: str, df: pd.DataFrame) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def _coerce(series: pd.Series, value, is_date: bool, is_numeric: bool):
    if is_date:
        return pd.to_datetime(series, errors="coerce"), pd.to_datetime(value, errors="coerce")
    if is_numeric:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = value
        return pd.to_numeric(series, errors="coerce"), v
    return series.astype(str).str.strip(), str(value).strip()


def _apply_filter(df: pd.DataFrame, f: dict, col_mapping: dict) -> pd.Series:
    col, op = f["column"], f["op"]
    series = df[col]
    is_date = _is_date_col(col, col_mapping)
    is_num = (not is_date) and _is_numeric_col(col, df)

    if op == "is_null":
        return series.isna() | (series.astype(str).str.strip().isin(["", "nan", "None"]))
    if op == "not_null":
        return ~(series.isna() | (series.astype(str).str.strip().isin(["", "nan", "None"])))
    if op == "in":
        vals = f["value"] if isinstance(f["value"], list) else [f["value"]]
        return series.astype(str).str.strip().str.lower().isin([str(x).strip().lower() for x in vals])
    if op == "contains":
        return series.astype(str).str.contains(str(f["value"]), case=False, na=False)

    s, v = _coerce(series, f["value"], is_date, is_num)
    if op == "==":
        return (s.str.lower() == str(f["value"]).strip().lower()) if (not is_date and not is_num) else (s == v)
    if op == "!=":
        return (s.str.lower() != str(f["value"]).strip().lower()) if (not is_date and not is_num) else (s != v)
    if op == ">":
        return s > v
    if op == ">=":
        return s >= v
    if op == "<":
        return s < v
    if op == "<=":
        return s <= v
    if op == "between":
        _, v2 = _coerce(series, f.get("value2"), is_date, is_num)
        lo, hi = v, v2
        try:
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
        except TypeError:
            pass
        return (s >= lo) & (s <= hi)
    raise PlanError(f"unsupported op at execution: {op}")


def _describe_filters(filters: list, logic: str) -> str:
    parts = []
    for f in filters:
        if f["op"] in ("is_null", "not_null"):
            parts.append(f'{f["column"]} {f["op"].replace("_", " ")}')
        elif f["op"] == "between":
            parts.append(f'{f["column"]} between {f["value"]} and {f.get("value2")}')
        else:
            parts.append(f'{f["column"]} {f["op"]} {f["value"]}')
    return f" {logic} ".join(parts) if parts else "all records"


def execute_plan(plan: dict, df: pd.DataFrame, col_mapping: dict) -> dict:
    """Execute a VALIDATED plan, returning the parse_nlq-compatible result dict:
    {mask, summary, intent, sort_col, sort_asc, agg_df, limit, n}.
    """
    masks = [_apply_filter(df, f, col_mapping) for f in plan["filters"]]
    if not masks:
        mask = pd.Series(True, index=df.index)
    elif plan["logic"] == "or":
        mask = reduce(lambda a, b: a | b, masks)
    else:
        mask = reduce(lambda a, b: a & b, masks)
    mask = mask.fillna(False)

    sort_col = plan["sort_by"]
    sort_asc = plan["sort_dir"] == "asc"
    limit = plan["limit"]

    if plan["intent"] == "aggregate":
        gb = plan["group_by"]
        work = df[mask]
        if work.empty:
            agg_df = pd.DataFrame(columns=["Value", "Avg DQ Score", "Critical Issues", "Records"])
            return {"mask": mask, "summary": f"🤖 Smart Query → no records to group by {gb}.",
                    "intent": "aggregate", "sort_col": None, "sort_asc": True,
                    "agg_df": agg_df, "limit": limit, "n": 0}
        grouped = work.groupby(work[gb].astype(str), dropna=False)
        avg_score = grouped["dq_score"].mean() if "dq_score" in work.columns else grouped.size()
        records = grouped.size()
        if "dq_severity" in work.columns:
            crit = grouped["dq_severity"].apply(lambda s: int((s == "CRITICAL").sum()))
        else:
            crit = records * 0
        agg_df = pd.DataFrame({
            "Value": avg_score.index.astype(str),
            "Avg DQ Score": avg_score.values,
            "Critical Issues": crit.reindex(avg_score.index).fillna(0).astype(int).values,
            "Records": records.reindex(avg_score.index).fillna(0).astype(int).values,
        }).sort_values("Avg DQ Score", ascending=True).reset_index(drop=True)
        if limit:
            agg_df = agg_df.head(limit)
        n = int(agg_df["Records"].sum())
        summary = (f"🤖 Smart Query → average quality by **{gb}** "
                   f"across {len(agg_df)} group(s); worst first.")
        return {"mask": mask, "summary": summary, "intent": "aggregate",
                "sort_col": None, "sort_asc": True, "agg_df": agg_df, "limit": limit, "n": n}

    # Filter intent
    n = int(mask.sum())
    desc = _describe_filters(plan["filters"], plan["logic"])
    extra = ""
    if sort_col:
        extra += f", sorted by {sort_col} {'↑' if sort_asc else '↓'}"
    if limit:
        extra += f", top {limit}"
    summary = f"🤖 Smart Query → {n:,} record(s) where {desc}{extra}."
    return {"mask": mask, "summary": summary, "intent": "filter",
            "sort_col": sort_col, "sort_asc": sort_asc, "agg_df": None,
            "limit": limit, "n": n}


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator — the single entry point app.py calls.
# ──────────────────────────────────────────────────────────────────────────────
def smart_query(
    query: str,
    df: pd.DataFrame,
    col_mapping: dict,
    rule_fallback: Callable[[str, pd.DataFrame, dict], dict],
    cfg: dict,
    guard: Optional[CostGuard] = None,
) -> tuple[dict, str, str]:
    """Return (result_dict, engine_label, note).

    engine_label is "llm" or "rules". `note` is a short human message when a
    fallback happened (empty on the happy path). Never raises — always returns a
    usable result via the rule engine if anything goes wrong.
    """
    if not cfg.get("enabled"):
        return rule_fallback(query, df, col_mapping), "rules", ""

    if guard is not None:
        ok, why = guard.can_call()
        if not ok:
            return rule_fallback(query, df, col_mapping), "rules", f"LLM budget spent ({why}); used rules."

    t0 = time.time()
    try:
        schema = build_schema_summary(df, col_mapping, include_samples=bool(cfg.get("send_samples")))
        plan, usage = translate_to_plan(query, schema, cfg)
        if guard is not None:
            guard.record(usage.get("prompt_chars", 0), usage.get("completion_chars", 0))
        norm = validate_plan(plan, df, col_mapping)
        result = execute_plan(norm, df, col_mapping)
        elapsed = time.time() - t0
        return result, "llm", f"planned in {elapsed:.1f}s"
    except (PlanError, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, TimeoutError) as e:
        return rule_fallback(query, df, col_mapping), "rules", f"LLM path failed ({type(e).__name__}); used rules."
    except Exception as e:  # never let the NLQ box crash the app
        return rule_fallback(query, df, col_mapping), "rules", f"LLM path error ({type(e).__name__}); used rules."
