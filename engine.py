"""
engine.py — DataQual AI core engine (pure logic, no Streamlit).
================================================================
Extracted from the former monolithic app.py so the UI layer stays thin.
Rule-based NLQ parser, column-type detection, role mapping, profiling,
the 8-step cleaning engine, the 5-dimension scoring engine, IsolationForest
anomaly detection, the multi-domain demo generator, and shared severity/
grade/colour constants. No Streamlit calls live here.
"""

import json
import re
import re as _re
import random
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PASTEL = {
    "indigo":  "#915466",   # rosewood primary
    "violet":  "#C79192",   # rosy taupe
    "sky":     "#5E6472",   # blue slate
    "teal":    "#1D9E75",
    "emerald": "#3B6D11",
    "amber":   "#EF9F27",
    "rose":    "#D85A30",
    "slate":   "#5E6472",
}

_STOPWORDS = {"show","me","find","get","list","give","display","all","the",
              "records","rows","entries","data","where","which","that","with",
              "have","has","are","is","a","an","of","in","for","and","or"}

def _tokens(q: str) -> list[str]:
    return [t for t in _re.sub(r"[^\w\s<>=!\.%]", " ", q.lower()).split()
            if t not in _STOPWORDS]

_SEV_MAP = {
    "critical":"CRITICAL","crit":"CRITICAL",
    "high":"HIGH",
    "medium":"MEDIUM","med":"MEDIUM","moderate":"MEDIUM",
    "low":"LOW",
    "clean":"CLEAN","good":"CLEAN","healthy":"CLEAN","ok":"CLEAN",
}

_GRADE_MAP = {"a":"A","b":"B","c":"C","d":"D","f":"F"}

_DIM_MAP = {
    "completeness":"dq_score_completeness","missing":"dq_score_completeness","null":"dq_score_completeness","empty":"dq_score_completeness",
    "validity":"dq_score_validity","format":"dq_score_validity","invalid":"dq_score_validity",
    "accuracy":"dq_score_accuracy","value":"dq_score_accuracy","range":"dq_score_accuracy",
    "consistency":"dq_score_consistency","consistent":"dq_score_consistency",
    "uniqueness":"dq_score_uniqueness","duplicate":"dq_score_uniqueness","duplication":"dq_score_uniqueness","duplicates":"dq_score_uniqueness",
}

_DIM_HUMAN = {
    "dq_score_completeness":"Missing Data",
    "dq_score_validity":"Format Accuracy",
    "dq_score_accuracy":"Value Accuracy",
    "dq_score_consistency":"Data Consistency",
    "dq_score_uniqueness":"Duplicate Check",
}

def _parse_number(tokens: list[str]) -> float | None:
    for t in tokens:
        try: return float(t.replace("%","").replace(",",""))
        except ValueError: pass
    return None

_CMP_OPS = {
    ">":">", ">=":">=", "<":"<", "<=":"<=", "=":"==", "==":"==",
    "above":">", "over":">", "greater than":">", "higher than":">",
    "more than":">", "bigger than":">", "exceeds":">", "exceed":">",
    "below":"<", "under":"<", "less than":"<", "lower than":"<", "smaller than":"<",
}

_NLQ_MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
               "july":7,"august":8,"september":9,"october":10,"november":11,
               "december":12,"jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,
               "aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}

def _apply_op(s, op, v):
    import numpy as np
    if op == ">":  return s.fillna(-np.inf) >  v
    if op == ">=": return s.fillna(-np.inf) >= v
    if op == "<":  return s.fillna(np.inf)  <  v
    if op == "<=": return s.fillna(np.inf)  <= v
    return s == v

def _match_col(hint, cols, df):
    hint = (hint or "").strip().strip("_").lower()
    if not hint: return None
    cand = [c for c in cols if c in df.columns]
    return next((c for c in cand
                 if hint == c.lower() or hint in c.lower() or c.lower() in hint
                 or hint.replace("_"," ") in c.lower().replace("_"," ")), None)

def _date_atom(c, df, date_cols):
    import pandas as pd
    if not date_cols: return None, None
    dcol = next((x for x in date_cols if x.lower() in c or x.split("_")[0].lower() in c),
                date_cols[0])
    ds = pd.to_datetime(df[dcol], errors="coerce")
    btw = _re.search(r'between\s+(\d{4}-\d{1,2}-\d{1,2})\s+and\s+(\d{4}-\d{1,2}-\d{1,2})', c)
    aft = _re.search(r'(?:after|since|from|starting)\s+(\d{4}-\d{1,2}-\d{1,2})', c)
    bef = _re.search(r'(?:before|until|till|up to|by)\s+(\d{4}-\d{1,2}-\d{1,2})', c)
    my  = _re.search(r'\b([a-z]+)\s+(\d{4})\b', c)
    yr  = _re.search(r'\b(?:in|during|for|year)\s+(\d{4})\b', c)
    if btw:
        d1 = pd.to_datetime(btw.group(1), errors="coerce")
        d2 = pd.to_datetime(btw.group(2), errors="coerce")
        if pd.notna(d1) and pd.notna(d2):
            lo, hi = min(d1, d2), max(d1, d2)
            return ds.between(lo, hi), f"{dcol} between {lo.date()} and {hi.date()}"
    if aft:
        d1 = pd.to_datetime(aft.group(1), errors="coerce")
        if pd.notna(d1): return ds >= d1, f"{dcol} on/after {d1.date()}"
    if bef:
        d2 = pd.to_datetime(bef.group(1), errors="coerce")
        if pd.notna(d2): return ds <= d2, f"{dcol} on/before {d2.date()}"
    if my and my.group(1) in _NLQ_MONTHS:
        mo = _NLQ_MONTHS[my.group(1)]; yy = int(my.group(2))
        return (ds.dt.year == yy) & (ds.dt.month == mo), f"{dcol} in {my.group(1).title()} {yy}"
    if yr:
        yy = int(yr.group(1))
        return ds.dt.year == yy, f"{dcol} in {yy}"
    return None, None

def _nlq_atom(clause, df, col_mapping):
    """Parse ONE clause -> (bool Series, label) or None.
    A clause may hold several predicates (e.g. "critical records after 2023-06-01");
    ALL of them are AND-combined so nothing in the clause is silently dropped."""
    import pandas as pd
    c = clause.replace("§§", "and").strip()
    num_cols  = col_mapping.get("numeric_columns", [])
    cat_cols  = col_mapping.get("categorical_columns", [])
    date_cols = [x for x in col_mapping.get("date_columns", []) if x in df.columns]

    m = pd.Series([True] * len(df), index=df.index)
    labels = []
    hit = False

    sev = [_SEV_MAP[t] for t in _tokens(c) if t in _SEV_MAP]
    if sev and "dq_severity" in df.columns:
        m &= df["dq_severity"].isin(sev); labels.append(f"severity = {' or '.join(sorted(set(sev)))}"); hit = True
    gm = _re.search(r'\bgrade\s+([a-f])\b', c)
    if gm and "dq_grade" in df.columns:
        g = gm.group(1).upper(); m &= df["dq_grade"] == g; labels.append(f"grade = {g}"); hit = True
    sm = _re.search(r'score\s*(' + _CMP_ALT + r')\s*([\d\.]+)', c)
    if sm and "dq_score" in df.columns:
        op = _CMP_OPS[sm.group(1).strip()]; v = float(sm.group(2))
        m &= _apply_op(df["dq_score"], op, v); labels.append(f"DQ score {op} {v:.0f}"); hit = True
    dmask, dlabel = _date_atom(c, df, date_cols)
    if dmask is not None:
        m &= dmask; labels.append(dlabel); hit = True
    nm = _re.search(r'([\w ]+?)\s*(' + _CMP_ALT + r')\s*([\d,\.]+)', c)
    if nm:
        col = _match_col(nm.group(1), num_cols, df)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            op = _CMP_OPS[nm.group(2).strip()]; v = float(nm.group(3).replace(",", ""))
            m &= _apply_op(df[col], op, v); labels.append(f"{col} {op} {v:,.0f}"); hit = True
    if any(w in c for w in ("anomal","outlier","flagged","suspicious","unusual")) and "is_anomaly" in df.columns:
        m &= df["is_anomaly"] == True; labels.append("anomalous"); hit = True
    if any(w in c for w in ("missing","null","empty","blank","incomplete")):
        m &= df.isnull().any(axis=1); labels.append("has missing values"); hit = True
    cm2 = _re.search(r'([\w ]+?)\s*(?:==|=|is|equals|contains)\s*["\']?([\w][\w \-]*?)["\']?$', c)
    if cm2 and not sm:  # avoid colliding with "score = 50" style
        col = _match_col(cm2.group(1), cat_cols, df)
        if col:
            val = cm2.group(2).strip().lower()
            ch = df[col].astype(str).str.lower().str.contains(_re.escape(val), na=False)
            if ch.any():
                m &= ch; labels.append(f"{col} contains '{val}'"); hit = True

    if not hit:
        return None
    return m, " + ".join(labels)

def _multi_condition(q, df, col_mapping):
    """Split on AND/OR connectors, parse each atom, combine left-to-right.
    Returns {mask,label,used_or,n} or None when fewer than 2 atoms parse."""
    qp = _re.sub(r'(\bbetween\b\s+\S+)\s+and\s+', r'\1 §§ ', q)
    parts = _re.split(r'\s+(and|or)\s+', qp)
    if len(parts) < 3:
        return None
    seq = [(None, parts[0])] + [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
    parsed = [(conn, _nlq_atom(clause, df, col_mapping)) for conn, clause in seq]
    good = [a for _, a in parsed if a is not None]
    if len(good) < 2:
        return None
    mask = None; labels = []; used_or = False
    for conn, a in parsed:
        if a is None:
            continue
        m, l = a
        if mask is None:
            mask = m.copy(); labels.append(l)
        elif conn == "or":
            mask = mask | m; labels.append(f"OR {l}"); used_or = True
        else:
            mask = mask & m; labels.append(f"AND {l}")
    return {"mask": mask, "label": " · ".join(labels), "used_or": used_or, "n": len(good)}

def parse_nlq(query: str, df: "pd.DataFrame", col_mapping: dict) -> dict:
    """
    Parse a natural-language query and return:
      {"mask": pd.Series(bool), "summary": str, "intent": str,
       "sort_col": str|None, "sort_asc": bool, "agg_df": pd.DataFrame|None}
    Returns None mask means 'return all'.
    """
    import pandas as pd, numpy as np
    q    = query.strip().lower()
    toks = _tokens(q)

    mask   = pd.Series([True]*len(df), index=df.index)
    intent = "filter"
    summary_parts = []
    sort_col  = None
    sort_asc  = False
    agg_df    = None
    limit     = None

    num_cols = col_mapping.get("numeric_columns", [])
    cat_cols = col_mapping.get("categorical_columns", [])
    id_cols  = col_mapping.get("id_columns", [])

    # ── 0. Multi-condition AND/OR pre-pass  (v2.7) ────────────────────────────
    num_m = None
    matched_col = None
    _mc = _multi_condition(q, df, col_mapping)
    _skip_predicate_blocks = _mc is not None
    if _mc is not None:
        mask &= _mc["mask"]
        summary_parts.append(_mc["label"])

    if not _skip_predicate_blocks:
        # ── 1. Severity filter ────────────────────────────────────────────────────
        sev_hits = [_SEV_MAP[t] for t in toks if t in _SEV_MAP]
        if sev_hits and "dq_severity" in df.columns:
            mask &= df["dq_severity"].isin(sev_hits)
            summary_parts.append(f"severity = {' or '.join(sev_hits)}")

        # ── 2. Grade filter ───────────────────────────────────────────────────────
        grade_m = _re.search(r'\bgrade\s+([a-fA-F])\b', q)
        if grade_m and "dq_grade" in df.columns:
            g = grade_m.group(1).upper()
            mask &= df["dq_grade"] == g
            summary_parts.append(f"grade = {g}")

        # ── 3. DQ score threshold ─────────────────────────────────────────────────
        score_m = _re.search(r'score\s*(below|under|less than|<|above|over|greater than|>|between)\s*([\d\.]+)(?:\s*and\s*([\d\.]+))?', q)
        if score_m and "dq_score" in df.columns:
            op, v1, v2 = score_m.group(1), float(score_m.group(2)), score_m.group(3)
            if op in ("below","under","less than","<"):
                mask &= df["dq_score"] < v1
                summary_parts.append(f"DQ score < {v1:.0f}")
            elif op in ("above","over","greater than",">"):
                mask &= df["dq_score"] > v1
                summary_parts.append(f"DQ score > {v1:.0f}")
            elif op == "between" and v2:
                mask &= df["dq_score"].between(v1, float(v2))
                summary_parts.append(f"DQ score between {v1:.0f}–{float(v2):.0f}")

        # ── 4. Anomaly / unusual filter ───────────────────────────────────────────
        anom_words = {"unusual","anomal","anomaly","anomalous","outlier","outliers","flagged","suspicious","weird","strange","risky","risk"}
        if any(w in q for w in anom_words):
            if "is_anomaly" in df.columns:
                mask &= df["is_anomaly"] == True
                summary_parts.append("flagged as anomalous by AI")
            elif "isolation_score" in df.columns:
                mask &= df["isolation_score"] >= 70
                summary_parts.append("AI risk score ≥ 70")

        # ── 5. Duplicate records ──────────────────────────────────────────────────
        dup_words = {"duplicate","duplicates","duplicated","dup","dups","repeated"}
        if any(w in q for w in dup_words):
            dup_col = next((c for c in id_cols if c in df.columns), None)
            if dup_col:
                dupes = df[dup_col].duplicated(keep=False)
                mask &= dupes
                summary_parts.append(f"duplicate {dup_col}")
            elif "dq_score_uniqueness" in df.columns:
                mask &= df["dq_score_uniqueness"] < 50
                summary_parts.append("uniqueness score < 50")

        # ── 6. Missing / null values ──────────────────────────────────────────────
        miss_words = {"missing","null","nulls","empty","blank","nan","incomplete"}
        if any(w in q for w in miss_words) and not any(w in q for w in ("completeness","score")):
            null_mask = df.isnull().any(axis=1)
            mask &= null_mask
            summary_parts.append("has missing values")

        # ── 7. Dimension weakness filter ─────────────────────────────────────────
        dim_hits = [(k, _DIM_MAP[k]) for k in _DIM_MAP if k in q]
        if dim_hits:
            thresh_m = _re.search(r'(below|under|<|less than)\s*([\d\.]+)', q)
            thresh = float(thresh_m.group(2)) if thresh_m else 70.0
            for kw, col in dim_hits:
                if col in df.columns:
                    mask &= df[col] < thresh
                    summary_parts.append(f"{_DIM_HUMAN[col]} score < {thresh:.0f}")

        # ── 8. Numeric column value filter  (amount > 50000, price < 100) ─────────
        # Expanded regex: captures "higher than", "more than", "lower than" etc.
        num_m = _re.search(
            r'([\w_]+(?:\s+[\w_]+){0,3})\s*(?:is\s+|are\s+|was\s+)?(>=|<=|>|<|=|==|above|below|over|under|'
            r'higher than|lower than|more than|less than|greater than|'
            r'bigger than|smaller than|exceeds|exceed)\s*([\d,\.]+)', q)
        if num_m:
            # Strip any leading stopwords the greedy regex may have consumed
            # e.g. "where session duration sec" → "session_duration_sec"
            _raw_hint = num_m.group(1).strip().split()
            while _raw_hint and _raw_hint[0].lower() in _STOPWORDS:
                _raw_hint.pop(0)
            col_hint = "_".join(_raw_hint)
            op_str      = num_m.group(2).strip()
            val         = float(num_m.group(3).replace(",",""))
            # Match col_hint against actual column names — try substring both ways
            matched_col = next(
                (c for c in df.columns
                 if col_hint and (col_hint in c.lower() or c.lower() in col_hint
                                  or col_hint.replace("_"," ") in c.lower().replace("_"," "))),
                None)
            if matched_col and pd.api.types.is_numeric_dtype(df[matched_col]):
                op_map = {
                    ">":">",">=":">=","<":"<","<=":"<=","=":"==","==":"==",
                    "above":">","over":">","greater than":">",
                    "higher than":">","more than":">","bigger than":">","exceeds":">","exceed":">",
                    "below":"<","under":"<","less than":"<",
                    "lower than":"<","smaller than":"<",
                }
                op = op_map.get(op_str, ">")
                if op == ">":    mask &= df[matched_col].fillna(-np.inf) > val
                elif op == ">=": mask &= df[matched_col].fillna(-np.inf) >= val
                elif op == "<":  mask &= df[matched_col].fillna(np.inf)  < val
                elif op == "<=": mask &= df[matched_col].fillna(np.inf)  <= val
                elif op == "==": mask &= df[matched_col] == val
                summary_parts.append(f"{matched_col} {op} {val:,.0f}")

        # ── 9. Categorical value filter  (vendor = 'Accenture', status = paid) ───
        cat_m = _re.search(r'([\w_]+)\s*(?:=|is|equals|called|named)\s*["\']?([\w\s]+)["\']?', q)
        if cat_m:
            col_hint = cat_m.group(1).replace(" ","_")
            val_hint = cat_m.group(2).strip()
            matched_col = next(
                (c for c in df.columns if col_hint in c.lower() or c.lower() in col_hint), None)
            if matched_col and matched_col not in ("dq_severity","dq_grade","is_anomaly"):
                col_vals = df[matched_col].astype(str).str.lower()
                hit = col_vals.str.contains(val_hint, na=False)
                if hit.sum() > 0:
                    mask &= hit
                    summary_parts.append(f"{matched_col} contains '{val_hint}'")

        # ── 9b. Date-range filter  (v2.6) ─────────────────────────────────────────
        # Supports: "after 2023-06-01", "before 2024-01-01", "since 2023-01-01",
        #           "between 2023-01-01 and 2023-12-31", "in January 2024", "in 2024".
        # Date columns are stored as "%Y-%m-%d" strings post-cleaning → re-parse here.
        _present_date_cols = [c for c in col_mapping.get("date_columns", []) if c in df.columns]
        if _present_date_cols:
            _dcol = next((c for c in _present_date_cols
                          if c.lower() in q or c.split("_")[0].lower() in q),
                         _present_date_cols[0])
            _dser = pd.to_datetime(df[_dcol], errors="coerce")
            _MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
                       "july":7,"august":8,"september":9,"october":10,"november":11,
                       "december":12,"jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,
                       "aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}
            _date_applied = False

            _btw = _re.search(r'between\s+(\d{4}-\d{1,2}-\d{1,2})\s+and\s+(\d{4}-\d{1,2}-\d{1,2})', q)
            _aft = _re.search(r'(?:after|since|from|starting)\s+(\d{4}-\d{1,2}-\d{1,2})', q)
            _bef = _re.search(r'(?:before|until|till|up to|by)\s+(\d{4}-\d{1,2}-\d{1,2})', q)
            _my  = _re.search(r'\b([a-z]+)\s+(\d{4})\b', q)
            _yr  = _re.search(r'\b(?:in|during|for|year)\s+(\d{4})\b', q)

            if _btw:
                _d1 = pd.to_datetime(_btw.group(1), errors="coerce")
                _d2 = pd.to_datetime(_btw.group(2), errors="coerce")
                if pd.notna(_d1) and pd.notna(_d2):
                    _lo, _hi = min(_d1, _d2), max(_d1, _d2)
                    mask &= _dser.between(_lo, _hi)
                    summary_parts.append(f"{_dcol} between {_lo.date()} and {_hi.date()}")
                    _date_applied = True
            if not _date_applied and (_aft or _bef):
                if _aft:
                    _d1 = pd.to_datetime(_aft.group(1), errors="coerce")
                    if pd.notna(_d1):
                        mask &= _dser >= _d1
                        summary_parts.append(f"{_dcol} on/after {_d1.date()}")
                        _date_applied = True
                if _bef:
                    _d2 = pd.to_datetime(_bef.group(1), errors="coerce")
                    if pd.notna(_d2):
                        mask &= _dser <= _d2
                        summary_parts.append(f"{_dcol} on/before {_d2.date()}")
                        _date_applied = True
            if not _date_applied and _my and _my.group(1) in _MONTHS:
                _mo = _MONTHS[_my.group(1)]; _yy = int(_my.group(2))
                mask &= (_dser.dt.year == _yy) & (_dser.dt.month == _mo)
                summary_parts.append(f"{_dcol} in {_my.group(1).title()} {_yy}")
                _date_applied = True
            if not _date_applied and _yr:
                _yy = int(_yr.group(1))
                mask &= _dser.dt.year == _yy
                summary_parts.append(f"{_dcol} in {_yy}")
                _date_applied = True

    # ── 10. Aggregation: which column has most issues ──────────────────────────
    # Only trigger on explicit grouping language — NOT on "top/worst/best" (those = limit)
    AGG_TRIGGERS = {"which","who","breakdown","group"}
    _FREQ_AGG_PHRASES = {"most used","most common","most popular","most frequent",
                         "highest used","most occurrences","used most","commonly used"}
    # Check raw query q (not toks) — "which/who" are stripped by _STOPWORDS
    # IMPORTANT: freq phrases ("highest used" etc.) must NOT steal "Top N entity" queries —
    # those belong to block 11 which handles them with correct count-sort logic
    _has_top_n   = bool(_re.search(r'top\s+\d+', q))
    agg_phrase   = any(w in q.split() for w in AGG_TRIGGERS) or (
        any(p in q for p in _FREQ_AGG_PHRASES) and not _has_top_n
    )
    # Also catch "by <col>" pattern explicitly
    by_m = _re.search(r'\bby\s+([\w_]+)', q)
    if agg_phrase or (by_m and not _re.search(r'top\s+\d+', q)):
        grp_col = None
        # If "by <col>" explicitly stated, try to match that column
        if by_m:
            by_hint = by_m.group(1)
            grp_col = next((c for c in cat_cols if c in df.columns and
                            (by_hint in c.lower() or c.lower() in by_hint)), None)
        # Fallback: find a categorical column mentioned in the query
        if not grp_col:
            grp_col = next((c for c in cat_cols if c in df.columns and
                            any(hint in q for hint in [c.lower(), c.split("_")[0].lower()])), None)
        # Only aggregate if we found an explicit group column — never silently default
        if grp_col:
            intent = "aggregate"
            agg_df = (
                df[mask].groupby(grp_col, dropna=False)
                .agg(
                    records=("dq_score","count"),
                    avg_score=("dq_score","mean"),
                    critical=(
                        "dq_severity",
                        lambda s: (s == "CRITICAL").sum()
                    ),
                )
                .reset_index()
                .rename(columns={grp_col: "Value", "records":"Records",
                                  "avg_score":"Avg DQ Score","critical":"Critical Issues"})
                .sort_values("Avg DQ Score")
            )
            agg_df["Avg DQ Score"] = agg_df["Avg DQ Score"].round(1)
            summary_parts.append(f"grouped by {grp_col}")

    # ── 11. Top N ─────────────────────────────────────────────────────────────
    top_m = _re.search(r'top\s+(\d+)', q)
    # Entity words: when "top N [entities]" refers to groups, not individual records
    # Covers all 6 demo domains + generic groupable nouns
    _ENTITY_WORDS = {
        # Finance / generic
        "companies","company","vendors","vendor","reps","rep","customers","customer",
        "products","product","suppliers","supplier","teams","team","regions","region",
        "departments","department","categories","category","channels","channel",
        "accounts","account","clients","client","partners","partner",
        # Sales / CRM
        "stages","stage","sources","source","leads","lead",
        # HR / People
        "employees","employee","managers","manager","locations","location",
        "titles","title","roles","role","ratings","rating",
        # E-commerce / Supply-chain
        "orders","order","shipments","shipment","warehouses","warehouse",
        "carriers","carrier","methods","method","countries","country",
        "origins","origin","statuses","status",
        # Product Analytics
        "platforms","platform","events","event","features","feature",
        "tiers","tier","types","type","users","user","sessions","session",
        "pages","page","devices","device","browsers","browser",
    }
    # Detect frequency-intent: "most used", "most common", "most popular",
    # "highest used", "most frequent", "popular", "common", "used most"
    _FREQ_PHRASES = {"most used","most common","most popular","most frequent",
                     "highest used","used most","popular","commonly used","frequently used"}
    freq_intent = any(p in q for p in _FREQ_PHRASES)

    if top_m:
        limit = int(top_m.group(1))
        toks_set = set(_tokens(q))
        # Also check raw q.split() for entity words (some are stripped by stopwords)
        raw_toks = set(w.lower() for w in q.split())
        entity_hit = (toks_set | raw_toks) & _ENTITY_WORDS
        if entity_hit and intent != "aggregate":
            # "Top 10 highest used platforms" / "Top 5 companies by revenue"
            entity_word = entity_hit.pop()
            # Match to a categorical column — try entity word against col name
            grp_col = next(
                (c for c in cat_cols if c in df.columns and
                 any(h in c.lower() for h in [entity_word, entity_word.rstrip("s"),
                                               entity_word + "s"])),
                None
            )
            # Fallback: find first cat col whose name appears anywhere in the query
            if not grp_col:
                grp_col = next(
                    (c for c in cat_cols if c in df.columns and
                     (c.lower() in q or c.split("_")[0].lower() in q)),
                    None
                )
            if not grp_col and cat_cols:
                grp_col = next((c for c in cat_cols if c in df.columns), None)
            if grp_col:
                # Determine sort metric:
                # frequency-intent  → sort by Records (count) DESC
                # explicit num col  → sort by that col avg DESC
                # default           → sort by Avg DQ Score ASC (worst first)
                if freq_intent:
                    metric_col  = "dq_score"
                    sort_by_col = "Records"
                    sort_asc_agg = False
                elif num_m and matched_col and matched_col in df.columns and pd.api.types.is_numeric_dtype(df[matched_col]):
                    metric_col  = matched_col
                    sort_by_col = f"Avg {metric_col.replace('_',' ').title()}"
                    sort_asc_agg = False
                else:
                    metric_col  = "dq_score"
                    sort_by_col = "Avg DQ Score"
                    sort_asc_agg = True   # worst-first (lowest score = most issues)

                intent = "aggregate"
                agg_df = (
                    df[mask].groupby(grp_col, dropna=False)
                    .agg(
                        records=(metric_col, "count"),
                        avg_metric=(metric_col, "mean"),
                        avg_score=("dq_score", "mean"),
                        critical=("dq_severity", lambda s: (s=="CRITICAL").sum()),
                    )
                    .reset_index()
                    .rename(columns={
                        grp_col: "Value",
                        "records": "Records",
                        "avg_metric": f"Avg {metric_col.replace('_',' ').title()}",
                        "avg_score": "Avg DQ Score",
                        "critical": "Critical Issues",
                    })
                    .sort_values(sort_by_col, ascending=sort_asc_agg)
                    .head(limit)
                )
                agg_df[f"Avg {metric_col.replace('_',' ').title()}"] = (
                    agg_df[f"Avg {metric_col.replace('_',' ').title()}"].round(1))
                freq_label = "most used " if freq_intent else ""
                summary_parts.append(f"top {limit} {freq_label}{entity_word} by {'count' if freq_intent else metric_col.replace('_',' ')}")
                limit = None  # already sliced in agg_df
        else:
            # Records-based top N — sort by relevant numeric col or dq_score
            if num_m and matched_col and matched_col in df.columns:
                sort_col = matched_col
                sort_asc = False
            else:
                sort_col = "dq_score"
                sort_asc = True

    # ── 12. Sort direction hints ──────────────────────────────────────────────
    if any(w in q for w in ("worst","lowest","lowest score","bad")):
        sort_col = sort_col or "dq_score"; sort_asc = True
    if any(w in q for w in ("best","highest","highest score")):
        sort_col = sort_col or "dq_score"; sort_asc = False
    if "risk" in q or "risky" in q:
        if "isolation_score" in df.columns:
            sort_col = "isolation_score"; sort_asc = False

    # ── Build final summary ───────────────────────────────────────────────────
    n = int(mask.sum())
    if not summary_parts:
        if n == len(df):
            summary_txt = f"Showing all {len(df):,} records — try a more specific query."
        else:
            summary_txt = f"Found {n:,} records."
    else:
        criteria = " · ".join(summary_parts)
        summary_txt = f"Found **{n:,} records** matching: {criteria}."

    if limit:
        summary_txt += f" Showing top {limit}."

    return {
        "mask":     mask,
        "summary":  summary_txt,
        "intent":   intent,
        "sort_col": sort_col,
        "sort_asc": sort_asc,
        "agg_df":   agg_df,
        "limit":    limit,
        "n":        n,
    }

NLQ_EXAMPLES = [
    "Show critical records",
    "Find anomalous records",
    "Records with score below 60",
    "Top 20 worst records",
    "Show records with missing values",
    "Find duplicate records",
    "High severity with low completeness",
    "Which vendors have the most issues",
    "Show high risk anomalies",
    "Records with format accuracy below 50",
    "Grade F records",
    "Show flagged records sorted by risk",
    "High or critical records",
]

def _dynamic_examples(df: pd.DataFrame, col_mapping: dict) -> list:
    """
    Generate context-aware NLQ queries based on the actual uploaded dataset.
    Every query here is verified to match the NLQ parser's patterns — no dead ends.
    """
    examples = []
    num_cols  = col_mapping.get("numeric_columns", [])
    cat_cols  = col_mapping.get("categorical_columns", [])
    date_cols = col_mapping.get("date_columns", [])

    # ── Tier 1: DQ queries that always work (engine-native) ──────────────────
    examples += [
        "Show critical records",
        "Find anomalous records",
        "Grade F records",
        "Records with score below 60",
    ]

    # ── Tier 2: Categorical column aggregations (uses "which … has most issues")
    # Pattern: "Which <col> has the most issues" → triggers AGG_TRIGGERS via "which"
    for col in cat_cols[:3]:
        if col in df.columns:
            clean = col.replace("_", " ")
            examples.append(f"Which {clean} has the most issues")

    # ── Tier 3: Top-N frequency aggregation (uses "Top N most used <col>")
    # Pattern: "Top N most used <col>" → entity_hit + freq_intent
    for col in cat_cols[:2]:
        if col in df.columns:
            # Use singular form as the "entity word" — engine strips trailing s
            entity = col.replace("_", " ").rstrip("s")
            n_unique = df[col].nunique()
            top_n = min(5, max(3, n_unique // 2))
            examples.append(f"Top {top_n} most used {entity}")

    # ── Tier 4: Numeric threshold queries (uses actual p75 / p25 values)
    # Pattern: "Show records where <col> is above/below <value>"
    for col in num_cols[:2]:
        if col in df.columns:
            num_series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(num_series) > 5:
                clean = col.replace("_", " ")
                try:
                    q75 = num_series.quantile(0.75)
                    # Only emit if value is sensible (positive, non-trivial)
                    if q75 > 0:
                        examples.append(f"Records where {clean} is above {int(q75):,}")
                except Exception:
                    pass

    # ── Tier 4b: Multi-condition AND/OR (v2.7)
    if num_cols:
        _c0 = num_cols[0]
        _s = pd.to_numeric(df[_c0], errors="coerce").dropna()
        if len(_s) > 5:
            try:
                _q75 = _s.quantile(0.75)
                if _q75 > 0:
                    examples.append(
                        f"Critical records and {_c0.replace('_',' ')} above {int(_q75):,}")
            except Exception:
                pass
    examples.append("High or critical records")

    # ── Tier 5: Date / null queries + date-range (v2.6)
    for col in date_cols[:1]:
        examples.append(f"Show records with missing {col.replace('_', ' ')}")
        try:
            _ds = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(_ds) > 5:
                _yr = int(_ds.dt.year.median())
                examples.append(f"Records after {_yr}-01-01")
        except Exception:
            pass

    # ── Tier 6: Always-useful fallbacks ──────────────────────────────────────
    examples += [
        "Show records with missing values",
        "Top 20 worst records",
        "Find duplicate records",
    ]

    # Deduplicate while preserving order, cap at 12
    seen = set(); out = []
    for ex in examples:
        key = ex.lower().strip()
        if key not in seen:
            seen.add(key); out.append(ex)
        if len(out) >= 12:
            break
    return out

COL_TYPE_META = {
    "numeric":     {"icon": "🔢", "label": "Numeric",     "color": "#378ADD"},
    "date":        {"icon": "📅", "label": "Date",        "color": "#1D9E75"},
    "categorical": {"icon": "🏷️",  "label": "Categorical", "color": "#E24B4A"},
    "id":          {"icon": "🆔", "label": "ID / Key",    "color": "#534AB7"},
    "text":        {"icon": "📝", "label": "Free text",   "color": "#BA7517"},
    "boolean":     {"icon": "☑️",  "label": "Boolean",    "color": "#639922"},
}

SEV_ORDER  = ["CLEAN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

SEV_COLORS = {"CLEAN": "#639922", "LOW": "#EF9F27",
              "MEDIUM": "#BA7517", "HIGH": "#D85A30", "CRITICAL": "#E24B4A"}

SEV_LABELS = {
    "CLEAN":    "✅ Clean — Ready to use",
    "LOW":      "🟡 Low — Worth monitoring",
    "MEDIUM":   "🟠 Medium — Schedule a fix",
    "HIGH":     "🔶 High — Review today",
    "CRITICAL": "🔴 Critical — Act now",
}

DIM_COLORS = {
    "Missing Data":     "#378ADD",
    "Format Accuracy":  "#E24B4A",
    "Value Accuracy":   "#3B6D11",
    "Data Consistency": "#1D9E75",
    "Duplicate Check":  "#534AB7",
}

PLACEHOLDER_VALUES = {"UNKNOWN", "N/A", "NA", "NONE", "", "NULL", "NAN", "-"}

DEMO_DOMAINS = {
    "💰 Finance / Invoices":    "finance",
    "📈 Sales / CRM":           "sales",
    "👥 HR / People":           "hr",
    "🛒 E-commerce / Orders":   "ecommerce",
    "🚚 Supply Chain":          "supply_chain",
    "📱 Product Analytics":     "product",
}

_VENDORS    = ["Accenture Ltd","  KPMG Advisory  ","Deloitte & Touche","ernst & young",
               "PwC Services","GARTNER INC","Infosys BPO","Wipro Technologies",
               "IBM Global  ","Capgemini SE","  Oracle Corp","SAP AG",None,"N/A","UNKNOWN"]

_CURRENCIES = ["USD","EUR","GBP","AUD","INR","XYZ","usd","Eur",None]

_INV_STATUS = ["PAID","PENDING","OVERDUE","CANCELLED","DRAFT","paid","  Pending  ",None,""]

_SALES_REPS   = ["Alice Johnson","Bob Smith","  carol white  ","DAVID LEE","Emma Brown",
                 "Frank Zhang","grace kim","Hina Patel",None,"UNKNOWN REP"]

_COMPANIES    = ["TechCorp Inc","  Globex Ltd","Initech","Umbrella Corp","Hooli",
                 "Pied Piper","Initrode","ACME Corp","Dunder Mifflin",None]

_DEAL_STAGES  = ["Prospecting","Qualification","Proposal","Negotiation","Closed Won",
                 "Closed Lost","closed won","PROPOSAL","discovery",None]

_LEAD_SOURCES = ["Website","Cold Call","Referral","LinkedIn","Trade Show","EMAIL","website",None]

_DEPARTMENTS  = ["Engineering","Marketing","Sales","Finance","HR","Operations",
                 "engineering","MARKETING","Legal",None]

_JOB_TITLES   = ["Software Engineer","Senior Engineer","Manager","Director","Analyst",
                 "Associate","VP","  intern  ","CONTRACTOR",None]

_EMP_STATUS   = ["Active","Inactive","On Leave","Terminated","active","ACTIVE",None,""]

_LOCATIONS    = ["New York","London","Bangalore","Singapore","Sydney","  Berlin  ","REMOTE",None]

_PRODUCTS     = ["Laptop Pro 15","Wireless Mouse","USB-C Hub","Mechanical Keyboard",
                 "Monitor 27in","Webcam HD","laptop pro 15","  USB-C Hub  ",None]

_CATEGORIES   = ["Electronics","Accessories","Peripherals","Storage","Networking","electronics",None]

_ORDER_STATUS = ["Delivered","Shipped","Processing","Cancelled","Returned",
                 "delivered","SHIPPED","pending",None,""]

_PAYMENT_METHODS = ["Credit Card","PayPal","Bank Transfer","Crypto","credit card","PAYPAL",None]

_SUPPLIERS    = ["FastShip Co","  Global Freight  ","QuickLog","PrimeMover","CargoExpress",
                 "fastship co","QUICKLOG",None,"UNKNOWN SUPPLIER"]

_WAREHOUSES   = ["WH-EAST","WH-WEST","WH-NORTH","WH-SOUTH","WH-CENTRAL","wh-east",None]

_SHIP_STATUS  = ["On Time","Delayed","In Transit","Delivered","Lost","on time","DELAYED",None,""]

_CARRIERS     = ["FedEx","DHL","UPS","USPS","DPD","fedex","  DHL  ",None]

_FEATURES     = ["Dashboard","Search","Export","Upload","Settings","Notifications",
                 "Report Builder","API","dashboard","SEARCH",None]

_EVENTS       = ["page_view","click","form_submit","download","signup","login",
                 "PAGE_VIEW","Click","form submit",None,""]

_PLATFORMS    = ["web","mobile_ios","mobile_android","desktop","WEB","Mobile",None]

_USER_TIERS   = ["free","pro","enterprise","Free","PRO","ENTERPRISE",None,""]

# ── Cleaning / scoring configuration ─────────────────────────────────────────
# The engine reads a module-level ``CONFIG`` at call time. The UI layer builds
# its own settings from the sidebar and applies them with ``configure(**kw)``
# before running the pipeline; headless callers just use these defaults.
DEFAULT_CONFIG = {
    "null_fill_string":         "UNKNOWN",
    "null_fill_numeric":        0.0,
    "min_amount":               0.0,
    "max_amount":               10_000_000.0,
    "outlier_zscore_threshold": 3.0,
    "duplicate_subset":         "",
}
CONFIG = dict(DEFAULT_CONFIG)

def configure(**overrides):
    """Apply UI/user overrides onto the engine CONFIG (unknown keys ignored)."""
    for k, v in overrides.items():
        if k in CONFIG:
            CONFIG[k] = v
    return CONFIG

def _rnd_date(start, end):
    d   = start + timedelta(days=random.randint(0, (end - start).days))
    fmt = random.choice(["%Y-%m-%d"]*5 + ["%d/%m/%Y", "%m-%d-%Y", "bad-date", ""])
    return None if fmt == "" else d.strftime(fmt)

def generate_demo(n: int = 300, seed: int = 42, domain: str = "finance") -> pd.DataFrame:
    """Generate realistic messy demo data for the chosen domain.
    seed — same value = same dataset every run (reproducible). Change it for fresh data.
    """
    random.seed(seed); np.random.seed(seed)
    s, e = date(2023, 1, 1), date(2025, 12, 31)
    rows = []

    if domain == "finance":
        for i in range(1, n + 1):
            amt = round(random.uniform(-500, 250_000), 2)
            if random.random() < 0.05: amt = round(random.uniform(1_000_000, 15_000_000), 2)
            if random.random() < 0.08: amt = -abs(amt)
            tax = round(amt * random.uniform(0.05, 0.18), 2) if amt > 0 else None
            rows.append({
                "invoice_id":     i,
                "invoice_number": f"INV-{random.randint(1000,9999)}",
                "vendor_name":    random.choice(_VENDORS),
                "invoice_date":   _rnd_date(s, e),
                "due_date":       _rnd_date(s, e),
                "payment_date":   _rnd_date(s, e) if random.random() > 0.35 else None,
                "amount":         amt if random.random() > 0.05 else None,
                "tax_amount":     tax,
                "total_amount":   round(amt + (tax or 0), 2) if random.random() > 0.05 else None,
                "currency":       random.choice(_CURRENCIES),
                "status":         random.choice(_INV_STATUS),
                "department":     random.choice(["Finance","IT","HR","Operations",None]),
                "cost_centre":    random.choice(["CC100","CC200","CC300","CC999",None]),
            })

    elif domain == "sales":
        for i in range(1, n + 1):
            deal_val = round(random.uniform(1_000, 500_000), 2)
            rows.append({
                "deal_id":            f"DEAL-{i:05d}",
                "company_name":       random.choice(_COMPANIES),
                "sales_rep":          random.choice(_SALES_REPS),
                "deal_stage":         random.choice(_DEAL_STAGES),
                "deal_value":         deal_val if random.random() > 0.07 else None,
                "close_probability":  round(random.uniform(0,100)) if random.random() > 0.1 else None,
                "expected_close_date":_rnd_date(s, e),
                "created_date":       _rnd_date(s, e),
                "lead_source":        random.choice(_LEAD_SOURCES),
                "num_employees":      random.choice([10,50,200,500,1000,5000,None,0,-1]),
                "annual_revenue":     round(random.uniform(100_000,50_000_000),2) if random.random()>0.15 else None,
                "currency":           random.choice(_CURRENCIES),
                "region":             random.choice(["North America","Europe","APAC","LATAM","MEA",None]),
            })

    elif domain == "hr":
        for i in range(1, n + 1):
            salary = round(random.uniform(30_000, 250_000), 2)
            rows.append({
                "employee_id":      f"EMP-{i:05d}",
                "full_name":        random.choice(["  John Smith  ","Jane Doe","ROBERT JOHNSON",
                                                    "Maria Garcia","Wei Zhang","Priya Sharma",
                                                    "james wilson","Emily Brown",None]),
                "department":       random.choice(_DEPARTMENTS),
                "job_title":        random.choice(_JOB_TITLES),
                "hire_date":        _rnd_date(date(2010,1,1), date(2025,12,31)),
                "termination_date": _rnd_date(s, e) if random.random() < 0.12 else None,
                "base_salary":      salary if random.random() > 0.06 else None,
                "bonus_pct":        round(random.uniform(0,30),1) if random.random()>0.2 else None,
                "location":         random.choice(_LOCATIONS),
                "status":           random.choice(_EMP_STATUS),
                "performance_rating": random.choice([1,2,3,4,5,None,0,6,"N/A"]),
                "manager_id":       f"EMP-{random.randint(1,50):05d}" if random.random()>0.1 else None,
            })

    elif domain == "ecommerce":
        for i in range(1, n + 1):
            qty   = random.choice([1,1,1,2,3,5,10,0,-1,None])
            price = round(random.uniform(5, 2_000), 2)
            rows.append({
                "order_id":       f"ORD-{i:06d}",
                "customer_id":    f"CUST-{random.randint(1,int(n*0.6)):05d}",
                "product_name":   random.choice(_PRODUCTS),
                "category":       random.choice(_CATEGORIES),
                "order_date":     _rnd_date(s, e),
                "ship_date":      _rnd_date(s, e) if random.random()>0.15 else None,
                "quantity":       qty,
                "unit_price":     price if random.random()>0.05 else None,
                "total_price":    round((qty or 0)*price,2) if qty and price and random.random()>0.08 else None,
                "discount_pct":   round(random.uniform(0,50),1) if random.random()>0.6 else 0,
                "order_status":   random.choice(_ORDER_STATUS),
                "payment_method": random.choice(_PAYMENT_METHODS),
                "country":        random.choice(["US","UK","DE","IN","AU","CA","SG","FR","  US  ",None]),
            })

    elif domain == "supply_chain":
        for i in range(1, n + 1):
            qty_ord = random.randint(10, 5000)
            rows.append({
                "shipment_id":          f"SHP-{i:06d}",
                "supplier_name":        random.choice(_SUPPLIERS),
                "carrier":              random.choice(_CARRIERS),
                "warehouse":            random.choice(_WAREHOUSES),
                "order_date":           _rnd_date(s, e),
                "expected_date":        _rnd_date(s, e),
                "actual_delivery_date": _rnd_date(s, e) if random.random()>0.25 else None,
                "quantity_ordered":     qty_ord,
                "quantity_received":    qty_ord-random.randint(0,50) if random.random()>0.1 else None,
                "unit_cost":            round(random.uniform(0.5,500),2) if random.random()>0.06 else None,
                "total_cost":           round(qty_ord*random.uniform(0.5,500),2) if random.random()>0.08 else None,
                "shipment_status":      random.choice(_SHIP_STATUS),
                "country_origin":       random.choice(["China","India","Germany","USA","Vietnam","  China  ",None]),
            })

    elif domain == "product":
        session_ids = [f"sess_{random.randint(100000,999999)}" for _ in range(max(50, n//3))]
        user_ids    = [f"usr_{random.randint(10000,99999)}"    for _ in range(max(30, n//4))]
        for i in range(1, n + 1):
            dur = random.randint(1, 900) if random.random()>0.1 else None
            rows.append({
                "event_id":           f"EVT-{i:07d}",
                "user_id":            random.choice(user_ids + [None]*3),
                "session_id":         random.choice(session_ids),
                "event_type":         random.choice(_EVENTS),
                "feature_name":       random.choice(_FEATURES),
                "platform":           random.choice(_PLATFORMS),
                "event_timestamp":    _rnd_date(s, e),
                "session_duration_sec": dur,
                "page_load_ms":       random.randint(50,8000) if random.random()>0.12 else None,
                "user_tier":          random.choice(_USER_TIERS),
                "country":            random.choice(["US","UK","IN","DE","BR","FR","CA",None,"XX"]),
                "is_conversion":      random.choice([True,False,None,"true","FALSE"]),
            })

    df = pd.DataFrame(rows)

    # ── Guarantee dirty data: inject known-bad rows ───────────────────────────
    # Identify numeric columns (excluding ID/index column)
    _num_cols = [c for c in df.columns
                 if df[c].dtype in [float, int] and c != df.columns[0]]
    _str_cols = [c for c in df.columns if df[c].dtype == object]
    _date_cols = [c for c in df.columns
                  if any(h in c.lower() for h in ("date","time","day","month","year"))]

    # 1. Duplicates (~7%)
    n_dupes = max(8, int(len(df) * 0.07))
    dupes   = df.sample(min(n_dupes, len(df)), random_state=seed+1).copy()

    # 2. All-null rows (2% of n) — forces completeness = 0 → CRITICAL
    null_rows = pd.DataFrame([{c: None for c in df.columns}
                               for _ in range(max(5, n // 40))])

    # 3. CRITICAL accuracy rows — large NEGATIVE values on ALL numeric cols
    #    This triggers: _negative + _outlier flags on every num col
    #    With N num cols: penalty = (neg_pen + out_pen) × N >> 70 → accuracy < 30 → CRITICAL
    n_crit = max(15, n // 20)          # ~5% guaranteed CRITICAL rows
    crit_rows = df.sample(min(n_crit, len(df)), random_state=seed+2).copy()
    for col in _num_cols:
        # Large negative absolute value: triggers _negative AND _outlier
        crit_rows[col] = -999_999_999.0

    # 4. Consistency-violation rows — inverted date pairs (end before start)
    #    Forces dq_score_consistency < 40 → CRITICAL
    n_cons = max(10, n // 30)
    cons_rows = df.sample(min(n_cons, len(df)), random_state=seed+5).copy()
    # Swap every pair of date columns to force ordering violations
    if len(_date_cols) >= 2:
        for i in range(0, len(_date_cols) - 1, 2):
            c1, c2 = _date_cols[i], _date_cols[i + 1]
            # Set c1 to a far-future date and c2 to a far-past date → consistency violation
            cons_rows[c1] = "2099-12-31"
            cons_rows[c2] = "2000-01-01"
    elif len(_date_cols) == 1:
        cons_rows[_date_cols[0]] = "9999-99-99"  # unparseable → validity hit
    # Also make these rows have negative numerics → CRITICAL via accuracy too
    for col in _num_cols:
        cons_rows[col] = -999_999.0

    # 5. Empty-string / placeholder rows — forces validity failures
    blank_rows = df.sample(min(8, len(df)), random_state=seed+3).copy()
    for col in _str_cols[:4]:
        blank_rows[col] = random.choice(["", "  ", "NULL", "N/A", "UNKNOWN", "???"])
    for col in _num_cols[:2]:
        blank_rows[col] = None   # null numeric → completeness hit

    # 6. Mixed-type / garbage rows — format accuracy failures
    bad_rows = df.sample(min(6, len(df)), random_state=seed+4).copy()
    for col in _num_cols:
        bad_rows[col] = -abs(bad_rows[col].fillna(0)) * 100  # negative, also outlier

    df = (pd.concat([df, dupes, null_rows, crit_rows, cons_rows, blank_rows, bad_rows],
                    ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True))

    # Reset ID column to sequential
    first_col = df.columns[0]
    df[first_col] = range(1, len(df)+1)
    return df

def detect_col_types(df: pd.DataFrame) -> dict:
    """
    Auto-detect semantic type for every column.
    Returns {col_name: 'numeric'|'date'|'categorical'|'id'|'text'|'boolean'}
    No domain knowledge required — works purely from data shape.
    """
    types = {}
    n = max(len(df), 1)

    for col in df.columns:
        s = df[col]

        # Already a datetime dtype
        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "date"; continue

        # Numeric dtype
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique() <= 2 and set(s.dropna().unique()).issubset({0, 1}):
                types[col] = "boolean"
            elif s.nunique() / n > 0.90 and pd.api.types.is_integer_dtype(s):
                types[col] = "id"
            else:
                types[col] = "numeric"
            continue

        # Object / string columns
        sample = s.dropna().head(200).astype(str)
        if len(sample) == 0:
            types[col] = "text"; continue

        # Boolean-like strings
        uniq_lower = {v.lower().strip() for v in sample.unique()}
        if uniq_lower.issubset({"true","false","yes","no","1","0","y","n","t","f"}):
            types[col] = "boolean"; continue

        # Date strings — try parsing a sample
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.70:
                types[col] = "date"; continue
        except Exception:
            pass

        # Cardinality-based split
        nunique      = s.nunique()
        unique_ratio = nunique / n
        avg_len      = sample.str.len().mean()

        if unique_ratio > 0.85 and avg_len < 40:
            types[col] = "id"
        elif nunique <= max(30, n * 0.05):
            types[col] = "categorical"
        elif avg_len > 60:
            types[col] = "text"
        else:
            types[col] = "categorical"

    return types

def map_columns_to_roles(df: pd.DataFrame, detected_types: dict) -> dict:
    """
    Maps columns to pipeline roles based purely on detected type.
    No domain knowledge needed.
    Returns:
      {
        'date_columns':        [...],
        'numeric_columns':     [...],
        'id_columns':          [...],
        'categorical_columns': [...],
        'text_columns':        [...],
        'critical_fields':     [...],   # highest-importance cols for completeness
      }
    """
    mapping = {
        "date_columns":        [],
        "numeric_columns":     [],
        "id_columns":          [],
        "categorical_columns": [],
        "text_columns":        [],
        "boolean_columns":     [],
    }
    for col, dtype in detected_types.items():
        if   dtype == "date":        mapping["date_columns"].append(col)
        elif dtype == "numeric":     mapping["numeric_columns"].append(col)
        elif dtype == "id":          mapping["id_columns"].append(col)
        elif dtype == "categorical": mapping["categorical_columns"].append(col)
        elif dtype == "text":        mapping["text_columns"].append(col)
        elif dtype == "boolean":     mapping["boolean_columns"].append(col)

    # Critical fields = IDs + first 3 numerics + first 3 categoricals
    mapping["critical_fields"] = (
        mapping["id_columns"][:3]
        + mapping["numeric_columns"][:3]
        + mapping["categorical_columns"][:3]
    )
    return mapping

def profile_dataframe(df: pd.DataFrame, types_json: str) -> list:
    """
    Per-column statistical profile.
    types_json: JSON-serialised detected_types dict (for cache key).
    """
    detected_types = json.loads(types_json)
    n = max(len(df), 1)
    profiles = []

    for col in df.columns:
        s     = df[col]
        dtype = detected_types.get(col, "text")
        null_count = int(s.isna().sum())
        nunique    = int(s.nunique())

        p = {
            "column":       col,
            "type":         dtype,
            "null_pct":     round(null_count / n * 100, 1),
            "null_count":   null_count,
            "unique_count": nunique,
            "unique_pct":   round(nunique / n * 100, 1),
            "sample":       " · ".join(str(v) for v in s.dropna().unique()[:4]),
            # numeric extras (filled below if applicable)
            "min": None, "max": None, "mean": None,
            "zero_pct": None, "neg_pct": None,
        }

        if dtype in ("numeric", "id") or pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().any():
                p["min"]      = round(float(num.min()), 2)
                p["max"]      = round(float(num.max()), 2)
                p["mean"]     = round(float(num.mean()), 2)
                p["zero_pct"] = round(float((num == 0).mean() * 100), 1)
                p["neg_pct"]  = round(float((num < 0).mean() * 100), 1)

        profiles.append(p)

    return profiles

def run_cleaning(df: pd.DataFrame, col_mapping: dict) -> tuple:
    fs  = CONFIG["null_fill_string"]
    fn  = float(CONFIG["null_fill_numeric"])
    mn  = float(CONFIG["min_amount"])
    mx  = float(CONFIG["max_amount"])
    thr = float(CONFIG["outlier_zscore_threshold"])
    dup_raw = CONFIG["duplicate_subset"].strip()

    df = df.copy()

    # ── 1. Track rows that had nulls ──────────────────────────────────────────
    df["had_nulls"] = df.isnull().any(axis=1).astype(int)
    nulls_before = int(df.isnull().sum().sum())

    # ── 2. Fill nulls by type ─────────────────────────────────────────────────
    for col in df.columns:
        if col == "had_nulls": continue
        if df[col].dtype == object:
            df[col] = df[col].fillna(fs)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(fn)
    nulls_filled = nulls_before - int(df.isnull().sum().sum())

    # ── 3. Deduplication ─────────────────────────────────────────────────────
    if dup_raw:
        subset = [c.strip() for c in dup_raw.split(";") if c.strip() in df.columns]
    else:
        subset = col_mapping.get("id_columns", []) or None
    rows_before   = len(df)
    df            = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    dupes_removed = rows_before - len(df)

    # ── 4. Date normalization — all detected date columns ─────────────────────
    date_errors = 0
    for col in col_mapping.get("date_columns", []):
        if col not in df.columns: continue
        orig   = df[col].copy()
        parsed = pd.to_datetime(df[col], errors="coerce")
        err    = parsed.isna() & orig.notna() & (~orig.astype(str).isin(["", fs, "nan", "None"]))
        df[f"{col}_parse_error"] = err.astype(int)
        date_errors += int(err.sum())
        df[col] = parsed.dt.strftime("%Y-%m-%d").where(~parsed.isna(), other=fs)

    # ── 5. Numeric validation — all detected numeric columns ──────────────────
    neg_flags = 0
    for col in col_mapping.get("numeric_columns", []):
        if col not in df.columns: continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fn)
        df[f"{col}_negative"]  = (df[col] < 0).astype(int)
        df[f"{col}_below_min"] = (df[col] < mn).astype(int)
        df[f"{col}_above_max"] = (df[col] > mx).astype(int)
        neg_flags += int((df[col] < 0).sum())

    # ── 6. Categorical standardisation ────────────────────────────────────────
    cat_fixed = 0
    for col in col_mapping.get("categorical_columns", []):
        if col not in df.columns: continue
        before = df[col].astype(str).copy()
        df[col] = (df[col].astype(str).str.strip()
                          .str.upper()
                          .replace({"NAN": fs, "NONE": fs, "": fs, "NULL": fs}))
        cat_fixed += int((df[col] != before).sum())

    # ── 7. ID / text columns — strip whitespace, fill placeholders ────────────
    for col in col_mapping.get("id_columns", []) + col_mapping.get("text_columns", []):
        if col not in df.columns: continue
        df[col] = df[col].astype(str).str.strip()
        ph = df[col].str.upper().isin(PLACEHOLDER_VALUES)
        df.loc[ph, col] = fs

    # ── 8. Outlier flagging — all numeric columns ─────────────────────────────
    outliers = 0
    for col in col_mapping.get("numeric_columns", []):
        if col not in df.columns: continue
        num = pd.to_numeric(df[col], errors="coerce")
        std = num.std()
        z   = (num - num.mean()) / std if std and std > 0 else pd.Series(0.0, index=df.index)
        df[f"{col}_outlier"] = (z.abs() > thr).astype(int)
        outliers += int((z.abs() > thr).sum())

    stats = {
        "rows_input":            rows_before,
        "rows_output":           len(df),
        "duplicates_removed":    dupes_removed,
        "nulls_filled":          nulls_filled,
        "rows_with_nulls":       int(df["had_nulls"].sum()),
        "date_parse_errors":     date_errors,
        "negative_amount_flags": neg_flags,
        "categorical_fixed":     cat_fixed,
        "outliers_flagged":      outliers,
        "date_cols_processed":   len(col_mapping.get("date_columns", [])),
        "numeric_cols_processed":len(col_mapping.get("numeric_columns", [])),
        "cat_cols_processed":    len(col_mapping.get("categorical_columns", [])),
    }
    return df, stats

def _clamp(s): return s.clip(0, 100)

def run_scoring(df: pd.DataFrame, col_mapping: dict) -> tuple:
    df  = df.copy()
    n   = len(df)
    fs  = CONFIG["null_fill_string"]

    critical = col_mapping.get("critical_fields", [f for f in df.columns[:6]])
    num_cols = col_mapping.get("numeric_columns", [])
    id_cols  = col_mapping.get("id_columns", [])
    cat_cols = col_mapping.get("categorical_columns", [])
    date_cols= col_mapping.get("date_columns", [])

    # ── COMPLETENESS (20%) ────────────────────────────────────────────────────
    sc = pd.Series(100.0, index=df.index)
    if critical:
        ppf = 100.0 / len(critical)
        for f in critical:
            if f not in df.columns: continue
            sc -= (df[f].isna() | df[f].astype(str).str.strip().str.upper()
                   .isin(PLACEHOLDER_VALUES)).astype(float) * ppf
    df["dq_score_completeness"] = _clamp(sc).round(1)

    # ── VALIDITY (25%) ────────────────────────────────────────────────────────
    sv = pd.Series(100.0, index=df.index)
    # Date parse errors — penalise per column
    date_pen = max(10, int(60 / max(len(date_cols), 1)))
    for col in date_cols:
        ecol = f"{col}_parse_error"
        if ecol in df.columns:
            sv -= df[ecol].fillna(0).astype(int) * date_pen
    # Categorical columns with very high placeholder rate get a light penalty
    for col in cat_cols[:5]:
        if col in df.columns:
            ph_rate = df[col].astype(str).str.upper().isin(PLACEHOLDER_VALUES).mean()
            if ph_rate > 0.5:
                sv -= pd.Series(20.0, index=df.index)
    df["dq_score_validity"] = _clamp(sv).round(1)

    # ── ACCURACY (35%) ────────────────────────────────────────────────────────
    sa = pd.Series(100.0, index=df.index)
    neg_pen = max(20, int(70 / max(len(num_cols), 1)))
    out_pen = max(10, int(30 / max(len(num_cols), 1)))
    abv_pen = max(10, int(30 / max(len(num_cols), 1)))
    for col in num_cols:
        if f"{col}_negative"  in df.columns: sa -= df[f"{col}_negative"].fillna(0)  * neg_pen
        if f"{col}_above_max" in df.columns: sa -= df[f"{col}_above_max"].fillna(0)  * abv_pen
        if f"{col}_outlier"   in df.columns: sa -= df[f"{col}_outlier"].fillna(0)    * out_pen
    df["dq_score_accuracy"] = _clamp(sa).round(1)

    # ── CONSISTENCY (15%) ─────────────────────────────────────────────────────
    sco = pd.Series(100.0, index=df.index)
    # Universal: check pairs of date columns for logical ordering
    # e.g. start < end, created < updated, invoice < due
    START_HINTS = {"start", "created", "invoice", "hire", "open",  "from", "begin", "order"}
    END_HINTS   = {"end",   "due",     "close",   "term", "closed","to",   "expire","delivery","ship"}
    date_pairs  = []
    for c1 in date_cols:
        for c2 in date_cols:
            if c1 == c2: continue
            c1l = c1.lower(); c2l = c2.lower()
            c1_start = any(h in c1l for h in START_HINTS)
            c2_end   = any(h in c2l for h in END_HINTS)
            if c1_start and c2_end:
                date_pairs.append((c1, c2))
    for c1, c2 in date_pairs[:3]:  # check up to 3 pairs
        d1 = pd.to_datetime(df[c1], errors="coerce")
        d2 = pd.to_datetime(df[c2], errors="coerce")
        bad = d1.notna() & d2.notna() & (d2 < d1)
        sco -= bad.astype(float) * 40
    df["dq_score_consistency"] = _clamp(sco).round(1)

    # ── UNIQUENESS (5%) ───────────────────────────────────────────────────────
    su = pd.Series(100.0, index=df.index)
    if id_cols:
        # Penalise if ANY id column has duplicates
        for col in id_cols[:2]:
            if col in df.columns:
                su -= df[col].duplicated(keep=False).astype(float) * (50 / min(len(id_cols), 2))
    df["dq_score_uniqueness"] = _clamp(su).round(1)

    # ── COMPOSITE ────────────────────────────────────────────────────────────
    overall = (
        df["dq_score_completeness"] * 0.20 +
        df["dq_score_validity"]     * 0.25 +
        df["dq_score_accuracy"]     * 0.35 +
        df["dq_score_consistency"]  * 0.15 +
        df["dq_score_uniqueness"]   * 0.05
    ).clip(0, 100).round(1)

    df["dq_score"] = overall.astype(int)
    df["dq_grade"] = overall.apply(
        lambda s: "A" if s>=90 else "B" if s>=75 else "C" if s>=60 else "D" if s>=40 else "F"
    )

    def sev(r):
        if r.dq_score_accuracy < 30 or r.dq_score_consistency < 40: return "CRITICAL"
        if r.dq_score_accuracy < 60 or r.dq_score_validity < 40 or r.dq_score < 50: return "HIGH"
        if r.dq_score < 70 or min(r.dq_score_accuracy, r.dq_score_consistency,
                                  r.dq_score_validity, r.dq_score_completeness) < 60: return "MEDIUM"
        if r.dq_score < 85: return "LOW"
        return "CLEAN"

    df["dq_severity"] = df[["dq_score","dq_score_accuracy","dq_score_consistency",
                             "dq_score_validity","dq_score_completeness"]].apply(sev, axis=1)

    # ── ISSUE LOG ─────────────────────────────────────────────────────────────
    issue_rules = (
        [(f"{c}_parse_error", f"[Validity] {c}: date unparseable")   for c in date_cols] +
        [(f"{c}_negative",    f"[Accuracy] {c}: negative value")      for c in num_cols] +
        [(f"{c}_above_max",   f"[Accuracy] {c}: exceeds max threshold") for c in num_cols] +
        [(f"{c}_outlier",     f"[Accuracy] {c}: statistical outlier") for c in num_cols] +
        [("had_nulls",        "[Completeness] Row had missing values")]
    )
    issues = [""] * n
    for flag_col, label in issue_rules:
        if flag_col not in df.columns: continue
        for i, v in enumerate(df[flag_col].fillna(0).astype(int).tolist()):
            if v:
                issues[i] = (issues[i] + " | " + label) if issues[i] else label
    # Consistency date pair issues
    for c1, c2 in date_pairs[:3]:
        d1 = pd.to_datetime(df[c1], errors="coerce")
        d2 = pd.to_datetime(df[c2], errors="coerce")
        bad = (d1.notna() & d2.notna() & (d2 < d1)).tolist()
        lbl = f"[Consistency] {c2} is before {c1}"
        for i, v in enumerate(bad):
            if v:
                issues[i] = (issues[i] + " | " + lbl) if issues[i] else lbl
    df["dq_issues"] = issues

    grade_dist = df["dq_grade"].value_counts().to_dict()
    sev_dist   = df["dq_severity"].value_counts().to_dict()
    stats = {
        "mean_dq_score":    round(float(overall.mean()), 1),
        "median_dq_score":  round(float(overall.median()), 1),
        "perfect_rows":     int((overall >= 99).sum()),
        "avg_completeness": round(float(df["dq_score_completeness"].mean()), 1),
        "avg_validity":     round(float(df["dq_score_validity"].mean()), 1),
        "avg_accuracy":     round(float(df["dq_score_accuracy"].mean()), 1),
        "avg_consistency":  round(float(df["dq_score_consistency"].mean()), 1),
        "avg_uniqueness":   round(float(df["dq_score_uniqueness"].mean()), 1),
        **{f"grade_{k}": v for k, v in grade_dist.items()},
        **{f"sev_{k}":   v for k, v in sev_dist.items()},
    }
    return df, stats

def run_anomaly_detection(df: pd.DataFrame, col_mapping: dict) -> pd.DataFrame:
    """
    Adds two columns:
      isolation_score    – IsolationForest score mapped to 0–100 (higher = more anomalous)
      iqr_outlier_count  – number of numeric cols where the row is an IQR outlier
    """
    num_cols = [c for c in col_mapping.get("numeric_columns", []) if c in df.columns]
    result = df.copy()

    # ── IQR outlier count (per-column, independent) ───────────────────────────
    iqr_flags = pd.DataFrame(False, index=df.index, columns=num_cols)
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_flags[col] = df[col].lt(lo) | df[col].gt(hi)
    result["iqr_outlier_count"] = iqr_flags.sum(axis=1).astype(int)

    # ── IsolationForest (multivariate across all numeric cols) ────────────────
    if len(num_cols) >= 1 and len(df) >= 10:
        X = df[num_cols].copy()
        X = X.fillna(X.median())          # impute for model
        try:
            iso = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
                n_jobs=-1,
            )
            raw_scores = iso.score_samples(X)   # more negative = more anomalous
            # Normalise to 0–100 (100 = most anomalous)
            lo_s, hi_s = raw_scores.min(), raw_scores.max()
            if hi_s > lo_s:
                norm = 100 * (1 - (raw_scores - lo_s) / (hi_s - lo_s))
            else:
                norm = np.zeros(len(raw_scores))
            result["isolation_score"] = np.round(norm, 1)
        except Exception:
            result["isolation_score"] = 0.0
    else:
        result["isolation_score"] = 0.0

    # ── Combined anomaly flag ─────────────────────────────────────────────────
    result["is_anomaly"] = (
        (result["isolation_score"] >= 70) |
        (result["iqr_outlier_count"] >= 2)
    )
    return result

def build_story_banner(df_f: pd.DataFrame, col_mapping: dict) -> str:
    total     = len(df_f)
    if total == 0:
        return ""
    avg_score = df_f["dq_score"].mean()
    critical  = int((df_f["dq_severity"] == "CRITICAL").sum())
    high      = int((df_f["dq_severity"] == "HIGH").sum())
    anomalies = int(df_f["is_anomaly"].sum()) if "is_anomaly" in df_f.columns else 0
    clean     = int((df_f["dq_severity"] == "CLEAN").sum())

    if avg_score >= 85:
        headline  = "✅ Your data looks healthy overall."
        bdr, bg   = "#1D9E75", "#f0fdf4"
    elif avg_score >= 65:
        headline  = "⚠️ Your data is mostly reliable — a few things need attention before major decisions."
        bdr, bg   = "#BA7517", "#fffbeb"
    else:
        headline  = "🔴 Your data has significant quality issues that could affect business decisions."
        bdr, bg   = "#E24B4A", "#fff5f5"

    bullets = []
    if critical > 0:
        bullets.append(f"🔴 <strong>{critical:,} records</strong> need immediate action — resolve these before relying on this data.")
    if high > 0:
        bullets.append(f"🟠 <strong>{high:,} records</strong> have significant issues — review today.")
    if anomalies > 0:
        bullets.append(f"🚨 <strong>{anomalies:,} records</strong> look statistically unusual compared to the rest — verify before decisions.")
    if clean > 0:
        pct = round(100 * clean / total)
        bullets.append(f"✅ <strong>{clean:,} records ({pct}%)</strong> are fully clean and ready to use.")

    dims = {
        "Missing Data":     df_f["dq_score_completeness"].mean(),
        "Format Accuracy":  df_f["dq_score_validity"].mean(),
        "Value Accuracy":   df_f["dq_score_accuracy"].mean(),
        "Data Consistency": df_f["dq_score_consistency"].mean(),
        "Duplicate Check":  df_f["dq_score_uniqueness"].mean(),
    }
    lowest_name, lowest_val = min(dims.items(), key=lambda x: x[1])
    if lowest_val < 80:
        bullets.append(f"📐 <strong>{lowest_name}</strong> is your weakest area ({lowest_val:.0f}/100) — start here.")

    bullets_html = "".join(f'<li style="margin:7px 0;line-height:1.5">{b}</li>' for b in bullets)
    return f"""
    <div style="background:#ffffff;border:1.5px solid #C79192;border-left:5px solid {bdr};
                border-radius:14px;padding:20px 24px;margin-bottom:24px;
                box-shadow:0 2px 12px rgba(0,0,0,0.04);">
      <p style="margin:0 0 14px;font-size:15px;font-weight:700;color:#23022E;">{headline}</p>
      <ul style="margin:0;padding-left:20px;color:#5E6472;font-size:13.5px;line-height:1.9;">{bullets_html}</ul>
    </div>"""
