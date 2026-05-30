import re, logging
from datetime import datetime, timezone
from sqlite3 import Connection
from typing import Optional
import tiktoken
from memory_service.slots import is_singleton

logger = logging.getLogger(__name__)

# tiktoken encoder for token counting (o200k_base covers GPT-4o vocab)
try:
    _enc = tiktoken.get_encoding("o200k_base")
except Exception:
    _enc = None

LOW_SCORE_THRESHOLD = 0.15
PROFILE_SLOTS = {
    "identity.name", "identity.age", "identity.pronouns",
    "location.current", "location.previous", "location.hometown",
    "employment.current_company", "employment.current_role", "employment.previous_company",
    "relationship.partner", "preference.response_style", "preference.communication_style",
    "preference.diet",
}


def _count_tokens(text: str) -> int:
    if _enc is None:
        return len(text) // 4  # rough fallback: 4 chars ≈ 1 token
    return len(_enc.encode(text))


def _normalize_scores(raw_scores: list[float]) -> list[float]:
    """Normalize a list of scores to [0, 1] using min-max."""
    if not raw_scores:
        return []
    min_s = min(raw_scores)
    max_s = max(raw_scores)
    if max_s == min_s:
        return [1.0] * len(raw_scores)
    return [(s - min_s) / (max_s - min_s) for s in raw_scores]


def _recency_score(updated_at: str) -> float:
    """Score from 0-1 based on how recent the memory is. Decays over ~90 days."""
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days_old = max(0, (now - dt).days)
        return max(0.0, 1.0 - days_old / 90.0)
    except Exception:
        return 0.5


def _build_fts_query(query: str) -> str:
    """Build a simple FTS5 query: drop stopwords, OR-join remaining terms."""
    STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "do", "does", "did", "has", "have", "had", "i", "you", "we",
                 "they", "it", "this", "that", "what", "who", "where", "when",
                 "how", "which", "and", "or", "but", "in", "on", "at", "to",
                 "for", "of", "with", "by", "from", "user", "the", "their"}
    words = re.findall(r"\b\w{2,}\b", query.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    if not filtered:
        filtered = words  # if all stopwords, use all
    # FTS5 OR query — escape any special chars
    escaped = [re.sub(r'[^\w]', '', w) for w in filtered]
    escaped = [w for w in escaped if w]
    if not escaped:
        return "*"
    return " OR ".join(escaped)


# ── BM25 retrieval ─────────────────────────────────────────────────────────────

def _get_relevant_memories(conn: Connection, user_id: str, fts_query: str, limit: int = 20) -> list[dict]:
    """BM25 search over FTS5 index, filtered to user's active memories."""
    try:
        rows = conn.execute(
            """SELECT m.id, m.slot, m.entity_key, m.value_text, m.confidence,
                      m.source_turn_id, m.updated_at, m.evidence, m.active,
                      bm25(memories_fts) AS bm25_raw
               FROM memories_fts
               JOIN memories m ON memories_fts.rowid = m.rowid
               WHERE memories_fts MATCH ?
                 AND m.user_id = ?
                 AND m.active = 1
               ORDER BY bm25(memories_fts)
               LIMIT ?""",
            (fts_query, user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("FTS search failed (query=%r): %s", fts_query, e)
        return []


# ── Vector retrieval ───────────────────────────────────────────────────────────

def _vector_search(
    conn: Connection,
    user_id: str,
    query: str,
    limit: int = 50,
) -> list[dict]:
    """
    KNN search via sqlite-vec. Returns rows ordered by ascending distance
    (closer = more relevant). Over-fetches (limit=50) to allow per-user
    filtering in Python — the vec0 index is global.

    Returns empty list when sqlite-vec is unavailable or embedding fails.
    """
    from memory_service.db import VEC_ENABLED
    if not VEC_ENABLED:
        return []

    try:
        from memory_service.embeddings import embed, serialize_f32
        q_vec = embed(query)
        if q_vec is None:
            return []
        q_blob = serialize_f32(q_vec)
    except Exception as e:
        logger.warning("Query embedding failed: %s", e)
        return []

    try:
        # KNN over all vec rows; filter to user + active in Python
        knn_rows = conn.execute(
            """SELECT memory_rowid, distance
               FROM memories_vec
               WHERE embedding MATCH ?
               ORDER BY distance
               LIMIT ?""",
            (q_blob, limit * 3)   # over-fetch 3× to account for per-user filtering
        ).fetchall()
    except Exception as e:
        logger.warning("Vector KNN search failed: %s", e)
        return []

    if not knn_rows:
        return []

    # Fetch the full memory rows for the returned rowids
    rowid_map = {r["memory_rowid"]: r["distance"] for r in knn_rows}
    placeholders = ",".join("?" * len(rowid_map))
    mem_rows = conn.execute(
        f"""SELECT m.rowid as _rowid, m.id, m.slot, m.entity_key, m.value_text,
                   m.confidence, m.source_turn_id, m.updated_at, m.evidence, m.active
            FROM memories m
            WHERE m.rowid IN ({placeholders})
              AND m.user_id = ?
              AND m.active = 1""",
        (*rowid_map.keys(), user_id)
    ).fetchall()

    results = []
    for row in mem_rows:
        d = dict(row)
        d["vec_distance"] = rowid_map.get(d["_rowid"], 1e9)
        results.append(d)

    # Re-sort by distance ascending (best first) and truncate
    results.sort(key=lambda r: r["vec_distance"])
    return results[:limit]


# ── RRF fusion ─────────────────────────────────────────────────────────────────

def _rrf_fuse(
    bm25_ranked: list[dict],
    vec_ranked: list[dict],
    k: int = 60,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion.
    Returns {memory_id: rrf_score} — higher is better.
    When one channel is empty, scores are driven purely by the other channel.
    """
    scores: dict[str, float] = {}
    for rank, row in enumerate(bm25_ranked):
        mid = row["id"]
        scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
    for rank, row in enumerate(vec_ranked):
        mid = row["id"]
        scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ── Profile facts ──────────────────────────────────────────────────────────────

def _get_profile_facts(conn: Connection, user_id: str) -> list[dict]:
    """Return active Tier-1 profile facts for user, ordered by slot."""
    rows = conn.execute(
        """SELECT m.id, m.slot, m.entity_key, m.value_text, m.confidence,
                  m.source_turn_id, m.updated_at, m.evidence, m.active,
                  -- get previous value if exists
                  (SELECT value_text FROM memories prev
                   WHERE prev.user_id=m.user_id AND prev.slot=m.slot
                     AND prev.active=0 AND prev.supersedes IS NULL
                     -- pick the most recent inactive one that was superseded
                     -- simplification: just get any inactive for same slot
                   LIMIT 1) as prior_value
           FROM memories m
           WHERE m.user_id=? AND m.active=1 AND m.slot IN ({})
           ORDER BY m.slot
        """.format(",".join("?" * len(PROFILE_SLOTS))),
        (user_id, *PROFILE_SLOTS)
    ).fetchall()
    return [dict(r) for r in rows]


def _get_recent_turns(conn: Connection, session_id: str, limit: int = 5) -> list[dict]:
    """Return recent turns for the session."""
    import json as _json
    rows = conn.execute(
        """SELECT id, messages_json, timestamp FROM turns
           WHERE session_id=? ORDER BY created_at DESC LIMIT ?""",
        (session_id, limit)
    ).fetchall()
    results = []
    for row in rows:
        try:
            msgs = _json.loads(row["messages_json"])
            user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            results.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "snippet": user_msg[:200],
            })
        except Exception:
            pass
    return results


# ── Context formatting ─────────────────────────────────────────────────────────

def _format_profile_section(profile_facts: list[dict]) -> tuple[str, list[dict]]:
    """Format Tier-1 profile facts into the '## Known facts' section."""
    if not profile_facts:
        return "", []

    slot_values: dict[str, dict] = {}
    for f in profile_facts:
        slot_values[f["slot"]] = f

    lines = []
    citations = []
    processed = set()

    _SLOT_LABELS = {
        "identity.name": "Name",
        "identity.age": "Age",
        "identity.pronouns": "Pronouns",
        "location.current": "Lives in",
        "location.hometown": "Hometown",
        "employment.current_company": "Works at",
        "employment.current_role": "Role",
        "relationship.partner": "Partner",
        "preference.response_style": "Prefers",
        "preference.communication_style": "Communication style",
        "preference.diet": "Diet",
    }

    for slot, label in _SLOT_LABELS.items():
        if slot in slot_values and slot not in processed:
            f = slot_values[slot]
            val = f["value_text"]
            date_str = f["updated_at"][:10] if f.get("updated_at") else ""

            # Augment location.current with previous
            if slot == "location.current":
                prev = slot_values.get("location.previous")
                if prev:
                    val = f"{val} (previously {prev['value_text']})"
                    processed.add("location.previous")

            # Augment employment.current_company with role and previous
            if slot == "employment.current_company":
                role = slot_values.get("employment.current_role")
                prev_co = slot_values.get("employment.previous_company")
                if role:
                    val = f"{val} as {role['value_text']}"
                    processed.add("employment.current_role")
                if prev_co:
                    val = f"{val} (previously at {prev_co['value_text']})"
                    processed.add("employment.previous_company")

            line = f"- {label}: {val}."
            if date_str:
                line = f"- {label}: {val}. (updated {date_str})"
            lines.append(line)
            processed.add(slot)
            if f.get("source_turn_id"):
                citations.append({"turn_id": f["source_turn_id"], "score": 1.0, "snippet": f["evidence"] or val})

    # Any remaining profile facts not in the label map
    for slot, f in slot_values.items():
        if slot not in processed:
            lines.append(f"- {slot.replace('.', ' ').title()}: {f['value_text']}.")
            if f.get("source_turn_id"):
                citations.append({"turn_id": f["source_turn_id"], "score": 1.0, "snippet": f["evidence"] or f["value_text"]})

    return "## Known facts about this user\n" + "\n".join(lines), citations


# ── Multi-hop expansion ────────────────────────────────────────────────────────

def _extract_hop_terms(rows: list[dict]) -> list[str]:
    """
    Extract entity_keys and salient value tokens from the top retrieved rows
    to use as a second-hop query expansion.
    Returns a list of short strings (entity names, values) worth querying.
    """
    terms = set()
    for row in rows[:5]:   # limit to top-5 first-pass hits
        key = row.get("entity_key", "").strip()
        if key and len(key) >= 2:
            terms.add(key)
        val = row.get("value_text", "")
        # Add first token of the value if it looks like a name/place (capitalised, 2-20 chars)
        for word in val.split():
            cleaned = re.sub(r"[^\w]", "", word)
            if cleaned and cleaned[0].isupper() and 2 <= len(cleaned) <= 20:
                terms.add(cleaned.lower())
    return list(terms)


def _multi_hop_search(
    conn: Connection,
    user_id: str,
    hop_terms: list[str],
    already_seen: set[str],
    limit: int = 10,
) -> list[dict]:
    """
    Second-pass retrieval using entity/value tokens from the first pass.
    Returns rows not already in *already_seen*, scored lower (hop penalty applied).
    """
    if not hop_terms:
        return []

    combined_query = " OR ".join(
        re.sub(r'[^\w]', '', t) for t in hop_terms if re.sub(r'[^\w]', '', t)
    )
    if not combined_query:
        return []

    bm25_rows = _get_relevant_memories(conn, user_id, combined_query, limit=limit)
    vec_rows = _vector_search(conn, user_id, " ".join(hop_terms), limit=limit)

    rrf = _rrf_fuse(bm25_rows, vec_rows)

    # Merge rows from both channels, deduplicate
    seen: dict[str, dict] = {}
    for row in bm25_rows + vec_rows:
        if row["id"] not in seen:
            seen[row["id"]] = row

    results = []
    for mid, rrf_score in sorted(rrf.items(), key=lambda x: -x[1]):
        if mid in already_seen:
            continue
        row = seen.get(mid)
        if row is None:
            continue
        recency = _recency_score(row.get("updated_at", ""))
        confidence = float(row.get("confidence", 1.0))
        # Hop-2 penalty: RRF score capped at 0.5 so first-pass hits always rank higher
        rrf_norm = min(rrf_score * 100, 0.5)
        row["_final_score"] = 0.70 * rrf_norm + 0.15 * recency + 0.10 * confidence + 0.05 * 1.0
        row["_hop"] = 2
        results.append(row)

    return results


# ── Main recall ────────────────────────────────────────────────────────────────

def recall(
    conn: Connection,
    *,
    query: str,
    session_id: str,
    user_id: Optional[str],
    max_tokens: int = 1024,
) -> dict:
    """
    Main recall function. Returns {"context": str, "citations": list}.
    Never raises — returns empty on cold/unrelated queries.

    Retrieval pipeline:
      1. Profile facts (Tier-1, unconditional).
      2. First-pass: BM25 + vector KNN → RRF-fused ranking.
      3. Second-pass (multi-hop): use entity/value tokens from top first-pass
         hits as secondary query to surface linked facts.
      4. Greedy context assembly under token budget.
    """
    if not user_id:
        user_id = f"anon:{session_id}"

    try:
        # ── Step 1: Profile facts ──────────────────────────────────────────────
        profile_facts = _get_profile_facts(conn, user_id)
        profile_ids = {f["id"] for f in profile_facts}

        # ── Step 2: First-pass hybrid retrieval ───────────────────────────────
        fts_query = _build_fts_query(query)
        bm25_rows = _get_relevant_memories(conn, user_id, fts_query, limit=20)
        vec_rows = _vector_search(conn, user_id, query, limit=20)

        rrf_scores = _rrf_fuse(bm25_rows, vec_rows)

        # Build a deduplicated row map
        all_rows: dict[str, dict] = {}
        for row in bm25_rows + vec_rows:
            if row["id"] not in all_rows:
                all_rows[row["id"]] = row

        # Apply final scoring with RRF + recency + confidence
        rrf_values = list(rrf_scores.values())
        rrf_norm_map = {}
        if rrf_values:
            min_r, max_r = min(rrf_values), max(rrf_values)
            for mid, rrf_s in rrf_scores.items():
                if max_r == min_r:
                    rrf_norm_map[mid] = 1.0
                else:
                    rrf_norm_map[mid] = (rrf_s - min_r) / (max_r - min_r)

        first_pass: list[dict] = []
        for mid, row in all_rows.items():
            rrf_n = rrf_norm_map.get(mid, 0.0)
            recency = _recency_score(row.get("updated_at", ""))
            confidence = float(row.get("confidence", 1.0))
            active_boost = 1.0 if row.get("active") else 0.0
            score = 0.70 * rrf_n + 0.15 * recency + 0.10 * confidence + 0.05 * active_boost
            row["_final_score"] = score
            row["_hop"] = 1
            first_pass.append(row)

        first_pass.sort(key=lambda r: r["_final_score"], reverse=True)

        # Noise resistance: if no profile facts AND top score < threshold → empty
        top_score = first_pass[0]["_final_score"] if first_pass else 0.0
        if not profile_facts and top_score < LOW_SCORE_THRESHOLD:
            return {"context": "", "citations": []}

        # ── Step 3: Multi-hop expansion ───────────────────────────────────────
        hop_terms = _extract_hop_terms(first_pass)
        first_pass_ids = {r["id"] for r in first_pass} | profile_ids
        hop_rows = _multi_hop_search(conn, user_id, hop_terms, already_seen=first_pass_ids)

        # Combine: first-pass hits, then hop-2 hits (already penalised)
        ranked_rows = first_pass + hop_rows

        # ── Step 4: Recent session turns ──────────────────────────────────────
        recent_turns = _get_recent_turns(conn, session_id, limit=5)

        # ── Step 5: Assemble context under token budget ───────────────────────
        all_citations: list[dict] = []
        sections: list[str] = []
        tokens_used = 0

        # Priority 1: profile facts
        profile_section, profile_citations = _format_profile_section(profile_facts)
        if profile_section:
            profile_tokens = _count_tokens(profile_section)
            if tokens_used + profile_tokens <= max_tokens * 2:
                sections.append(profile_section)
                tokens_used += profile_tokens
                all_citations.extend(profile_citations)

        # Priority 2: query-relevant memories (greedy fill)
        RELEVANT_HEADER = "## Relevant from past conversations\n"
        relevant_header_tokens = _count_tokens(RELEVANT_HEADER)
        relevant_lines: list[str] = []
        relevant_citations: list[dict] = []

        for m in ranked_rows:
            if m["id"] in profile_ids:
                continue
            ts = m.get("updated_at", "")[:10]
            val = m["value_text"]
            evidence = m.get("evidence", "")
            snippet = evidence if evidence else val
            hop_label = " [via related memory]" if m.get("_hop") == 2 else ""
            line = f"- [{ts}] {val}{hop_label}"
            line_tokens = _count_tokens(line)
            header_overhead = relevant_header_tokens if not relevant_lines else 0
            if tokens_used + header_overhead + line_tokens > max_tokens * 2:
                break
            relevant_lines.append(line)
            tokens_used += header_overhead + line_tokens
            if m.get("source_turn_id"):
                relevant_citations.append({
                    "turn_id": m["source_turn_id"],
                    "score": round(m.get("_final_score", 0.0), 4),
                    "snippet": snippet[:200],
                })

        if relevant_lines:
            sections.append("## Relevant from past conversations\n" + "\n".join(relevant_lines))
            all_citations.extend(relevant_citations)

        # Priority 3: recent session turns (if budget allows)
        recent_lines: list[str] = []
        for turn in recent_turns[:2]:
            ts = turn["timestamp"][:10] if turn.get("timestamp") else ""
            snippet = turn["snippet"]
            line = f"- [{ts}] {snippet}"
            line_tokens = _count_tokens(line)
            if tokens_used + line_tokens > max_tokens * 2:
                break
            recent_lines.append(line)
            tokens_used += line_tokens

        if recent_lines:
            if relevant_lines:
                sections[-1] += "\n" + "\n".join(recent_lines)
            else:
                sections.append("## Relevant from past conversations\n" + "\n".join(recent_lines))

        context = "\n\n".join(sections)
        return {"context": context, "citations": all_citations}

    except Exception as e:
        logger.error("recall failed for user=%s session=%s: %s", user_id, session_id, e)
        return {"context": "", "citations": []}


# ── Search endpoint ────────────────────────────────────────────────────────────

def search(
    conn: Connection,
    *,
    query: str,
    session_id: Optional[str],
    user_id: Optional[str],
    limit: int = 10,
) -> list[dict]:
    """
    Search memories. Returns structured results (not prose). Used by /search endpoint.
    Uses hybrid BM25+vector retrieval with RRF fusion.
    """
    import json as _json
    if not user_id and not session_id:
        return []

    if not user_id:
        user_id_filter = None
    else:
        user_id_filter = user_id

    fts_query = _build_fts_query(query)

    # BM25 leg
    bm25_rows = _get_relevant_memories(conn, user_id_filter or "", fts_query, limit=limit * 2) if user_id_filter else []

    # Session filter fallback (if only session_id provided)
    if not bm25_rows and session_id:
        try:
            rows = conn.execute(
                """SELECT m.id, m.slot, m.entity_key, m.value_text, m.confidence,
                          m.source_session_id, m.source_turn_id, m.updated_at,
                          m.evidence, m.attributes_json, m.active,
                          bm25(memories_fts) AS bm25_raw
                   FROM memories_fts
                   JOIN memories m ON memories_fts.rowid = m.rowid
                   WHERE memories_fts MATCH ?
                     AND m.source_session_id = ?
                     AND m.active = 1
                   ORDER BY bm25(memories_fts)
                   LIMIT ?""",
                (fts_query, session_id, limit)
            ).fetchall()
            bm25_rows = [dict(r) for r in rows]
        except Exception as e:
            logger.warning("Session-scoped FTS search failed: %s", e)

    # Vector leg
    vec_rows = _vector_search(conn, user_id_filter or "", query, limit=limit * 2) if user_id_filter else []

    # Fuse and rerank
    rrf_scores = _rrf_fuse(bm25_rows, vec_rows)

    all_rows: dict[str, dict] = {}
    for row in bm25_rows + vec_rows:
        if row["id"] not in all_rows:
            all_rows[row["id"]] = row

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:limit]

    results = []
    for mid, score in ranked:
        row = all_rows.get(mid)
        if row is None:
            continue
        try:
            attrs = _json.loads(row.get("attributes_json") or "{}")
        except Exception:
            attrs = {}
        results.append({
            "content": row["value_text"],
            "score": round(score, 6),
            "session_id": row.get("source_session_id") or "",
            "timestamp": row.get("updated_at", ""),
            "metadata": attrs,
        })

    return results
