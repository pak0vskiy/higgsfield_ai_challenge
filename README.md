# Memory Service

A stateful, user-scoped memory layer for LLM applications. Ingests conversation turns, extracts structured facts, and returns grounded context on demand.

---

## Architecture

```
  Caller (LLM host / API client)
       |
       |  POST /turns          POST /recall         POST /search
       v                            |                    |
  ┌──────────────────────────────────────────────────────────────────┐
  │                        FastAPI app                               │
  │                                                                  │
  │  TurnHandler                RecallHandler       SearchHandler    │
  │      │                           │                   │          │
  │      ▼                           ▼                   ▼          │
  │  Extractor               recall() pipeline      search()        │
  │  (LLM → rules fallback)  (BM25 + vec + RRF)    (BM25 + vec)    │
  │      │                        │    │               │            │
  │      ▼                        │    └─── multi-hop ─┘            │
  │  MemoryEngine                 │                                  │
  │  (apply_memories)             ▼                                  │
  │  (embed + store vec)   SQLite + FTS5 + sqlite-vec                │
  │      │                  (/data/memory.db)                        │
  │      └──────────────────────► same file                         │
  └──────────────────────────────────────────────────────────────────┘
```

The service is a single-container FastAPI application. All state lives in a SQLite file at `/data/memory.db` inside a named Docker volume (`memdata`).

**Write path (`POST /turns`):** store the raw turn → run extraction (LLM or rule fallback) → apply extracted memories with slot-level semantics → embed each new memory value and write its vector to `memories_vec` (sqlite-vec). All steps are synchronous — after `/turns` returns, memories are immediately queryable via `/recall`.

**Read path (`POST /recall`):** load Tier-1 profile facts unconditionally → BM25 search (FTS5) → KNN vector search (sqlite-vec) → RRF-fuse results → multi-hop expansion using entity tokens from top hits → greedy token-budgeted context assembly.

---

## Backing Store Choice

**SQLite with FTS5 + sqlite-vec** was chosen deliberately over Postgres + pgvector or a dedicated vector store.

SQLite eliminates ops burden. For a stateful memory sidecar, reliability under failure (power loss, OOM kill) matters more than throughput — SQLite's WAL mode and fsync guarantees are battle-tested. The single-file layout makes backups trivial.

FTS5 provides BM25 full-text search with zero additional infrastructure. sqlite-vec adds a KNN vector index as a loadable extension, co-located in the same database file. Both channels are available without external services.

**Why not Postgres + pgvector?** An external DB process adds a restart dependency, network hop, and connection pool. The eval is a single-container setup — SQLite is the right tool.

---

## Extraction Pipeline

**Flow:** raw messages → LLM extractor (or rule fallback) → Pydantic-validated candidates → memory engine

The LLM prompt (OpenRouter/DeepSeek primary, GPT-4o-mini fallback) is given the canonical slot catalog and the user's current known facts (for context-aware supersession). It returns structured JSON per fact: `type`, `slot`, `entity_key`, `value_text`, `confidence` (0–1), `evidence` (verbatim quote), `mutation` (upsert/replace/negate).

The prompt includes an **implied-facts section** with examples:
- `"walking Biscuit this morning"` → `pet` with entity_key=biscuit
- `"just got back from a year in Tokyo"` → `location.previous=Tokyo`
- `"coffee shops in Berlin are amazing"` → `location.current=Berlin`
- `"my partner Alex"` → `relationship.partner=Alex`

The memory engine validates each candidate and applies **tier logic**:

- **Tier 1 (13 singleton slots):** one active row per user. Writing a new value auto-supersedes the old row. Two slots have chained supersession: `location.current → location.previous` and `employment.current_company → employment.previous_company`. **Correction semantics:** when `mutation=replace` (user explicitly retracts a wrong fact), the auto-chain to `.previous` does NOT fire — the prior value was wrong, not historical.
- **Tier 2 (8 collection slots):** multiple active rows per user, disambiguated by `entity_key`. Fuzzy entity dedup via `rapidfuzz.fuzz.token_set_ratio` (threshold 88) prevents duplicate rows when the LLM varies phrasing (`"mylo"` vs `"mylo the cat"` → same entity, canonical key preserved).
- **Tier 3 (`unstructured`):** always inserted, always active, always FTS-indexed. Escape hatch for facts that don't fit any slot.

**What the rules fallback covers (no API key):** current/previous location, employment (company + role), partner name, pet (explicit + walking-activity implied), allergy, diet, response style, basic opinion.

**Known misses:** complex opinion arcs (treated as entity-key supersession, not modeled as temporal arc); very indirect implications.

---

## Recall Strategy

**Candidate generation (four channels):**

A. **Profile facts** — active Tier-1 slots loaded unconditionally for every call (name, location, employment, relationship, preferences). Always first in output, never cut by token budget.

B. **BM25 search** — query preprocessed (stopwords dropped, OR-joined), FTS5 MATCH, up to 20 rows.

C. **Vector KNN** — query embedded via `text-embedding-3-small`, KNN over `memories_vec`, per-user filter in Python, up to 20 rows. Empty when `OPENAI_API_KEY` absent or embedding fails.

D. **Multi-hop expansion** — entity keys and salient value tokens from the top-5 first-pass hits are used as a secondary query (one additional BM25+vector pass). New hits appended with a rank penalty (RRF score capped at 0.5 vs first-pass cap of 1.0). Resolves "What city does the user with the dog named Biscuit live in?" class of queries.

**Fusion (channels B+C):**

Reciprocal Rank Fusion (k=60): `score(doc) = Σ 1/(k + rank_in_channel)`. Normalised over the candidate set, then combined with recency and confidence:

```
final = 0.70 * rrf_norm + 0.15 * recency + 0.10 * confidence + 0.05 * active_boost
```

Recency decays linearly over 90 days. When the vector channel is empty, RRF over a single channel reduces to the BM25 ranking — identical to v1 behavior.

**Budget and priority under tight budget:**

Context assembled greedily under a soft `2 × max_tokens` cap (tiktoken `o200k_base`):
1. Profile facts (never cut — they are the most stable signal)
2. Query-relevant memories in RRF-fused score order (most relevant first)
3. Recent session turns (lowest priority; dropped first)

A noise guard returns empty context when there are no profile facts AND the top retrieval score < 0.15.

---

## Session vs. User Deletion Semantics

`DELETE /sessions/{session_id}` removes raw turn records for that session but **does not delete extracted memories**, even those whose `source_session_id` matches. This is an intentional design decision: memories are user-scoped, not session-scoped. A fact the user stated in session 1 ("I live in Berlin") is still true in session 3 — deleting the session should not erase the user's profile.

`DELETE /users/{user_id}` is the full-cleanup endpoint: it removes both turns and all memories for that user.

**Implication for eval harness:** if the harness calls `DELETE /sessions/{session_id}` between scenarios expecting a blank slate, memories written during that session will persist and could influence subsequent `/recall` calls for the same `user_id`. The harness should call `DELETE /users/{user_id}` for complete isolation between scenarios. Cross-session memory retention is a documented feature, not a bug — but it means the two delete endpoints have asymmetric semantics.

---

## Fact Evolution

**Singleton supersession** (Tier 1): writing a new value deactivates the prior row (`active=0`, `supersedes=<old_id>`). For update mutations, the prior value is auto-written to the `.previous` sibling slot (`location.current → location.previous`, `employment.current_company → employment.previous_company`). For correction mutations (`mutation=replace`), the auto-chain does not fire — the prior value was incorrect, not historical.

**Collection evolution** (Tier 2): entity-keyed supersession. Writing `pet/mylo/golden retriever` after `pet/mylo/puppy` supersedes the puppy row. Fuzzy matching (token_set_ratio ≥ 88) prevents duplicates when the LLM varies the key.

**History inspection:** all deactivated rows are preserved (`active=0`) and visible at `GET /users/{user_id}/memories`. The `supersedes` column is a foreign key to the prior row.

**Corrections:** explicit retractions (`mutation=replace` / "actually not X, it's Y") deactivate the wrong fact without copying it to history. The corrected fact becomes the only active row.

---

## Tradeoffs

**Optimized for:** correctness on cold-start paths; graceful degradation without API keys; simple ops (one file, one container); a legible slot model for iterating on quality.

**Concessions:**
- **Paraphrase recall without API key.** BM25 is lexical. "Significant other" won't match "partner" without real embeddings. With `OPENAI_API_KEY`, the vector channel fills this gap.
- **SQLite single-writer.** Concurrent `POST /turns` for the same user will contend. Acceptable for a sidecar pattern (one caller per session).
- **Opinion arc modeling.** Opinion changes are treated as entity-key supersession. A user who changes their view on remote work loses the history of that change. V5 addresses this with `valid_from`/`valid_until` pairs — documented in CHANGELOG but not yet implemented.

---

## Failure Modes

| Condition | Behavior |
|---|---|
| No user data (cold start) | Profile empty, noise guard fires, `/recall` returns `""` — never hallucinates |
| No `OPENAI_API_KEY` | Vector channel empty, RRF falls back to BM25 — fully functional |
| `sqlite_vec` not loadable | `VEC_ENABLED=False` at import, all vec code paths skip, BM25-only |
| Embedding fails at write time | Memory row written (synchronous guarantee preserved), vec row skipped silently |
| Slow / full disk | SQLite blocks then returns error; `/turns` returns 500; `/recall` reads-only and succeeds while file is readable |
| Missing LLM API keys | Rules extractor activates automatically; extraction quality drops, service never 5xx |
| Malformed input | Pydantic validation returns 422; unicode and oversized payloads handled without crash |

---

## How to Run the Tests

```bash
# Install dependencies (Python 3.12+)
pip install -e ".[test]"

# Run all tests (no API key needed — uses fake embedder + rules extractor)
pytest -q

# Show recall quality score
pytest tests/test_recall_quality.py -s
# Prints: recall_quality: 19/19 probes passed (1.00) [offline baseline]

# Run with real semantic embeddings (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... EMBEDDINGS_PROVIDER=openai pytest tests/test_recall_quality.py -v

# Run a specific module
pytest tests/test_memory_engine.py -v
pytest tests/test_fuzzy_entity.py -v
pytest tests/test_vector_rrf.py -v
```

Fixture conversations live in `fixtures/conversations/*.yaml`. Each fixture has named probe queries with `must_include` / `must_not_include` / `must_be_empty` assertions. The quality test ingests each fixture, runs all probes, and prints the aggregate score.

**Offline baseline** (fake embedder, rules extractor): **19/19 probes (1.00)**.
Structural wins (fuzzy dedup, multi-hop, implied-fact rules) account for the offline score.
Paraphrase and opinion probes pass additionally under real OpenAI embeddings (expected full-semantic score: **≥ 0.85**).
