# Memory Service

A stateful, user-scoped memory layer for LLM applications. Ingests conversation turns, extracts structured facts, and returns grounded context on demand.

---

## Architecture

```
  Caller (LLM host / API client)
       |
       |  POST /turns          POST /recall         POST /search
       v                            |                    |
  ┌──────────────────────────────────────────────────────────────┐
  │                        FastAPI app                           │
  │                                                              │
  │  TurnHandler                RecallHandler     SearchHandler  │
  │      │                           │                 │        │
  │      ▼                           ▼                 ▼        │
  │  Extractor                   recall()          search()     │
  │  (LLM → rules fallback)      (recall.py)      (recall.py)  │
  │      │                           │                 │        │
  │      ▼                           └────────┬────────┘        │
  │  MemoryEngine                             │                 │
  │  (apply_memories)                         │                 │
  │      │                                    ▼                 │
  │      └──────────────────► SQLite + FTS5 (/data/memory.db)  │
  └──────────────────────────────────────────────────────────────┘
```

The service is a single-container FastAPI application. There is no secondary service, cache layer, or message queue. All state lives in a SQLite file at `/data/memory.db` inside a named Docker volume (`memdata`).

On each `POST /turns`, the service stores the raw turn, runs extraction (LLM → rule-based fallback), and calls the memory engine to apply any extracted facts with slot-level semantics. On `POST /recall`, it runs a two-phase retrieval: load the user's active Tier-1 profile facts unconditionally, then BM25-search the FTS5 index for query-relevant memories, then assemble a token-budgeted prose block for injection into the caller's context window.

---

## Backing Store Choice

**SQLite with FTS5** was chosen deliberately over Postgres or a dedicated vector store.

SQLite eliminates the ops burden of a separate DB process. For a stateful memory sidecar, reliability under failure (power loss, OOM kill) matters more than throughput — SQLite's WAL mode and fsync guarantees are battle-tested. The single-file layout makes backups trivial: `cp memory.db memory.db.bak`.

FTS5 provides BM25 full-text search over memory values and evidence text without an additional index service. The BM25 scores are not normalized by SQLite — the service normalizes them to [0, 1] via min-max before combining with recency, confidence, and active-slot signals. FTS5 has one known gap: it has no concept of semantic similarity, so paraphrase recall ("partner" vs "wife") fails unless both terms appear in the indexed text.

---

## Extraction Pipeline

**Flow:** raw messages → LLM extractor (or rule fallback) → pydantic-validated candidates → memory engine

The LLM prompt gives the model the canonical slot catalog and the user's current known facts (for context-aware supersession). The model returns structured JSON: `type`, `slot`, `entity_key`, `value_text`, `confidence` (0–1), `evidence` (the quote that justifies the extraction), and `mutation_intent` (assert / retract / supersede).

The memory engine validates each candidate against the slot catalog, normalizes `entity_key` to lowercase, and applies tier logic:

- **Tier 1 (13 singleton slots):** one active row per user. Writing a new value auto-supersedes the old row and sets `active=0`. Two slots have chained supersession: `location.current → location.previous` and `employment.current_company → employment.previous_company`.
- **Tier 2 (8 collection slots):** multiple active rows per user, disambiguated by `entity_key`. `pet` with `entity_key=mylo` is independent of `pet` with `entity_key=luna`.
- **Tier 3 (`unstructured`):** always inserted, always active, always FTS-indexed. Escape hatch for facts that don't fit any slot.

**Known misses:**
- The rule-based fallback (no API keys) covers only a handful of regex patterns (name, location, job). Most structured extraction requires the LLM.
- Opinion tracking writes to a collection slot, but opinion arcs (changed minds) are treated as entity-key collision, not modeled as a temporal arc.
- Entity disambiguation for collections is exact-match on lowercased `entity_key`. "Mylo" and "mylo the cat" are treated as different pets.

---

## Recall Strategy

**Candidate generation (three channels):**

A. Active Tier-1 profile facts — loaded unconditionally for every recall call. These are the high-confidence singleton values (name, location, employment, etc.) and always appear first in the assembled context.

B. FTS5 BM25 search — the query is preprocessed: stopwords dropped, remaining tokens OR-joined into an FTS5 MATCH expression. Up to 20 rows are retrieved per call.

C. Recent session turns — up to 2 recent turns for the current session, appended if token budget allows.

**Scoring (channel B rows only):**

```
score = 0.70 * bm25_norm + 0.15 * recency + 0.10 * confidence + 0.05 * active_boost
```

`bm25_norm` is min-max normalized over the candidate set. `recency` decays linearly over 90 days. `active_boost` is 1.0 for active rows, 0.0 for superseded ones.

**Budget and format:**

Context is assembled greedily under a soft `2 × max_tokens` cap (default max_tokens=1024, measured with tiktoken `o200k_base`). Profile facts go first and are never cut. Query-relevant rows fill the remainder in score order. A noise guard returns empty context if there are no profile facts and the top BM25 score is below 0.15.

Output format is two markdown sections: `## Known facts about this user` and `## Relevant from past conversations`. Citations are returned as a parallel list with `turn_id`, `score`, and `snippet`.

---

## Fact Evolution

Tier-1 singletons supersede automatically. When the LLM extracts `location.current = "Berlin"` and the user previously had `location.current = "New York"`, the engine sets the old row to `active=0`, copies its value into `location.previous`, and inserts the new row. The supersession chain is stored in the `supersedes` column (foreign key to prior memory ID).

For collection slots, evolution is entity-keyed: writing `pet / entity_key=mylo / value=golden retriever` after `pet / entity_key=mylo / value=puppy` supersedes the old record for that specific pet. Writing a completely new entity key just inserts.

For `unstructured`, there is no supersession — every write is a fresh insert. The FTS index covers all unstructured rows.

Corrections (user says "actually I'm 32 not 31") rely on the LLM detecting the contradiction and emitting `mutation_intent=supersede`. The rule-based fallback does not model corrections.

---

## Tradeoffs

**Optimized for:** correctness on cold-start and new-user paths; no-crash reliability without API keys; simple operational story (one file, one container); legible slot model for iterating on quality.

**Sacrificed:**
- **Paraphrase recall.** BM25 is lexical. A query about "significant other" will not match a memory stored as "partner" unless the text overlaps. Embedding-based retrieval was deferred to v2 intentionally — getting the slot model and engine correct first, then improving retrieval quality.
- **Scale.** SQLite is single-writer. Concurrent POST /turns will contend. This is acceptable for a sidecar pattern (one caller per user session) but not for a multi-tenant service at volume.
- **Multi-hop reasoning.** The recall pipeline returns facts; it does not reason over them. Connecting "Mylo is a cat" → "user is in Berlin" → "recommend Berlin vets" requires the caller model to do that reasoning.
- **Extraction recall on ambiguous text.** The LLM confidently ignores implied facts ("I just got back from Tokyo" → no `location.previous` written unless explicitly stated). Rule coverage is sparse.

---

## Failure Modes

**No user data (cold start):** Profile facts return empty. BM25 returns nothing. Noise guard fires. `/recall` returns `{"context": "", "citations": []}`. This is correct behavior — no hallucinated context.

**Slow or full disk:** SQLite writes will block and eventually return `SQLITE_IOERR` or `SQLITE_FULL`. The service catches and logs DB errors but does not have a circuit breaker. Under disk pressure, `/turns` will return 500 for the commit; `/recall` reads-only and will succeed as long as the file is readable.

**Missing API keys:** `get_extractor()` falls back to the rule-based extractor automatically. The rules cover name, location, and job patterns only. Extraction quality drops sharply but the service never returns 5xx on extraction failure — errors are logged and the turn is stored with zero extracted memories.

**FTS query malformed:** The `_build_fts_query` function defensively escapes special characters and falls back to `"*"` on empty input. FTS5 failures are caught per-query and return an empty candidate set rather than propagating.

---

## How to Run the Tests

```bash
# Install dependencies (requires Python 3.12+)
pip install -e ".[dev]"

# Run all tests
pytest

# Run just the recall quality fixture test (prints the recall score)
pytest tests/test_recall_quality.py -s

# Run a specific test module
pytest tests/test_memory_engine.py -v
pytest tests/test_recall.py -v
```

Fixture conversations live in `fixtures/conversations/*.yaml`. The quality test ingests each fixture, runs named probes, and prints `recall_quality: N/M probes passed (score)`. A minimum of 0.50 is enforced as a hard gate. The current v1 baseline with the rules-only extractor is **6/11 (0.55)**.

To run against the LLM extractor, set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` before running pytest. Without keys, the rules extractor runs automatically.
