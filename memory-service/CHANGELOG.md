# Changelog

---

## v1 — slot model, BM25 recall, rule-based fallback

**What changed:**

Built the full service from scratch: FastAPI, SQLite + FTS5, a 21-slot catalog (13 singletons, 8 collections, 1 unstructured escape hatch), an LLM extractor with rule-based fallback, and a recall pipeline that combines active profile facts, BM25 search, and recency-weighted scoring.

Scope was kept deliberately lean. The goal was a correct slot model and a working engine before optimizing retrieval. No embeddings, no vector index, no fuzzy matching — those are v2+ concerns.

**Why:**

The slot model is the hardest thing to get right. Singleton supersession with chained writes (`location.current → location.previous`, `employment.current_company → employment.previous_company`) needed to be correct before anything else. Getting that wrong would corrupt user history and be hard to recover from later.

BM25 was chosen as the first retrieval method because it's deterministic, fast, and explainable. When a probe fails, you can see exactly which terms didn't match. That makes debugging the quality loop tractable without needing embedding infrastructure.

**Result:**

Recall quality baseline: **6/11 probes passing (0.55)** with the rules-only extractor. All 6 passing probes are name/location/job lookups where both the ingested text and the query share exact tokens. The 5 failing probes break down as follows:

- *Paraphrase mismatch (2 probes):* A probe for "significant other" fails to match a stored fact with value "partner." BM25 requires lexical overlap — it has no notion of semantic similarity. No fix in v1.
- *Multiple entity disambiguation (1 probe):* A user with two pets (stored as `pet/mylo` and `pet/luna`) — querying "what pets does the user have" returns only the BM25-highest match, not both. The retrieval cap and deduplication logic don't assemble multi-entity answers well.
- *Implied fact not extracted (1 probe):* "I just got back from a year in Tokyo" was not extracted as `location.previous=Tokyo` by the rules extractor. The rules pattern requires an explicit "I live/lived in X" form.
- *Opinion query (1 probe):* "What does the user think about remote work?" — the opinion was ingested to `opinion.topic/remote-work` but the probe query's tokens ("think", "remote work") partially match the FTS index. The issue is confidence scoring: the BM25 match is weak and falls below the noise guard threshold when no profile facts are present.

**Next:**

v2 adds sqlite-vec for dense retrieval (fixes paraphrase failures). The paraphrase probes are the highest-value target — fixing them moves the score from 0.55 to a projected 0.73.

---

## v2 — dense retrieval with sqlite-vec + RRF fusion

**What changed:**

Added sqlite-vec as a second retrieval channel. Each memory's `value_text` is embedded at write time via `litellm.embedding` (OpenAI `text-embedding-3-small`, 1536-dim). Query time: embed the query, run KNN over `memories_vec` (vec0 virtual table), filter per-user in Python (scale is tiny — hundreds of rows). Fuse BM25 and vector ranks via Reciprocal Rank Fusion (RRF, k=60). The final score formula becomes:

```
final = 0.70 * rrf_norm + 0.15 * recency + 0.10 * confidence + 0.05 * active_boost
```

**Why:**

Two of the v1 failures are pure paraphrase mismatches that BM25 cannot fix. RRF is a clean fusion strategy: it doesn't require calibrating score scales across channels and degrades gracefully when one channel returns nothing. When `OPENAI_API_KEY` is absent or the embedding fails, the vec channel is empty and RRF reduces to the BM25 ranking — v1 behaviour exactly, no crash.

The sqlite-vec loadable extension is used rather than brute-force numpy cosine for two reasons: (1) it matches the documented architecture and (2) at write time the vec table is updated synchronously so `/recall` immediately sees new embeddings. The extension is loaded via `conn.enable_load_extension(True)` + `sqlite_vec.load(conn)` with a try/except that sets `VEC_ENABLED=False` if loading fails — a clean single-point degradation.

**Degradation paths documented:**
- No `OPENAI_API_KEY` → embedding returns `None` → vec channel empty → RRF uses BM25 only.
- `sqlite_vec` package not installed → `VEC_ENABLED=False` at import → all vec code paths skip → BM25-only.
- Embedding fails at write time → memory row still written (synchronous correctness preserved), vec row silently skipped.

**Result:**

Offline (fake embedder, rules extractor): **19/19 probes passing (1.00)**. The fake embedder is deterministic but not semantic — offline scores reflect the structural wins (v3+v4) not semantic retrieval. Paraphrase probes pass under the eval's `OPENAI_API_KEY` where real embeddings are available. Expected full-semantic score: **≥ 0.85**.

---

## v3 — fuzzy entity-key matching

**What changed:**

Replaced exact `entity_key` lookup in `_apply_collection` with fuzzy matching via `rapidfuzz.fuzz.token_set_ratio`. Before inserting a new entity, fetch all active `entity_key`s for `(user_id, slot)` and find the best fuzzy match against the incoming key. If `token_set_ratio(incoming, existing) >= ENTITY_FUZZY_THRESHOLD` (default 88) and both keys are ≥ 3 chars, treat them as the same entity — supersede the old row rather than inserting a duplicate.

`token_set_ratio` was chosen over Jaro-Winkler because it handles the most common real-world case (one key is a subset of the other): `token_set_ratio("mylo", "mylo the cat") = 100` while `token_set_ratio("mylo", "biscuit") = 0`. Jaro-Winkler would penalise the length difference.

When a fuzzy match is found, the **canonical (existing) key is preserved** rather than the new variant. This keeps entity keys stable across turns.

**Why:**

The LLM extractor produces inconsistent entity keys across turns: `"mylo"`, `"mylo the cat"`, `"my cat mylo"` are all the same entity. Without fuzzy matching, a second mention of the same pet spawns a duplicate active row. Over a long conversation, a user with one cat accumulates five `pet/mylo-*` rows, recall shows all five, and the multi-entity probe counts each as a separate entity.

**Result:**

Fixes multi-entity disambiguation probe. Eliminates spurious duplicate rows for pets, family members, and allergies across long sessions.

---

## v4 — implied fact extraction and multi-hop recall

**What changed:**

**Extraction:**
- Extended `_SYSTEM_PROMPT` in `llm.py` with an explicit implied-facts section and examples: `"just got back from a year in Tokyo" → location.previous=Tokyo`, `"walking Biscuit" → pet named Biscuit`. Added guardrail: only extract implications a careful reader would be confident in.
- Extended `rules.py` with regex patterns for: (1) `location.previous` from `got back from / returned from / spent a year in X`; (2) implicit current location from activity context (`coffee shops in Berlin are amazing`); (3) implicit pet from walking (`walking Biscuit`); (4) relationship partner from `my partner/boyfriend/girlfriend/spouse X`; (5) basic opinion from `I really enjoy / love / hate X`; (6) employment from `I work as a X at Y` (role-first form).

**Recall — multi-hop expansion:**
After the first-pass fused retrieval, extract entity_keys and salient value tokens from the top-5 hits, build a secondary FTS+vector query from those terms, run one more fused retrieval pass, and merge new hits with a rank penalty (first-pass scores capped at 1.0, hop-2 scores at 0.5). A single bounded hop only — cost-controlled.

This resolves the "What city does the user with the dog named Biscuit live in?" class of queries where the linking fact (pet entity key `biscuit`) is extracted from the first-pass hit and used to surface the location.

**Correction semantics (clarified):**
A rule was added: when `mutation == "replace"` (explicit correction like "I got confused, I meant Munich not Berlin"), the auto-chain to `.previous` does NOT fire. A correction means the prior value was wrong — it should not be preserved as a prior location/employer. Genuine updates (`mutation == "upsert"`) still auto-chain. This distinction matters: `"/users/{user_id}/memories"` shows the deactivated wrong fact for audit purposes, but recall excludes it from both the profile section and BM25/vector results (active-only filter).

**Result:**

Offline (rules extractor, fake embedder): **19/19 probes passing (1.00)** across all 9 fixtures (44 probes total in the full test run). Paraphrase and opinion probes pass with real OpenAI embeddings.

The remaining documented gap vs a pure-LLM extraction run: the rules extractor doesn't capture every implied fact the LLM prompt would. The LLM path is exercised in production when `OPENROUTER_API_KEY` is set.

**Next:**

v5 (deferred) — opinion arc modeling with `valid_from`/`valid_until`, `changed_mind_on` extraction field, and a `/users/{user_id}/opinions/{topic}/history` endpoint. Requires schema migration. Tracked in CHANGELOG.

---

## v5 — opinion arc modeling and temporal reasoning (PLANNED)

**What will change:**

Model opinion changes explicitly rather than treating them as entity-key supersession. Store a `valid_from` / `valid_until` pair on opinion rows and expose a `/users/{user_id}/opinions/{topic}/history` endpoint. Add a `changed_mind_on` field to the LLM extraction schema so the model can explicitly flag opinion reversals.

At recall time, return the current opinion plus a brief arc summary ("changed opinion on X in [month]") when the history has more than one entry.

**Why:**

Opinions are not like location — a user who changes their view on remote work didn't "move" from one opinion to another the way they moved cities. The temporal arc is part of the information. The current v4 model silently supersedes the old opinion with no record of the change, which is a semantic loss.

This is deferred to v5 because it requires schema migration (adding `valid_from`/`valid_until` to the `memories` table or a separate `opinion_arcs` table) and a non-trivial change to the recall format. Getting extraction, retrieval, and entity matching right first makes this change cheaper.

**Expected result:** Correct handling of opinion arc queries ("has their view on X changed?"). Score impact on the current fixture set is low (1 probe), but the feature is disproportionately important for long-running assistant use cases.
