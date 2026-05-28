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

## v2 — dense retrieval with sqlite-vec + RRF fusion (PLANNED)

**What will change:**

Add sqlite-vec as a second retrieval channel. Embed memory `value_text` at write time using a small local model (planned: `nomic-embed-text` or `text-embedding-3-small` via OpenAI). Fuse BM25 and vector scores via Reciprocal Rank Fusion (RRF) before final scoring.

**Why:**

Two of the five v1 failures are pure paraphrase mismatches. BM25 cannot fix these — they require a semantic signal. RRF is a clean fusion strategy: it doesn't require calibrating score scales across retrieval methods and degrades gracefully when one channel returns nothing.

Embeddings also help the opinion query failure: "think about remote work" will embed closer to "opinion on remote work policies" than any BM25 term match could.

**Expected result:** 8–9/11 probes passing (~0.73–0.82). Paraphrase and opinion probe failures resolved. Multi-entity and implied-fact failures remain.

---

## v3 — fuzzy entity key matching (PLANNED)

**What will change:**

Replace exact `entity_key` lookup with fuzzy matching using `rapidfuzz`. When the engine writes `pet/mylo-the-cat` and a prior row exists as `pet/mylo`, they should be recognized as the same entity and trigger supersession rather than insertion. Apply the same fuzzy match at recall time to group entity variants under the same display name.

**Why:**

The multi-entity disambiguation failure in v1 is partly a recall assembly problem, but the root cause is upstream: the LLM produces inconsistent entity keys across turns (`"mylo"` vs `"mylo the golden"` vs `"my dog"`). Lowercasing normalizes case but doesn't handle morphological or descriptive variation. A fuzzy threshold (Jaro-Winkler ≥ 0.88) plus length constraints should handle most real-world cases without false merges.

**Expected result:** Fixes the multi-entity probe. Eliminates spurious duplicate pet/family/allergy records that accumulate across sessions when the LLM varies its entity key phrasing.

---

## v4 — implied fact extraction and multi-turn inference (PLANNED)

**What will change:**

Extend the LLM extraction prompt to explicitly ask for implied or inferable facts alongside stated ones. Add a second extraction pass for location-change signals ("I just got back from X", "I'm moving to X next month"). Explore a lightweight multi-hop recall step: after initial recall, re-query with retrieved entity names as secondary terms.

**Why:**

One of the v1 failures is a pure extraction miss: an implied `location.previous` was not written because the rules extractor only matches explicit statements and the LLM prompt didn't ask for inferences. Catching these requires prompt-level changes plus a clearly scoped policy on what "implied" means (to avoid hallucinated extractions).

Multi-hop recall ("Mylo is a cat" → find vet recommendations in stored memories) is a different problem: the recall pipeline today is a single-pass BM25+vector search. A second pass using retrieved entity names as query terms is a cheap approximation that covers the most common case.

**Expected result:** Resolves the implied-fact extraction probe. Multi-hop improvement is harder to quantify without new fixtures; estimate 1 additional probe passing.

---

## v5 — opinion arc modeling and temporal reasoning (PLANNED)

**What will change:**

Model opinion changes explicitly rather than treating them as entity-key supersession. Store a `valid_from` / `valid_until` pair on opinion rows and expose a `/users/{user_id}/opinions/{topic}/history` endpoint. Add a `changed_mind_on` field to the LLM extraction schema so the model can explicitly flag opinion reversals.

At recall time, return the current opinion plus a brief arc summary ("changed opinion on X in [month]") when the history has more than one entry.

**Why:**

Opinions are not like location — a user who changes their view on remote work didn't "move" from one opinion to another the way they moved cities. The temporal arc is part of the information. The current v1 model silently supersedes the old opinion with no record of the change, which is a semantic loss.

This is deferred to v5 because it requires schema migration (adding `valid_from`/`valid_until` to the `memories` table or a separate `opinion_arcs` table) and a non-trivial change to the recall format. Getting extraction, retrieval, and entity matching right first makes this change cheaper.

**Expected result:** Correct handling of opinion arc queries ("has their view on X changed?"). Score impact on the current fixture set is low (1 probe), but the feature is disproportionately important for long-running assistant use cases.
