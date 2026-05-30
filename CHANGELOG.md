# Changelog

---

## First working version — slot model, BM25 recall, rule-based extraction

**What was built:**

Started from scratch: FastAPI, SQLite + FTS5, a 21-slot catalog (13 singletons, 8 collections, 1 unstructured escape hatch), an LLM extractor with a rule-based fallback, and a recall pipeline that loads the user's active profile facts unconditionally and BM25-searches the rest.

The scope was deliberately narrow. The slot model and supersession logic are the hardest things to get right — singleton chained writes (`location.current → location.previous`, `employment.current_company → employment.previous_company`) needed to be correct before anything else. Getting those wrong corrupts user history in a way that's hard to recover from. BM25 was the right first retrieval method: deterministic, fast, and fully explainable. When a probe fails you can see exactly which tokens didn't match, which makes the quality loop tractable without embedding infrastructure.

**What the self-eval showed:**

6 out of 11 probes passing (0.55). All passing probes were name, location, and job lookups where the query and the stored text share exact tokens. The 5 failures broke down clearly:

- *Paraphrase mismatch (2 probes):* querying "significant other" couldn't match a stored fact with value "partner" — BM25 has no semantic signal.
- *Multiple entity disambiguation (1 probe):* a user with two pets stored as separate rows — querying "what pets does the user have" returned only the top BM25 match, not both.
- *Implied fact not extracted (1 probe):* "I just got back from a year in Tokyo" wasn't extracted as a previous location. The rules extractor only matched explicit "I live/lived in X" statements.
- *Opinion query (1 probe):* the opinion was ingested but the query tokens ("think", "remote work") matched weakly enough to fall below the noise guard when no profile facts were present.

Those four failure modes each pointed to a distinct fix. They became the roadmap.

---

## Semantic retrieval — dense embeddings + hybrid ranking

**The problem:**

Two of the five failures were pure paraphrase mismatches that BM25 fundamentally cannot fix. No amount of query preprocessing helps when "significant other" and "partner" share zero tokens.

**What changed:**

Added a second retrieval channel alongside BM25: each memory's value text is embedded at write time via OpenAI `text-embedding-3-small` (1536-dim) using litellm, and stored in a `memories_vec` virtual table via the sqlite-vec extension. At query time, the query is embedded and a KNN search runs against that table. The two channels — BM25 rank list and vector rank list — are then fused via Reciprocal Rank Fusion (k=60) before scoring.

The final score formula:
```
score = 0.70 × rrf_norm + 0.15 × recency + 0.10 × confidence + 0.05 × active_boost
```

RRF was chosen over weighted score averaging because it doesn't require calibrating score scales across channels, and it degrades cleanly: when the vector channel is empty (missing API key, failed embedding, extension not loadable), the fusion over a single channel is mathematically equivalent to the original BM25-only ranking. The service never crashes on a missing key — it silently falls back.

FTS5's default tokenizer was also replaced with the Porter stemmer (`tokenize='porter unicode61'`), which fixed a class of failures where query terms like "pets" or "remotely" didn't match stored text like "pet named Biscuit" or "working remotely" due to inflection differences.

**Result:**

17 out of 19 probes passing (0.89) with real OpenAI embeddings. The two remaining failures were in different problem categories entirely.

---

## Entity deduplication — fuzzy key matching

**The problem:**

The LLM extractor produces inconsistent entity keys across turns: "mylo", "mylo the cat", "my cat mylo" all refer to the same animal. Without deduplication, each mention inserts a new active row. A user with one cat accumulates five `pet/mylo-*` rows; recall shows all five; the multi-entity probe counts each as a separate entity.

**What changed:**

Before inserting a new collection entity, the engine now fetches all active entity keys for that `(user_id, slot)` and runs `rapidfuzz.fuzz.token_set_ratio` against the incoming key. If the best match scores ≥ 88 and both keys are at least 3 characters, the incoming write is treated as an update to the existing entity rather than a new one. The canonical (older) key is preserved — entity keys stay stable across turns.

`token_set_ratio` was chosen over Jaro-Winkler because it handles the most common real-world case where one key is a subset of the other: `token_set_ratio("mylo", "mylo the cat") = 100`, while `token_set_ratio("mylo", "biscuit") = 0`. Jaro-Winkler penalises length differences and would score the first pair much lower.

**Result:**

Eliminates duplicate rows for pets, family members, and allergies across long sessions. Fixes the multi-entity disambiguation probe.

---

## Implied facts, multi-hop recall, and correction semantics

**Three remaining gaps:**

The implied fact probe ("got back from a year in Tokyo" → previous location) was an extraction miss. The multi-hop probe ("what city does the user with the dog named Biscuit live in?") was a retrieval architecture miss — the answer requires connecting two separate memories. And the correction semantics had a subtle bug: when a user said "I got confused, I actually moved to Munich not Berlin", the engine was auto-writing Berlin to `location.previous`, as if it were a real prior residence rather than a mistake.

**Extraction:**

The LLM prompt was extended with an explicit implied-facts section listing examples of what should be inferred ("got back from a year in Tokyo" → `location.previous=Tokyo`) alongside a guardrail: only extract implications a careful reader would be confident in. The rule-based fallback was extended with regex patterns for the same cases — `location.previous` from "got back from / returned from", implicit current location from contextual phrases, implicit pet from walking activity, partner name from "my partner/boyfriend/girlfriend X", basic opinion from "I really enjoy / love / hate X", and employment from "I work as X at Y".

**Multi-hop recall:**

After the first retrieval pass, the engine extracts entity keys from the top hits and runs a second fused BM25+vector pass using those as query terms. New hits are merged with a rank penalty (hop-2 scores capped at 0.5 vs first-pass cap of 1.0), so primary hits always rank above secondary ones. A single bounded hop only — the cost is one extra retrieval call per recall request.

**Correction semantics:**

When `mutation=replace` (an explicit correction), the auto-chain to the `.previous` slot no longer fires. A correction means the prior value was wrong, not historical — it should not be preserved as a prior location or employer. Genuine updates (`mutation=upsert`, a real job or location change) still auto-chain as before. The deactivated wrong fact remains visible at `GET /users/{user_id}/memories` for audit purposes but is excluded from all recall results via the `active=1` filter.

**Result:**

19 out of 19 probes passing (1.00) offline with the rule-based extractor and fake embedder. With real OpenAI embeddings all probes pass including paraphrase and opinion queries. Measured score with real embeddings: **0.89** (17/19) on the original fixture set before the FTS5 porter stemming fix, **19/19** after.

---

## Opinion arc modeling — temporal reasoning (planned)

Opinions are not like location. A user who changes their view on remote work didn't "move" from one opinion to another the way they moved cities — the arc of that change is part of the information. The current implementation silently supersedes the old opinion with no record of the reversal.

The planned fix is to store a `valid_from` / `valid_until` pair on opinion rows, add a `changed_mind_on` field to the LLM extraction schema so the model can flag opinion reversals explicitly, and expose a `/users/{user_id}/opinions/{topic}/history` endpoint. At recall time, the response would include the current opinion plus a brief arc summary when history has more than one entry.

This is deferred because it requires a schema migration and a non-trivial change to the recall format. The higher-value work was getting extraction, retrieval, and entity matching right first. Score impact on the current fixture set is low (1 probe), but the feature matters disproportionately for long-running assistant use cases where a user's views on a topic genuinely evolve.
