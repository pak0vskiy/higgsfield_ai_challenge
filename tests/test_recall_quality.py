"""
Recall quality fixture test.
Ingests scripted conversations, runs probe queries, reports recall@k.
This is the iteration loop for CHANGELOG entries.

Baseline by version:
  v1 (rules-only, BM25): 6/11 probes passing (0.55)
  v4 (rules + implied-facts, BM25 + fake-vec RRF): target ≥ 0.65 offline
      Semantic probes (paraphrase, opinion) require OPENAI_API_KEY to pass.
"""
import os
import yaml
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "conversations"
pytestmark = pytest.mark.asyncio

HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))


def load_fixtures():
    fixtures = []
    for yaml_file in sorted(FIXTURES_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        fixtures.append((yaml_file.name, data))
    return fixtures


async def ingest_fixture(client, fixture_data: dict):
    user_id = fixture_data["user_id"]
    for turn in fixture_data["turns"]:
        payload = {
            "session_id": turn["session_id"],
            "user_id": user_id,
            "messages": turn["messages"],
            "timestamp": turn["timestamp"],
            "metadata": {}
        }
        r = await client.post("/turns", json=payload)
        assert r.status_code == 201, f"Failed to ingest turn: {r.text}"


async def run_probe(client, user_id: str, probe: dict) -> tuple[bool, str]:
    r = await client.post("/recall", json={
        "query": probe["query"],
        "session_id": "probe-session",
        "user_id": user_id,
        "max_tokens": 1024
    })
    assert r.status_code == 200
    context = r.json()["context"]

    # Check must_be_empty
    if probe.get("must_be_empty"):
        passed = context == ""
        reason = "empty" if passed else f"expected empty but got: {context[:100]}"
        return passed, reason

    # Check must_include
    missing = [s for s in probe.get("must_include", []) if s.lower() not in context.lower()]
    forbidden = [s for s in probe.get("must_not_include", []) if s.lower() in context.lower()]

    if missing:
        return False, f"missing: {missing}"
    if forbidden:
        return False, f"forbidden strings present: {forbidden}"
    return True, "ok"


async def test_recall_quality(client):
    fixtures = load_fixtures()
    assert fixtures, "No fixtures found in fixtures/conversations/"

    total_probes = 0
    passed_probes = 0
    failures = []

    for fname, fixture in fixtures:
        user_id = fixture["user_id"]
        await ingest_fixture(client, fixture)

        for probe in fixture.get("probes", []):
            total_probes += 1
            passed, reason = await run_probe(client, user_id, probe)
            if passed:
                passed_probes += 1
            else:
                failures.append(f"{fname} | probe='{probe['query']}' | {reason}")

    score = passed_probes / total_probes if total_probes else 0
    print(f"\n=== Recall Quality ===")
    print(f"recall_quality: {passed_probes}/{total_probes} probes passed ({score:.2f})")
    print(f"Embeddings provider: {os.environ.get('EMBEDDINGS_PROVIDER', 'openai')}")
    print(f"OpenAI key present: {HAS_OPENAI_KEY}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")

    # We report but don't assert a minimum score — score evolves across CHANGELOG versions.
    # Minimum bar: ≥ 65% of probes must pass offline (with fake embedder, rules extractor).
    # Probes that require semantic embeddings are expected to fail offline.
    assert total_probes > 0, "No probes were run"
    assert passed_probes / total_probes >= 0.65, (
        f"Recall quality too low: {passed_probes}/{total_probes} ({score:.2f}). "
        f"Expected >= 0.65 offline.\nFailures:\n" + "\n".join(failures)
    )


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set — skipping semantic recall test")
async def test_semantic_recall_quality(client):
    """
    Full semantic recall test using real OpenAI embeddings.
    Only runs when OPENAI_API_KEY is set (set EMBEDDINGS_PROVIDER=openai externally).
    Expected score: ≥ 0.80 with semantic embeddings.
    """
    # Force real embeddings for this test
    import os as _os
    _os.environ["EMBEDDINGS_PROVIDER"] = "openai"

    fixtures = load_fixtures()
    total_probes = 0
    passed_probes = 0
    failures = []

    for fname, fixture in fixtures:
        user_id = fixture["user_id"] + "_semantic"  # separate namespace
        # Re-ingest under a different user to avoid cross-test contamination
        for turn in fixture["turns"]:
            payload = {
                "session_id": turn["session_id"] + "_sem",
                "user_id": user_id,
                "messages": turn["messages"],
                "timestamp": turn["timestamp"],
                "metadata": {},
            }
            r = await client.post("/turns", json=payload)
            assert r.status_code == 201

        for probe in fixture.get("probes", []):
            total_probes += 1
            passed, reason = await run_probe(client, user_id, probe)
            if passed:
                passed_probes += 1
            else:
                failures.append(f"{fname} | probe='{probe['query']}' | {reason}")

    score = passed_probes / total_probes if total_probes else 0
    print(f"\n=== Semantic Recall Quality ===")
    print(f"recall_quality (semantic): {passed_probes}/{total_probes} probes passed ({score:.2f})")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")

    assert passed_probes / total_probes >= 0.80, (
        f"Semantic recall too low: {passed_probes}/{total_probes} ({score:.2f}). "
        f"Expected >= 0.80 with OpenAI embeddings.\nFailures:\n" + "\n".join(failures)
    )
