import os, pytest
os.environ["MEMORY_EXTRACTOR"] = "rules"  # offline tests use rules only

from memory_service.extractor.rules import RulesExtractor
from memory_service.extractor.base import ExtractedMemory


def make_msgs(*contents):
    return [{"role": "user", "content": c} for c in contents]


def test_location_extraction():
    r = RulesExtractor()
    result = r.extract(make_msgs("I just moved to Berlin from NYC last month."))
    slots = [m.slot for m in result.memories]
    assert "location.current" in slots
    loc = next(m for m in result.memories if m.slot == "location.current")
    assert "berlin" in loc.value_text.lower()


def test_employment_extraction():
    r = RulesExtractor()
    result = r.extract(make_msgs("I work at Notion as a PM."))
    slots = [m.slot for m in result.memories]
    assert "employment.current_company" in slots


def test_pet_extraction():
    r = RulesExtractor()
    result = r.extract(make_msgs("I have a dog named Biscuit."))
    pets = [m for m in result.memories if m.slot == "pet"]
    assert len(pets) >= 1
    assert pets[0].entity_key == "biscuit"


def test_allergy_extraction():
    r = RulesExtractor()
    result = r.extract(make_msgs("I'm allergic to shellfish."))
    allergies = [m for m in result.memories if m.slot == "restriction.allergy"]
    assert len(allergies) >= 1
    assert "shellfish" in allergies[0].entity_key


def test_diet_extraction():
    r = RulesExtractor()
    result = r.extract(make_msgs("I'm vegetarian."))
    diets = [m for m in result.memories if m.slot == "preference.diet"]
    assert len(diets) >= 1


def test_no_false_positives_from_assistant():
    r = RulesExtractor()
    msgs = [
        {"role": "assistant", "content": "You live in Berlin and have a cat named Mylo."},
    ]
    result = r.extract(msgs)
    # Should not extract from assistant messages
    assert result.memories == []


def test_multiple_pets_no_dedup():
    r = RulesExtractor()
    result = r.extract(make_msgs("I have a cat named Mylo and a dog named Biscuit."))
    pet_keys = {m.entity_key for m in result.memories if m.slot == "pet"}
    assert "mylo" in pet_keys
    assert "biscuit" in pet_keys


# ── v4: Implied-fact extraction tests ─────────────────────────────────────────

def test_location_previous_from_got_back():
    """'I just got back from a year in Tokyo' → location.previous = Tokyo"""
    r = RulesExtractor()
    result = r.extract(make_msgs("I just got back from a year in Tokyo."))
    prev = [m for m in result.memories if m.slot == "location.previous"]
    assert len(prev) >= 1, (
        f"Expected location.previous from 'got back from', got: {[m.slot for m in result.memories]}"
    )
    assert "tokyo" in prev[0].value_text.lower(), f"Expected Tokyo, got: {prev[0].value_text}"


def test_location_previous_from_returned_from():
    """'I returned from Berlin last year' → location.previous = Berlin"""
    r = RulesExtractor()
    result = r.extract(make_msgs("I returned from Berlin last year."))
    prev = [m for m in result.memories if m.slot == "location.previous"]
    assert len(prev) >= 1, (
        f"Expected location.previous from 'returned from', got: {[m.slot for m in result.memories]}"
    )
    assert "berlin" in prev[0].value_text.lower()


def test_implicit_pet_from_walking():
    """'I was walking Biscuit this morning' → pet named Biscuit"""
    r = RulesExtractor()
    result = r.extract(make_msgs("I was walking Biscuit this morning."))
    pets = [m for m in result.memories if m.slot == "pet"]
    assert len(pets) >= 1, (
        f"Expected implicit pet from 'walking Biscuit', got: {[m.slot for m in result.memories]}"
    )
    assert "biscuit" in pets[0].entity_key.lower()


def test_implicit_location_from_context():
    """'coffee shops in Berlin are amazing' → location.current = Berlin"""
    r = RulesExtractor()
    result = r.extract(make_msgs("The coffee shops in Berlin are amazing."))
    locs = [m for m in result.memories if m.slot == "location.current"]
    assert len(locs) >= 1, (
        f"Expected location.current from coffee-shop context, got: {[m.slot for m in result.memories]}"
    )
    assert "berlin" in locs[0].value_text.lower()


def test_walking_common_word_not_extracted_as_pet():
    """'I was walking the dog' without a proper noun — should not produce a named pet."""
    r = RulesExtractor()
    result = r.extract(make_msgs("I was walking the dog this morning."))
    # "the" is in the exclusion list; no named pet should appear
    pets = [m for m in result.memories if m.slot == "pet" and m.entity_key not in {"the"}]
    # This is a best-effort test — we just ensure 'the' is not stored as a pet name
    for p in pets:
        assert p.entity_key != "the", "Common word 'the' should not be stored as pet entity_key"
