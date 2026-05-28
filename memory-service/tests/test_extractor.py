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
