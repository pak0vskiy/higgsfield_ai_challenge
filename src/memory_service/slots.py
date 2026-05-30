from enum import Enum
from typing import NamedTuple


class SlotTier(str, Enum):
    SINGLETON = "singleton"   # one active per user_id; auto-supersession
    COLLECTION = "collection" # multiple per user_id; entity_key disambiguates
    UNSTRUCTURED = "unstructured"  # escape hatch


class SlotDef(NamedTuple):
    slot: str
    tier: SlotTier
    description: str
    # For SINGLETON slots that have a paired "previous" slot
    previous_slot: str | None = None


SLOT_CATALOG: list[SlotDef] = [
    # Tier 1 — canonical singletons
    SlotDef("identity.name",                SlotTier.SINGLETON,   "person's full name"),
    SlotDef("identity.age",                 SlotTier.SINGLETON,   "person's age"),
    SlotDef("identity.pronouns",            SlotTier.SINGLETON,   "preferred pronouns"),
    SlotDef("location.current",             SlotTier.SINGLETON,   "current city/country of residence", previous_slot="location.previous"),
    SlotDef("location.previous",            SlotTier.SINGLETON,   "previous location (auto-set on supersession)"),
    SlotDef("location.hometown",            SlotTier.SINGLETON,   "hometown"),
    SlotDef("employment.current_company",   SlotTier.SINGLETON,   "current employer", previous_slot="employment.previous_company"),
    SlotDef("employment.current_role",      SlotTier.SINGLETON,   "current job title/role"),
    SlotDef("employment.previous_company",  SlotTier.SINGLETON,   "previous employer (auto-set)"),
    SlotDef("relationship.partner",         SlotTier.SINGLETON,   "romantic partner"),
    SlotDef("preference.response_style",    SlotTier.SINGLETON,   "preferred response style (concise, detailed, etc.)"),
    SlotDef("preference.communication_style", SlotTier.SINGLETON, "communication preference"),
    SlotDef("preference.diet",              SlotTier.SINGLETON,   "dietary preference (vegetarian, vegan, etc.)"),

    # Tier 2 — entity-keyed collections
    SlotDef("pet",                          SlotTier.COLLECTION,  "pet (entity_key = pet name)"),
    SlotDef("family_member",                SlotTier.COLLECTION,  "family member (entity_key = relation or name)"),
    SlotDef("restriction.allergy",          SlotTier.COLLECTION,  "food or other allergy (entity_key = allergen)"),
    SlotDef("skill.using",                  SlotTier.COLLECTION,  "technology or skill the person uses"),
    SlotDef("preference.food",              SlotTier.COLLECTION,  "food preference (entity_key = food item)"),
    SlotDef("opinion.topic",                SlotTier.COLLECTION,  "opinion on a topic (entity_key = topic)"),
    SlotDef("project.current",              SlotTier.COLLECTION,  "current project (entity_key = project name)"),
    SlotDef("event.upcoming",               SlotTier.COLLECTION,  "upcoming event (entity_key = event description)"),
]

# Build lookup dicts for fast access
_by_slot: dict[str, SlotDef] = {s.slot: s for s in SLOT_CATALOG}
_singleton_slots: set[str] = {s.slot for s in SLOT_CATALOG if s.tier == SlotTier.SINGLETON}
_collection_slots: set[str] = {s.slot for s in SLOT_CATALOG if s.tier == SlotTier.COLLECTION}


def get_slot(slot: str) -> SlotDef | None:
    return _by_slot.get(slot)


def is_singleton(slot: str) -> bool:
    return slot in _singleton_slots


def is_collection(slot: str) -> bool:
    return slot in _collection_slots


def is_valid_slot(slot: str) -> bool:
    return slot in _by_slot or slot == "unstructured"


def get_previous_slot(slot: str) -> str | None:
    sd = _by_slot.get(slot)
    return sd.previous_slot if sd else None


# Canonical prompt description for the LLM
SLOT_LIST_FOR_PROMPT = "\n".join(
    f"  {s.slot} ({s.tier.value}): {s.description}"
    for s in SLOT_CATALOG
)
