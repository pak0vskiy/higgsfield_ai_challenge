import re
from .base import ExtractedMemory, ExtractionResult

# Pattern definitions: (pattern, slot, type)
# Each is tried against the user text

_PATTERNS = [
    # ── Current location ─────────────────────────────────────────────────────
    # "I live in X", "I moved to X", "I just moved to X", "I'm in X", "I'm based in X",
    # "I'm in X now", "I'm settling in X"
    (re.compile(
        r"i(?:'m|\s+am|\s+(?:just\s+)?moved?|\s+live|\s+(?:am\s+)?based)"
        r"\s+(?:to\s+|in\s+|based\s+in\s+)([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)",
        re.I),
     "location.current", "fact"),
    # Handles "I'm in London" / "I'm in London now" / "I'm settling back in London"
    (re.compile(
        r"i'm\s+(?:in|settling\s+(?:back\s+)?in)\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)",
        re.I),
     "location.current", "fact"),

    # Implicit current location: "coffee shops in Berlin are amazing" / etc.
    (re.compile(
        r"(?:coffee\s+shops?|restaurants?|streets?|city|neighbourhood|neighborhood|transit|vibes?)\s+in\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)",
        re.I),
     "location.current", "fact"),

    # ── Previous location (implied) ───────────────────────────────────────────
    # "I just got back from a year in X", "I got back from X", "returned from X"
    (re.compile(
        r"(?:just\s+)?(?:got\s+back\s+from|returned\s+from|coming\s+back\s+from)"
        r"\s+(?:a\s+\w+\s+in\s+)?([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)",
        re.I),
     "location.previous", "fact"),
    (re.compile(
        r"(?:i\s+)?(?:spent|lived\s+in)\s+(?:a\s+\w+\s+in|the\s+last\s+\w+\s+in)\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)",
        re.I),
     "location.previous", "fact"),
    (re.compile(
        r"(?:after|following)\s+(?:a\s+\w+\s+in|my\s+time\s+in)\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)",
        re.I),
     "location.previous", "fact"),
    # "I was in Tokyo for a year" / "I was living in Tokyo"
    (re.compile(
        r"i\s+was\s+(?:in|living\s+in)\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)\s+for\s+a\s+\w+",
        re.I),
     "location.previous", "fact"),

    # ── Employment ────────────────────────────────────────────────────────────
    # "I work at X", "I work for X", "I joined X", "I started at X",
    # "I just started at X", "I recently joined X", "I am at X"
    (re.compile(
        r"i\s+(?:just\s+|recently\s+|currently\s+)?(?:work\s+(?:at|for)|joined|started\s+at|am\s+at|am\s+working\s+at)"
        r"\s+([A-Z][a-zA-Z0-9\s&,]+?)(?:\s+as\b|\s+last\b|\s+now\b|\.|,|$)",
        re.I),
     "employment.current_company", "fact"),
    # "I work as a software engineer at Acme Corp" (role-first form)
    (re.compile(
        r"i\s+work\s+as\s+(?:a\s+|an\s+)?(?:[a-z][a-z\s]+?)\s+at\s+([A-Z][a-zA-Z0-9\s&,]+?)(?:\.|,|$)",
        re.I),
     "employment.current_company", "fact"),

    # Employment role: "as a X", "as an X"
    (re.compile(r"\bas\s+a[n]?\s+([a-z][a-z\s]+?)(?:\s+at\b|\s+for\b|\.|,|$)", re.I),
     "employment.current_role", "fact"),

    # ── Relationships ──────────────────────────────────────────────────────────
    # "my partner Alex", "my boyfriend/girlfriend/spouse/partner/husband/wife X"
    (re.compile(
        r"my\s+(?:partner|boyfriend|girlfriend|spouse|husband|wife)\s+([A-Z][a-z]+)",
        re.I),
     "relationship.partner", "fact"),

    # ── Pets (explicit) ───────────────────────────────────────────────────────
    # "my cat named X", "my dog X", "I have a dog named X", "a cat named X"
    (re.compile(r"(?:(?:my|i\s+have\s+a)\s+|(?:and\s+)?a\s+)(\w+)\s+named\s+([A-Z][a-z]+)", re.I),
     "pet", "fact"),
    # Pet fallback: "my dog Biscuit" (no "named")
    (re.compile(r"(?:my)\s+(\w+)\s+([A-Z][a-z]+)(?!\s+named)", re.I),
     "pet", "fact"),

    # ── Pets (implicit from activity) ────────────────────────────────────────
    # "walking Biscuit", "walking my dog Biscuit", "took Biscuit for a walk"
    (re.compile(r"\b(?:walking|walked|walk(?:ing)?)\s+(?:my\s+(?:\w+\s+)?)?([A-Z][a-z]+)\b", re.I),
     "pet_implicit", "fact"),
    (re.compile(r"\b(?:took|take|taking)\s+([A-Z][a-z]+)\s+(?:for\s+a\s+)?(?:walk|run|jog)\b", re.I),
     "pet_implicit", "fact"),

    # ── Allergy ───────────────────────────────────────────────────────────────
    (re.compile(r"(?:allergic\s+to|allergy\s+to)\s+([a-z]+(?:\s+[a-z]+)?)", re.I),
     "restriction.allergy", "fact"),

    # ── Diet ──────────────────────────────────────────────────────────────────
    (re.compile(r"(?:i'?m\s+(?:a\s+)?|i\s+eat\s+)(vegetarian|vegan|kosher|halal|pescatarian|omnivore)", re.I),
     "preference.diet", "preference"),

    # ── Response style ────────────────────────────────────────────────────────
    (re.compile(
        r"(?:i\s+prefer\s+|please\s+(?:be\s+|keep\s+(?:it\s+)?)|give\s+me\s+)"
        r"(concise|brief|detailed?|short|long|direct|formal|casual)\s*(?:answers?|responses?|replies?)?",
        re.I),
     "preference.response_style", "preference"),

    # ── Opinion (basic) ───────────────────────────────────────────────────────
    # "I really enjoy X", "I love X", "I hate X", "I think X is Y"
    # entity_key = normalised topic, value = full sentence
    (re.compile(
        r"i\s+(?:really\s+)?(?:enjoy|love|hate|dislike|prefer|like)\s+([\w\s]+?)(?:\.|,|$)",
        re.I),
     "opinion.topic", "opinion"),
]

# Animals we recognise for implicit pet extraction (walking/activity patterns)
_PET_ACTIVITY_EXCLUSIONS = {
    "the", "my", "around", "along", "down", "up", "back", "out", "in",
    "slowly", "quickly", "fast", "home", "away", "outside", "inside",
}


def _extract_from_text(text: str) -> list[ExtractedMemory]:
    memories = []
    text_lower = text.lower()

    # Check for correction signals
    is_correction = bool(re.search(
        r"\b(?:actually|not\s+\w+[,\s]+\w+|i\s+meant|sorry[,\s]+i\s+meant|correction)\b",
        text_lower
    ))

    for pattern, slot, mem_type in _PATTERNS:
        for m in pattern.finditer(text):
            groups = [g for g in m.groups() if g]
            if not groups:
                continue

            if slot == "pet":
                # groups[0] = animal type, groups[1] = name
                animal = groups[0].lower()
                if animal in {"cat", "dog", "fish", "bird", "rabbit", "hamster", "turtle", "snake", "horse"}:
                    name = groups[1] if len(groups) > 1 else animal
                    memories.append(ExtractedMemory(
                        type="fact",
                        slot="pet",
                        entity_key=name.lower(),
                        value_text=f"{animal} named {name}",
                        attributes={"species": animal, "name": name},
                        confidence=0.80,
                        evidence=m.group(0),
                        mutation="replace" if is_correction else "upsert",
                    ))

            elif slot == "pet_implicit":
                name = groups[0].strip()
                name_lower = name.lower()
                if name_lower in _PET_ACTIVITY_EXCLUSIONS:
                    continue
                if len(name) < 3 or not name[0].isupper():
                    continue
                memories.append(ExtractedMemory(
                    type="fact",
                    slot="pet",
                    entity_key=name_lower,
                    value_text=f"pet named {name}",
                    attributes={"name": name},
                    confidence=0.70,
                    evidence=m.group(0),
                    mutation="upsert",
                ))

            elif slot == "restriction.allergy":
                allergen = groups[0].strip()
                memories.append(ExtractedMemory(
                    type="fact",
                    slot="restriction.allergy",
                    entity_key=allergen.lower(),
                    value_text=f"allergic to {allergen}",
                    confidence=0.85,
                    evidence=m.group(0),
                    mutation="upsert",
                ))

            elif slot in ("location.current", "location.previous"):
                value = groups[0].strip()
                # Stop at trailing contextual words
                value = re.split(r"\s+(?:from|last|now|and|but|though|since|for)\b", value)[0].strip()
                if not value:
                    continue
                # Lower confidence for context-implied location
                conf = 0.70 if slot == "location.current" and "walking" in text_lower else 0.75
                memories.append(ExtractedMemory(
                    type=mem_type,
                    slot=slot,
                    entity_key="",
                    value_text=value,
                    confidence=conf,
                    evidence=m.group(0),
                    mutation="replace" if is_correction else "upsert",
                ))

            elif slot == "relationship.partner":
                name = groups[0].strip()
                memories.append(ExtractedMemory(
                    type="fact",
                    slot="relationship.partner",
                    entity_key="",
                    value_text=name,
                    confidence=0.90,
                    evidence=m.group(0),
                    mutation="replace" if is_correction else "upsert",
                ))

            elif slot == "opinion.topic":
                topic_raw = groups[0].strip().lower()
                # Truncate very long opinion values
                if len(topic_raw) > 60:
                    continue
                # Skip very short / stopword-only opinions
                if len(topic_raw.split()) == 0:
                    continue
                # Use first two significant words as entity_key
                topic_words = [w for w in topic_raw.split() if len(w) > 2][:3]
                entity_key = " ".join(topic_words) if topic_words else topic_raw
                # Full opinion as value
                value = m.group(0).strip()
                memories.append(ExtractedMemory(
                    type="opinion",
                    slot="opinion.topic",
                    entity_key=entity_key,
                    value_text=value,
                    confidence=0.70,
                    evidence=m.group(0),
                    mutation="upsert",
                ))

            else:
                value = groups[0].strip()
                memories.append(ExtractedMemory(
                    type=mem_type,
                    slot=slot,
                    entity_key="",
                    value_text=value,
                    confidence=0.75,
                    evidence=m.group(0),
                    mutation="replace" if is_correction else "upsert",
                ))

    return memories


class RulesExtractor:
    def extract(self, messages: list[dict], user_context: str = "") -> ExtractionResult:
        """Extract memories from messages using regex patterns."""
        all_memories: list[ExtractedMemory] = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                all_memories.extend(_extract_from_text(content))

        # Deduplicate by (slot, entity_key) — keep highest confidence
        seen: dict[tuple, ExtractedMemory] = {}
        for m in all_memories:
            key = (m.slot, m.entity_key)
            if key not in seen or m.confidence > seen[key].confidence:
                seen[key] = m

        return ExtractionResult(
            memories=list(seen.values()),
            extractor_used="rules",
        )
