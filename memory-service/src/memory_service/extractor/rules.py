import re
from .base import ExtractedMemory, ExtractionResult

# Pattern definitions: (pattern, slot, type)
# Each is tried against the user text

_PATTERNS = [
    # Location: "I live in X", "I moved to X", "I just moved to X", "I'm in X", "I'm based in X"
    (re.compile(r"i\s+(?:\w+\s+)?(?:live|moved?|am|'m)\s+(?:to|in|based\s+in)\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)", re.I),
     "location.current", "fact"),

    # Employment company: "I work at X", "I work for X", "I joined X", "I started at X"
    (re.compile(r"(?:i\s+(?:work\s+(?:at|for)|joined|started\s+at|am\s+at))\s+([A-Z][a-zA-Z0-9\s&,]+?)(?:\s+as\b|\s+last\b|\s+now\b|\.|,|$)", re.I),
     "employment.current_company", "fact"),

    # Employment role: "as a X", "as an X" (only when company was also mentioned)
    (re.compile(r"\bas\s+a[n]?\s+([a-z][a-z\s]+?)(?:\s+at\b|\s+for\b|\.|,|$)", re.I),
     "employment.current_role", "fact"),

    # Pet: "my cat named X", "my dog X", "I have a dog named X", "a cat named X"
    (re.compile(r"(?:(?:my|i\s+have\s+a)\s+|(?:and\s+)?a\s+)(\w+)\s+named\s+([A-Z][a-z]+)", re.I),
     "pet", "fact"),
    # Pet fallback: "my dog Biscuit" (no "named")
    (re.compile(r"(?:my)\s+(\w+)\s+([A-Z][a-z]+)(?!\s+named)", re.I),
     "pet", "fact"),

    # Allergy: "allergic to X", "allergy to X", "I can't eat X"
    (re.compile(r"(?:allergic\s+to|allergy\s+to)\s+([a-z]+(?:\s+[a-z]+)?)", re.I),
     "restriction.allergy", "fact"),

    # Diet: "I'm vegetarian", "I'm vegan", "I eat kosher"
    (re.compile(r"(?:i'?m\s+(?:a\s+)?|i\s+eat\s+)(vegetarian|vegan|kosher|halal|pescatarian|omnivore)", re.I),
     "preference.diet", "preference"),

    # Response style: "I prefer concise", "keep it brief", "be detailed"
    (re.compile(r"(?:i\s+prefer\s+|please\s+(?:be\s+|keep\s+(?:it\s+)?)|give\s+me\s+)(concise|brief|detailed?|short|long|direct|formal|casual)\s*(?:answers?|responses?|replies?)?", re.I),
     "preference.response_style", "preference"),
]


def _extract_from_text(text: str) -> list[ExtractedMemory]:
    memories = []
    text_lower = text.lower()

    # Check for correction signals
    is_correction = bool(re.search(r"\b(?:actually|not\s+\w+[,\s]+\w+|i\s+meant|sorry[,\s]+i\s+meant|correction)\b", text_lower))

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
            else:
                value = groups[0].strip()
                # For location, stop at trailing contextual words
                if slot == "location.current":
                    value = re.split(r"\s+(?:from|last|now|and|but|though|since)\b", value)[0].strip()
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
