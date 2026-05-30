import json, os, logging
from typing import Any
import litellm
from pydantic import ValidationError
from .base import ExtractedMemory, ExtractionResult, MemoryType, MutationIntent
from .rules import RulesExtractor
from memory_service.slots import SLOT_LIST_FOR_PROMPT

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a memory extraction engine for an AI assistant.
Extract structured facts from the conversation below.

IMPORTANT RULES:
1. Only extract facts stated or strongly implied by the USER. Do not extract assistant statements as user facts.
2. Use ONLY the slot names from the canonical list below. Use "unstructured" for anything that doesn't fit.
3. For collection slots (pet, family_member, restriction.allergy, skill.using, etc.), set entity_key to the name/identifier.
4. For singleton slots (location.current, employment.current_company, etc.), leave entity_key as "".
5. Set mutation to "replace" or "correction" if the user is explicitly correcting a prior statement.
6. Confidence: 0.9+ for explicit, 0.7-0.9 for strong inference, 0.5-0.7 for weak inference. Omit if confidence < 0.5.

IMPLIED FACTS — extract these when a careful reader would be confident:
- "walking Biscuit this morning" → pet named Biscuit (confidence 0.85)
- "coffee shops in Berlin are amazing" → location.current = Berlin (confidence 0.70)
- "I just got back from a year in Tokyo" → location.previous = Tokyo (confidence 0.88)
- "returned from living in Paris" → location.previous = Paris (confidence 0.88)
- "spent two years in Singapore before moving here" → location.previous = Singapore (confidence 0.85)
- "my partner Alex" → relationship.partner = Alex (confidence 0.90)
- ONLY extract an implied fact if a careful reader would be confident. Never speculate.

CORRECTIONS — handle explicitly:
- "actually I moved to Munich, not Berlin" → location.current = Munich (mutation: replace), negate Berlin
- "sorry, I meant X not Y" → extract X, negate Y

CANONICAL SLOT LIST:
{slot_list}

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown:
{{
  "memories": [
    {{
      "type": "fact|preference|opinion|event|correction",
      "slot": "<canonical slot or unstructured>",
      "entity_key": "<identifier for collection slots, else empty string>",
      "value_text": "<human-readable value>",
      "attributes": {{}},
      "confidence": 0.95,
      "evidence": "<verbatim span from conversation>",
      "mutation": "upsert|replace|append|negate",
      "supersedes_hint": null
    }}
  ]
}}

CURRENT KNOWN FACTS ABOUT THIS USER (for context, to detect supersession):
{user_context}
"""

_rules_fallback = RulesExtractor()


def _build_messages_text(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        name = msg.get("name", "")
        if role == "tool" and name:
            lines.append(f"[tool:{name}]: {content}")
        else:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _parse_llm_response(raw: str) -> list[ExtractedMemory]:
    """Parse and validate LLM JSON response. Returns valid memories only."""
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            raise

    memories = []
    for item in data.get("memories", []):
        try:
            mem = ExtractedMemory(**item)
            if mem.confidence >= 0.5:  # filter very low confidence at parse time
                memories.append(mem)
        except (ValidationError, TypeError) as e:
            logger.warning("Skipping invalid memory candidate: %s — %s", item, e)
    return memories


class LLMExtractor:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openrouter")
        self.model = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
        self.fallback_model = os.getenv("LLM_FALLBACK_MODEL", "gpt-4o-mini")
        self._rules = RulesExtractor()

    def _get_model_string(self, model: str) -> str:
        """Convert model name to litellm format."""
        if self.provider == "openrouter":
            return f"openrouter/{model}"
        elif self.provider == "deepseek":
            return f"deepseek/{model}"
        elif self.provider == "openai":
            return model
        elif self.provider == "anthropic":
            return f"anthropic/{model}"
        return model

    def extract(self, messages: list[dict], user_context: str = "") -> ExtractionResult:
        conversation_text = _build_messages_text(messages)
        system = _SYSTEM_PROMPT.format(
            slot_list=SLOT_LIST_FOR_PROMPT,
            user_context=user_context or "None",
        )

        for attempt, model_str in enumerate([
            self._get_model_string(self.model),
            self.fallback_model,  # OpenAI gpt-4o-mini as fallback
        ]):
            try:
                response = litellm.completion(
                    model=model_str,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Extract memories from this conversation:\n\n{conversation_text}"},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=30,
                )
                raw = response.choices[0].message.content
                memories = _parse_llm_response(raw)
                logger.info("LLM extracted %d memories using %s", len(memories), model_str)
                return ExtractionResult(
                    memories=memories,
                    extractor_used="llm",
                    raw_llm_response=raw,
                )
            except Exception as e:
                logger.warning("LLM extraction attempt %d failed (%s): %s", attempt + 1, model_str, e)

        # Both LLM attempts failed — fall back to rules
        logger.warning("LLM extraction failed, falling back to rules")
        result = self._rules.extract(messages, user_context)
        return ExtractionResult(
            memories=result.memories,
            extractor_used="rules_fallback",
        )


def get_extractor():
    """Return the appropriate extractor based on env config."""
    extractor_type = os.getenv("MEMORY_EXTRACTOR", "llm").lower()
    if extractor_type == "rules":
        return RulesExtractor()
    return LLMExtractor()
