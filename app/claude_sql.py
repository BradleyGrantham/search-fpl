"""Question -> SQL with Claude Haiku 5.5 (Anthropic SDK), built around prompt caching.

The system prompt (rules + schema + examples, ~2.1K tokens) is identical for every request, so
it is cached with a 1-hour TTL: site traffic is bursty, and the 1-hour entry survives the
5-60 minute gaps that would expire the default 5-minute one. Each search then sends only the
question (~15-30 uncached tokens). `warm_up` writes the cache at startup so the first visitor
gets a cache hit too.
"""

import anthropic

MODEL = "claude-haiku-5-5"
# Thinking off: measured as accurate as adaptive/low on our questions, and faster and steadier
# (0.8-1.0s vs up to 1.8s). Thinking settings are part of the cache key, so warm_up must match.
THINKING = {"type": "disabled"}
MAX_TOKENS = 1024  # SQL is ~80-200 tokens

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Fail fast: a search box is useless after a few seconds (SDK default is 10 min, 2 retries).
        _client = anthropic.Anthropic(timeout=8.0, max_retries=1)
    return _client


def _system(prompt: str) -> list[dict]:
    return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def warm_up(prompt: str) -> dict:
    """Write the system prompt to the cache without generating anything (max_tokens=0)."""
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=0,
        thinking=THINKING,
        system=_system(prompt),
        messages=[{"role": "user", "content": "warmup"}],
    )
    return _usage(response)


def generate_sql(prompt: str, question: str, failed_sql: str | None = None, error: str | None = None):
    """Returns (raw model text, usage dict)."""
    messages = [{"role": "user", "content": question}]
    if failed_sql:
        messages += [
            {"role": "assistant", "content": failed_sql},
            {"role": "user", "content": f"That failed with: {error}\nFix it. Output only SQL."},
        ]
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking=THINKING,
        system=_system(prompt),
        messages=messages,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request ({response.stop_details})")
    text = next((block.text for block in response.content if block.type == "text"), "")
    return text, _usage(response)


def _usage(response) -> dict:
    u = response.usage
    return {
        "input": u.input_tokens,
        "cache_read": u.cache_read_input_tokens or 0,
        "cache_write": u.cache_creation_input_tokens or 0,
        "output": u.output_tokens,
    }
