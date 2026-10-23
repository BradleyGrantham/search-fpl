"""Question -> SQL with Claude Haiku 5.5, built around prompt caching.

The system prompt (~2.1K tokens) is the same for every search, so it is cached with a 1-hour
TTL; each search pays full price only for the question. The cache lives on Anthropic's side
and is shared by every Worker instance. The scheduled refresh re-warms it whenever the prompt
changes (e.g. a new gameweek).

`anthropic` is imported at module level on purpose: Python Workers snapshot memory after
top-level imports at deploy time, so the SDK's import cost isn't paid per request.
"""

import anthropic

MODEL = "claude-haiku-5-5"
# Thinking off: as accurate as adaptive/low on our questions, and faster and steadier.
# Thinking settings are part of the cache key, so warm_up must use the same value.
THINKING = {"type": "disabled"}
MAX_TOKENS = 1024  # SQL is ~80-200 tokens

_client: anthropic.AsyncAnthropic | None = None


def _get_client(api_key: str) -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        # Fail fast: a search box is useless after a few seconds (SDK default is 10 min, 2 retries).
        _client = anthropic.AsyncAnthropic(api_key=api_key, timeout=8.0, max_retries=1)
    return _client


def _system(prompt: str) -> list[dict]:
    return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


async def warm_up(api_key: str, prompt: str) -> dict:
    """Write the system prompt to the cache without generating anything (max_tokens=0)."""
    response = await _get_client(api_key).messages.create(
        model=MODEL,
        max_tokens=0,
        thinking=THINKING,
        system=_system(prompt),
        messages=[{"role": "user", "content": "warmup"}],
    )
    return _usage(response)


async def generate_sql(api_key: str, prompt: str, question: str,
                       failed_sql: str | None = None, error: str | None = None):
    """Returns (raw model text, usage dict)."""
    messages = [{"role": "user", "content": question}]
    if failed_sql:
        messages += [
            {"role": "assistant", "content": failed_sql},
            {"role": "user", "content": f"That failed with: {error}\nFix it. Output only SQL."},
        ]
    response = await _get_client(api_key).messages.create(
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
