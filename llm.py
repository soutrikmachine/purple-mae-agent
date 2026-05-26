"""
Optional LLM refinement layer.

Provider-agnostic via the OpenAI Python SDK with base_url override.
Default provider: OpenRouter (https://openrouter.ai/api/v1).

Whatever the LLM returns is filtered through strategy.sanitise_proposal and
sanitise_decision, so it cannot introduce M1-M5 violations. The deterministic
baseline is always armed as fallback.

When USE_LLM=false (default), this module is a pass-through with no LLM calls
and no external dependencies fetched.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import strategy
from strategy import GameState

logger = logging.getLogger("purple_mae.llm")

USE_LLM = os.environ.get("USE_LLM", "false").lower() in ("1", "true", "yes")
# Path A (asymmetric LLM use): individually gate the propose and decide
# paths. Each defaults to USE_LLM, so existing configs are unchanged. To
# isolate the LLM's contribution to one path, set the other to "false".
LLM_PROPOSE_ENABLED = os.environ.get(
    "LLM_PROPOSE_ENABLED", str(USE_LLM).lower()
).lower() in ("1", "true", "yes")
LLM_DECIDE_ENABLED = os.environ.get(
    "LLM_DECIDE_ENABLED", str(USE_LLM).lower()
).lower() in ("1", "true", "yes")
PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").lower()
MODEL = os.environ.get(
    "LLM_MODEL",
    "anthropic/claude-opus-4.7" if PROVIDER == "openrouter" else "claude-sonnet-4-20250514",
)
TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "45"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))


_SYSTEM_PROMPT = """You are a strategic advisor for a bargaining agent.
You receive game state and a baseline action computed by a game-theoretic strategy.
You may propose a refinement to the baseline; otherwise repeat the baseline.

Any refinement MUST satisfy:
  1. Self-value of any proposal >= BATNA + 1
  2. Each side keeps at least one item (no zero-sum allocations)
  3. Never accept an offer below BATNA

Respond with VALID JSON only, no prose:
  PROPOSE: {"allocation_self": [...], "allocation_other": [...], "rationale": "..."}
  DECIDE:  {"accept": true|false, "rationale": "..."}"""


_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not USE_LLM:
        return None

    if PROVIDER == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
        default_headers = {
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://agentbeats.dev"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Purple MAE Agent"),
        }
    elif PROVIDER == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = "https://api.anthropic.com/v1"
        default_headers = {}
    else:
        logger.error("Unknown LLM_PROVIDER=%s; LLM layer disabled.", PROVIDER)
        return None

    if not api_key:
        logger.warning(
            "USE_LLM=true but no API key for provider=%s; LLM layer disabled.",
            PROVIDER,
        )
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai SDK not installed; LLM layer disabled.")
        return None

    _client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=TIMEOUT_S,
        max_retries=1,
        default_headers=default_headers,
    )
    logger.info(
        "LLM client ready: provider=%s model=%s timeout=%.1fs",
        PROVIDER, MODEL, TIMEOUT_S,
    )
    return _client


def _state_summary(state: GameState) -> dict[str, Any]:
    return {
        "role": state.role,
        "round": state.round,
        "max_rounds": state.max_rounds,
        "discount": state.discount,
        "valuations_self": state.valuations_self,
        "batna_self": state.batna_self,
        "quantities": state.quantities,
        "current_offer_to_self": state.current_offer_to_self,
        "current_offer_to_other": state.current_offer_to_other,
    }


def _chat(user_payload: dict[str, Any]) -> dict[str, Any]:
    client = _get_client()
    if client is None:
        return {}
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        return _extract_json(text)
    except Exception as exc:
        logger.warning("LLM call failed (%s); using deterministic baseline.", exc)
        return {}


def refine_proposal(
    state: GameState,
    baseline_self: list[int],
    baseline_other: list[int],
    baseline_reason: str,
) -> tuple[list[int], list[int], str]:
    # Path A asymmetric gate: skip the LLM call entirely if the propose
    # path is disabled, even when USE_LLM=true.
    if not (USE_LLM and LLM_PROPOSE_ENABLED):
        return baseline_self, baseline_other, baseline_reason

    parsed = _chat({
        "task": "PROPOSE",
        "state": _state_summary(state),
        "baseline": {
            "allocation_self": baseline_self,
            "allocation_other": baseline_other,
            "rationale": baseline_reason,
        },
    })
    if not parsed:
        return baseline_self, baseline_other, baseline_reason

    cand_self = parsed.get("allocation_self", baseline_self)
    cand_other = parsed.get("allocation_other", baseline_other)
    rationale = parsed.get("rationale", baseline_reason)
    safe_self, safe_other = strategy.sanitise_proposal(state, cand_self, cand_other)
    return safe_self, safe_other, f"LLM: {rationale}"


def refine_decision(
    state: GameState,
    baseline_accept: bool,
    baseline_reason: str,
    offer_value_override: float | None = None,
) -> tuple[bool, str]:
    # Path A asymmetric gate.
    if not (USE_LLM and LLM_DECIDE_ENABLED):
        return baseline_accept, baseline_reason

    parsed = _chat({
        "task": "DECIDE",
        "state": _state_summary(state),
        "baseline": {"accept": baseline_accept, "rationale": baseline_reason},
    })
    if not parsed:
        return baseline_accept, baseline_reason

    cand_accept = bool(parsed.get("accept", baseline_accept))
    rationale = parsed.get("rationale", baseline_reason)
    safe = strategy.sanitise_decision(state, cand_accept, offer_value_override=offer_value_override)
    return safe, f"LLM: {rationale}"


def _extract_json(text: str) -> dict[str, Any]:
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
