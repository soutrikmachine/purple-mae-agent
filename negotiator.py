"""
Dispatcher for the Purple MAE agent.

The green agent (MAizeBargAIn's `RemoteNegotiator` in agents/remote.py) talks
to us over a stateful A2A conversation via `ToolProvider.talk_to_agent`. Each
message contains a prompt of this form:

    You are participating in the AgentBeats bargaining meta-game as 'challenger'.
    Action: PROPOSE.
    <instruction text>
    Always answer with valid JSON only.
    Observation:
    ```json
    {
      "pair": "challenger__vs__soft",
      "game_index": 7,
      "role": "row",
      "valuations_self": [60, 30, 90],
      "batna_self": 80,
      "discount": 0.98,
      "max_rounds": 5,
      "quantities": [7, 4, 1],
      "round_index": 1,
      "player_index": 0,
      "my_player_num": 1,
      "other_player_num": 2,
      "p1_outside_offer": [1, 800],
      "value_cap": 100,
      "example_offer": [3, 2, 0],
      "batna_player1": 80,
      "batna_player2": 0,
      "action": "propose",
      "pending_offer": { "offer_allocation_self": [3, 2, 1],
                         "offer_allocation_opp":  [4, 2, 0], ... }   (decisions only)
    }
    ```
    Allocation catalog (use `choice_id` ...): [...]
    <optional circle prompt text>

We respond with JSON (no fences needed; the green strips them anyway):

    PROPOSE   -> {"allocation_self": [...], "allocation_other": [...], "reason": "..."}
    ACCEPT    -> {"accept": true,  "reason": "..."}
    REJECT    -> {"accept": false, "reason": "...", "plan_allocation": [...optional...]}

The opponent's valuations and BATNA are STRIPPED from the observation by the
green (see remote.py:_build_observation lines 213-217). We rely on the
U[1,100] prior for opponent modelling.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import strategy
from strategy import GameState
import llm
try:
    import rl_proposer  # optional, no-op if module is the stub
    _HAVE_RL = True
except ImportError:
    _HAVE_RL = False
from session_store import session_store, SessionStore

logger = logging.getLogger("purple_mae.negotiator")


# ---------------------------------------------------------------------------
# Parsing the green's prompt
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of the prompt."""
    # Prefer fenced ```json``` blocks.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: outermost balanced braces.
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _detect_action(text: str, observation: dict[str, Any]) -> str:
    """
    Returns 'PROPOSE' or 'ACCEPT_OR_REJECT'.

    The green writes both an 'Action: X' line in the prompt header AND an
    'action' key in the JSON observation. We trust the JSON (lowercase) and
    fall back to the header (uppercase) and then to heuristics.
    """
    # JSON observation 'action' field is most reliable.
    obs_action = str(observation.get("action", "")).strip().lower()
    if obs_action == "propose":
        return "PROPOSE"
    if obs_action == "accept_or_reject":
        return "ACCEPT_OR_REJECT"

    # Prompt header line.
    upper = text.upper()
    if "ACTION: ACCEPT_OR_REJECT" in upper or "ACTION: ACCEPT/REJECT" in upper:
        return "ACCEPT_OR_REJECT"
    if "ACTION: PROPOSE" in upper:
        return "PROPOSE"

    # Heuristic: pending_offer present means we're being asked to decide.
    if observation.get("pending_offer"):
        return "ACCEPT_OR_REJECT"
    return "PROPOSE"


def _coerce_state(obs: dict[str, Any]) -> GameState | None:
    """Build a GameState from the green's observation dict."""
    if not obs:
        return None
    try:
        quantities = obs.get("quantities") or [7, 4, 1]
        valuations = obs.get("valuations_self") or [0] * len(quantities)
        batna = float(obs.get("batna_self", 0.0))

        # round_index is the actual field; round is the legacy name.
        round_idx = int(obs.get("round_index", obs.get("round", 1)))
        max_rounds = int(obs.get("max_rounds", 5))
        discount = float(obs.get("discount", 0.98))
        role = str(obs.get("role", "row"))

        # When deciding, the offer is under pending_offer.
        # See remote.py:set_offer_context — the proposer fills:
        #   offer_allocation_self : what WE (the receiver) keep if we accept
        #   offer_allocation_opp  : what the PROPOSER keeps
        offer_self: list[int] | None = None
        offer_other: list[int] | None = None
        pending = obs.get("pending_offer")
        if isinstance(pending, dict):
            if "offer_allocation_self" in pending:
                offer_self = [int(x) for x in pending["offer_allocation_self"]]
            # remote.py uses "offer_allocation_opp" (not "_other")
            if "offer_allocation_opp" in pending:
                offer_other = [int(x) for x in pending["offer_allocation_opp"]]
            elif "offer_allocation_other" in pending:
                offer_other = [int(x) for x in pending["offer_allocation_other"]]

        # Derive complement when only one side is given.
        if offer_self is not None and offer_other is None:
            offer_other = [
                int(q) - int(s) for q, s in zip(quantities, offer_self)
            ]

        return GameState(
            role=role,
            round=round_idx,
            max_rounds=max_rounds,
            discount=discount,
            valuations_self=[int(v) for v in valuations],
            batna_self=batna,
            quantities=[int(q) for q in quantities],
            current_offer_to_self=offer_self,
            current_offer_to_other=offer_other,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Failed to parse observation: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def handle_negotiation_message(message_text: str) -> dict[str, Any]:
    """
    Parse the green's prompt and return the action JSON.

    The session key is derived from the observation's (pair, game_index), not
    from any A2A context_id, so memory resets cleanly between games.
    """
    obs = _extract_json_block(message_text)
    state = _coerce_state(obs)
    action = _detect_action(message_text, obs)

    if state is None:
        # Unparseable. Emit a benign default: reject without accepting; the
        # green's RemoteNegotiator will treat this as a counteroffer attempt
        # and use its fallback heuristic for us this turn.
        logger.warning("Unparseable observation; defaulting to safe response.")
        if action == "PROPOSE":
            return {
                "allocation_self": [],
                "allocation_other": [],
                "reason": "unparseable observation",
            }
        return {"accept": False, "reason": "unparseable observation"}

    # Per-game M1 anchor.
    session_key = SessionStore.make_key(obs.get("pair"), obs.get("game_index"))
    state.previous_self_offer_value = session_store.get_last_self_value(session_key)

    if action == "PROPOSE":
        return _do_propose(state, session_key)
    # Pass the green-computed offer_value if present (avoids recomputation).
    offer_value_override = obs.get("offer_value")
    return _do_decide(state, offer_value_override)


def _do_propose(state: GameState, session_key: str | None) -> dict[str, Any]:
    # 1. Deterministic baseline.
    alloc_self, alloc_other, reason = strategy.propose(state)

    # 2. Optional RL candidate (uses NFSP/RNAD checkpoint if available).
    if _HAVE_RL and rl_proposer.is_enabled():
        rl_alloc = rl_proposer.maybe_propose(state)
        if rl_alloc is not None:
            rl_self, rl_other = rl_alloc
            # Pick the candidate with higher self-value, after sanitising.
            rl_self, rl_other = strategy.sanitise_proposal(state, rl_self, rl_other)
            if (
                strategy.offer_value(rl_self, state.valuations_self)
                > strategy.offer_value(alloc_self, state.valuations_self)
            ):
                alloc_self, alloc_other = rl_self, rl_other
                reason = f"RL: {reason}"

    # 3. Optional LLM refinement (clamped by sanitisers below).
    alloc_self, alloc_other, reason = llm.refine_proposal(
        state, alloc_self, alloc_other, reason
    )

    # 4. Final clamp.
    alloc_self, alloc_other = strategy.sanitise_proposal(state, alloc_self, alloc_other)

    # 5. Record self-value for M1 next round.
    self_v = strategy.offer_value(alloc_self, state.valuations_self)
    session_store.set_last_self_value(session_key, self_v)

    return {
        "allocation_self": alloc_self,
        "allocation_other": alloc_other,
        "reason": reason,
    }


def _do_decide(state: GameState, offer_value_override: float | None = None) -> dict[str, Any]:
    # 1. Deterministic baseline.
    accept, reason = strategy.decide_offer(state, offer_value_override=offer_value_override)

    # 2. Optional LLM refinement.
    accept, reason = llm.refine_decision(
        state, accept, reason, offer_value_override=offer_value_override
    )

    # 3. Final clamp (M4, M5).
    accept = strategy.sanitise_decision(state, accept, offer_value_override=offer_value_override)

    # When rejecting, attach a planned counter-allocation as a hint to the
    # green. This is optional but reduces parser ambiguity.
    if accept:
        return {"accept": True, "reason": reason}
    plan_self, _plan_other, _ = strategy.propose(state)
    return {
        "accept": False,
        "reason": reason,
        "plan_allocation": plan_self,
    }
