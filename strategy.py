"""
Deterministic negotiation strategy for the Purple MAE agent.

This module is the agent's reasoning core. By construction it never commits
the five negotiation mistakes from Smithline et al. (2025):

    M1: propose worse-for-self than your own previous offer
    M2: propose worse-for-self than your BATNA
    M3: propose degenerate divisions ([0,...,0] or all-of-Q)
    M4: accept below BATNA
    M5: walk away from above-BATNA offer in the final round

Even when the optional LLM layer is enabled, the LLM's output is filtered
through sanitise_proposal / sanitise_decision, so violations cannot escape.

Game model (OpenSpiel bargaining):
    - T item types with public quantities Q = (q_1, ..., q_T), typically (7, 4, 1)
    - Private valuations v_self in [1, 100]^T, opponent valuations NOT observable
    - Private BATNA b_self (outside option, scalar)
    - Discount gamma in (0, 1], up to R rounds (lightweight runner stops at 2)
    - Opponent valuations modelled as Uniform[1, 100]^T (per the green's sampler)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


# Expected per-unit opponent valuation under U[1, 100] prior.
EXPECTED_OPPONENT_VAL_PER_UNIT = 50.5

# Welfare/regret trade-off knob. Round-1 aspiration ceiling is
#     OPENING_AGGRESSIVENESS * max_attainable + (1 - it) * (BATNA + 1)
# Empirical sweet spot from the leaderboard: 0.75 (welfare-friendly, low regret).
OPENING_AGGRESSIVENESS = float(os.environ.get("OPENING_AGGRESSIVENESS", "0.805"))


@dataclass
class GameState:
    """Parsed game state from a green-agent observation."""

    role: str                       # "row" or "col"
    round: int                      # 1-indexed
    max_rounds: int
    discount: float
    valuations_self: list[int]
    batna_self: float
    quantities: list[int]
    # When set, we are deciding on an offer rather than originating a proposal.
    current_offer_to_self: list[int] | None = None
    current_offer_to_other: list[int] | None = None
    # M1 anchor (set externally from the session store).
    previous_self_offer_value: float | None = None
    # Identifiers used to seed the quasi-random tie-breaker. Same game ->
    # same seed -> same action (reproducibility); different games -> different
    # seeds -> mixed strategy across the empirical game matrix.
    pair_key: str | None = None
    game_index: int | None = None

    @property
    def total_items(self) -> int:
        return sum(self.quantities)

    @property
    def num_items_types(self) -> int:
        return len(self.quantities)


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def offer_value(allocation: Sequence[int], valuations: Sequence[float]) -> float:
    """Dot-product value of an allocation under given valuations."""
    return float(sum(a * v for a, v in zip(allocation, valuations)))


def estimated_opponent_value(allocation_to_other: Sequence[int]) -> float:
    """Expected opponent value of the allocation, under the U[1,100] prior."""
    return EXPECTED_OPPONENT_VAL_PER_UNIT * sum(allocation_to_other)


def is_degenerate(allocation: Sequence[int], quantities: Sequence[int]) -> bool:
    total = sum(allocation)
    return total == 0 or total == sum(quantities)


def max_attainable_value(state: GameState) -> float:
    """Value of keeping everything (upper bound for aspiration)."""
    return offer_value(state.quantities, state.valuations_self)


# ---------------------------------------------------------------------------
# Aspiration / concession schedule
# ---------------------------------------------------------------------------

def aspiration_target(state: GameState, max_attainable: float) -> float:
    """
    Time-decayed and discount-aware aspiration target value.

    Returns the minimum self-value we are willing to propose this round.
    Starts at OPENING_AGGRESSIVENESS * max + (1 - it) * (BATNA + 1), concedes
    linearly toward BATNA over R rounds, with extra concession when waiting
    is costly (gamma^remaining is small).
    """
    floor = state.batna_self + 1.0
    raw_ceiling = (
        OPENING_AGGRESSIVENESS * max_attainable
        + (1.0 - OPENING_AGGRESSIVENESS) * floor
    )
    ceiling = max(raw_ceiling, floor + 1.0)

    progress = (state.round - 1) / max(1, state.max_rounds - 1)
    remaining = max(0, state.max_rounds - state.round)
    waiting_penalty = 1.0 - (state.discount ** remaining)
    effective_progress = min(1.0, progress + 0.5 * waiting_penalty)

    target = ceiling - (ceiling - floor) * effective_progress
    return max(floor, target)


# ---------------------------------------------------------------------------
# Proposal generation
# ---------------------------------------------------------------------------

def _greedy_split(
    state: GameState, target_value: float
) -> tuple[list[int], list[int]]:
    """
    Build (alloc_self, alloc_other) by assigning each unit to whichever side
    values it more strongly *relative to the prior*, while ensuring self gets
    at least `target_value`.

    Since we don't observe opponent valuations, the "relative" criterion uses
    the prior mean (50.5) for the opponent — which means we keep items where
    we have a comparative advantage and hand back items where we don't.
    """
    T = state.num_items_types
    alloc_self = [0] * T
    alloc_other = list(state.quantities)

    # Item types in descending order of self-priority.
    order = sorted(
        range(T),
        key=lambda i: state.valuations_self[i] - EXPECTED_OPPONENT_VAL_PER_UNIT,
        reverse=True,
    )

    for i in order:
        while alloc_other[i] > 0:
            current = offer_value(alloc_self, state.valuations_self)
            if current >= target_value:
                break
            alloc_self[i] += 1
            alloc_other[i] -= 1
        if offer_value(alloc_self, state.valuations_self) >= target_value:
            break

    return alloc_self, alloc_other


# ---------------------------------------------------------------------------
# Quasi-random helpers
# ---------------------------------------------------------------------------
# Path 2 motivation: a fully deterministic spine is exploitable by best-
# responders (nfsp, rnad) and pushed off the MENE support by the solver.
# A truly randomised policy fixes that, but the user requires reproducibility.
#
# We use a *seeded* PRNG where the seed is a hash of the observation. Same
# observation -> same action (reproducibility); but the seed varies across
# games (different game_index, different valuations), so when the green's
# matrix-builder averages our row over 50 games it sees variation -> mixed
# strategy from the matrix's point of view -> equilibrium-friendly row.

def _seed_from_state(state: GameState, extra: int = 0) -> int:
    """Derive a deterministic PRNG seed from the game state.

    Hashes the fields that vary across games (pair, game_index, round,
    valuations, quantities, role) PLUS the current M1 anchor (which evolves
    within a game). Including the M1 floor ensures the seed reflects the
    full effective input to the candidate-selection step.

    `extra` lets a caller request a different seed for different sub-decisions
    within the same call (e.g. multiple candidate allocations).
    """
    import hashlib
    payload = (
        f"{state.pair_key or '_'}|"
        f"{state.game_index if state.game_index is not None else -1}|"
        f"{state.round}|{state.max_rounds}|"
        f"{state.role}|"
        f"{tuple(state.valuations_self)}|"
        f"{tuple(state.quantities)}|"
        f"{int(state.batna_self)}|"
        f"{int(state.previous_self_offer_value) if state.previous_self_offer_value is not None else -1}|"
        f"{extra}"
    )
    digest = hashlib.sha256(payload.encode()).digest()
    # Take first 8 bytes -> 64-bit unsigned int -> seed.
    return int.from_bytes(digest[:8], "big")


def _greedy_split_seeded(
    state: GameState,
    target_value: float,
    rng: "random.Random",
) -> tuple[list[int], list[int]]:
    """Variant of `_greedy_split` that quasi-randomises tie-breaks.

    Behaviour differs from `_greedy_split` ONLY when multiple item types have
    the same self-priority value. In that case, `rng` decides the order.
    Strict priorities are still respected.
    """
    T = state.num_items_types
    alloc_self = [0] * T
    alloc_other = list(state.quantities)

    # Compute priority groups. Items with the same priority are tied.
    priorities = [
        state.valuations_self[i] - EXPECTED_OPPONENT_VAL_PER_UNIT
        for i in range(T)
    ]
    # Sort by priority desc; within each tier, shuffle deterministically.
    indexed = sorted(range(T), key=lambda i: -priorities[i])
    # Bucketise: group consecutive entries with equal priority.
    order: list[int] = []
    j = 0
    while j < len(indexed):
        k = j
        while k < len(indexed) and priorities[indexed[k]] == priorities[indexed[j]]:
            k += 1
        tier = indexed[j:k]
        if len(tier) > 1:
            rng.shuffle(tier)
        order.extend(tier)
        j = k

    for i in order:
        while alloc_other[i] > 0:
            current = offer_value(alloc_self, state.valuations_self)
            if current >= target_value:
                break
            alloc_self[i] += 1
            alloc_other[i] -= 1
        if offer_value(alloc_self, state.valuations_self) >= target_value:
            break

    return alloc_self, alloc_other


def propose(state: GameState) -> tuple[list[int], list[int], str]:
    """
    Return (allocation_self, allocation_other, reason).

    Guarantees:
      * value(alloc_self) >= max(BATNA + 1, previous_self_offer_value)  -> M1, M2
      * not degenerate when total_items > 1                             -> M3
      * fully reproducible given identical inputs                       -> determinism
      * varies across games (different game_index / valuations)         -> mixed-strategy
    """
    import random

    base_target = aspiration_target(state, max_attainable_value(state))
    if state.previous_self_offer_value is not None:
        base_target = max(base_target, state.previous_self_offer_value)

    floor = state.batna_self + 1.0
    if state.previous_self_offer_value is not None:
        floor = max(floor, state.previous_self_offer_value)
    K_CANDIDATES = 6  # how many candidates to generate per call
    # Use a sentinel extra value so the selector seed never collides with any
    # candidate-generation seed. Without this, candidate k=0 uses extra=0 and
    # so does the selector — which artificially correlates the choice index
    # with the order in which candidates were produced, killing variance.
    selector_rng = random.Random(_seed_from_state(state, extra=-1))

    # Generate K candidate allocations using different sub-seeds.
    # Each sub-seed permutes tied priorities differently AND triggers a
    # different post-hoc neutral swap, producing pattern-level variation.
    candidates: list[tuple[list[int], list[int]]] = []
    seen_keys: set[tuple[int, ...]] = set()
    for k in range(K_CANDIDATES):
        sub_rng = random.Random(_seed_from_state(state, extra=k))
        cand_self, cand_other = _greedy_split_seeded(state, base_target, sub_rng)
        cand_self, cand_other = _post_process_candidate(
            state, cand_self, cand_other, floor, sub_rng
        )
        # Validate.
        self_v = offer_value(cand_self, state.valuations_self)
        if self_v < floor - 1e-9:
            continue
        if is_degenerate(cand_self, state.quantities) and state.total_items > 1:
            continue
        key = tuple(cand_self)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append((cand_self, cand_other))

    if not candidates:
        # Last resort: deterministic baseline.
        alloc_self, alloc_other = _greedy_split(state, base_target)
        # Run through standard fixups.
        alloc_self, alloc_other = _post_process_candidate(
            state, alloc_self, alloc_other, floor, selector_rng
        )
    else:
        # Hash-pick: same state -> same candidate (reproducibility).
        alloc_self, alloc_other = selector_rng.choice(candidates)

    # Final M2 safety guard.
    while (
        offer_value(alloc_self, state.valuations_self) < floor
        and sum(alloc_other) > 1
    ):
        moved = False
        order = sorted(
            range(state.num_items_types),
            key=lambda i: state.valuations_self[i],
            reverse=True,
        )
        for i in order:
            if alloc_other[i] > 1:
                alloc_self[i] += 1
                alloc_other[i] -= 1
                moved = True
                break
        if not moved:
            break

    self_v = offer_value(alloc_self, state.valuations_self)
    other_v_est = estimated_opponent_value(alloc_other)
    reason = (
        f"r{state.round}/{state.max_rounds}: target={base_target:.0f} "
        f"self={self_v:.0f} (BATNA={state.batna_self:.0f}) "
        f"opp_est={other_v_est:.0f} cands={len(candidates)}"
    )
    return alloc_self, alloc_other, reason


def _post_process_candidate(
    state: GameState,
    alloc_self: list[int],
    alloc_other: list[int],
    floor: float,
    rng: "random.Random",
) -> tuple[list[int], list[int]]:
    """Anti-degenerate fixups + optional neutral swap to diversify pattern.

    The neutral swap exchanges one unit between self and other if it leaves
    both M2 and the degenerate-ness check intact. This generates pattern
    variation that goes beyond simple tier-shuffle, without changing
    self-value materially.
    """
    # Anti-M3 fixups.
    if sum(alloc_other) == 0 and state.total_items > 1:
        give_back_idx = min(
            (i for i in range(state.num_items_types) if alloc_self[i] > 0),
            key=lambda i: state.valuations_self[i],
        )
        alloc_self[give_back_idx] -= 1
        alloc_other[give_back_idx] += 1
    if sum(alloc_self) == 0 and state.total_items > 1:
        take_idx = max(
            range(state.num_items_types),
            key=lambda i: state.valuations_self[i],
        )
        alloc_self[take_idx] += 1
        alloc_other[take_idx] -= 1

    # Optional neutral swap: try one rng-chosen exchange that preserves M2.
    # This is what most reliably creates *pattern* variance across candidates.
    T = state.num_items_types
    swappable = [
        (i, j)
        for i in range(T) for j in range(T)
        if i != j and alloc_self[i] > 0 and alloc_other[j] > 0
    ]
    if swappable:
        i, j = rng.choice(swappable)
        # Try swap: self gives 1 of i to other, takes 1 of j from other.
        delta_v = state.valuations_self[j] - state.valuations_self[i]
        new_self_v = offer_value(alloc_self, state.valuations_self) + delta_v
        if new_self_v >= floor - 1e-9:
            new_a_self = list(alloc_self)
            new_a_other = list(alloc_other)
            new_a_self[i] -= 1
            new_a_other[i] += 1
            new_a_self[j] += 1
            new_a_other[j] -= 1
            # Re-check anti-degenerate.
            if (
                (sum(new_a_self) > 0 or state.total_items <= 1)
                and (sum(new_a_other) > 0 or state.total_items <= 1)
            ):
                alloc_self, alloc_other = new_a_self, new_a_other

    return alloc_self, alloc_other


# ---------------------------------------------------------------------------
# Accept / reject decision
# ---------------------------------------------------------------------------

def expected_continuation_value(state: GameState) -> float:
    """
    Optimistic estimate of value if we reject and continue bargaining.
    Bounded below by BATNA.
    """
    if state.round >= state.max_rounds:
        return state.batna_self
    future = GameState(
        role=state.role,
        round=state.round + 1,
        max_rounds=state.max_rounds,
        discount=state.discount,
        valuations_self=state.valuations_self,
        batna_self=state.batna_self,
        quantities=state.quantities,
    )
    future_target = aspiration_target(future, max_attainable_value(state))
    return max(state.batna_self, state.discount * future_target)


def decide_offer(
    state: GameState,
    offer_value_override: float | None = None,
) -> tuple[bool, str]:
    """
    Return (accept, reason). Hard rules: never accept below BATNA (M4); never
    reject above-BATNA in the final round (M5, enforced by sanitiser below).

    The green's RemoteNegotiator passes the offer's value in the observation
    as `offer_value`; we accept it via `offer_value_override` to avoid
    recomputation (and to use the green's discounted value if it differs).
    """
    if offer_value_override is not None:
        self_v = float(offer_value_override)
    elif state.current_offer_to_self is not None:
        self_v = offer_value(state.current_offer_to_self, state.valuations_self)
    else:
        return False, "no offer provided"

    cont_v = expected_continuation_value(state)

    if self_v < state.batna_self:
        return False, f"reject: {self_v:.0f} < BATNA {state.batna_self:.0f}"

    if self_v >= cont_v:
        return True, (
            f"accept: {self_v:.0f} >= cont {cont_v:.0f} "
            f"(BATNA {state.batna_self:.0f})"
        )
    return False, (
        f"reject: cont {cont_v:.0f} > offer {self_v:.0f} (>= BATNA)"
    )


# ---------------------------------------------------------------------------
# Sanitisers (defence in depth for LLM/RL outputs)
# ---------------------------------------------------------------------------

def sanitise_proposal(
    state: GameState,
    alloc_self: Sequence[int],
    alloc_other: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Clamp a candidate proposal to satisfy M1, M2, M3 and conservation."""
    T = state.num_items_types
    a_self = [max(0, int(x)) for x in list(alloc_self)[:T]]
    a_other = [max(0, int(x)) for x in list(alloc_other)[:T]]

    while len(a_self) < T:
        a_self.append(0)
    while len(a_other) < T:
        a_other.append(0)

    # Restore conservation (trust a_self, derive a_other).
    for i in range(T):
        if a_self[i] + a_other[i] != state.quantities[i]:
            a_self[i] = min(state.quantities[i], max(0, a_self[i]))
            a_other[i] = state.quantities[i] - a_self[i]

    self_v = offer_value(a_self, state.valuations_self)
    floor = state.batna_self + 1.0
    if state.previous_self_offer_value is not None:
        floor = max(floor, state.previous_self_offer_value)

    if self_v < floor or is_degenerate(a_self, state.quantities):
        # Candidate violates invariants -> regenerate from baseline.
        a_self, a_other, _ = propose(state)

    return a_self, a_other


def sanitise_decision(
    state: GameState,
    accept_candidate: bool,
    offer_value_override: float | None = None,
) -> bool:
    """Override accept/reject if it would violate M4 or M5."""
    if offer_value_override is not None:
        self_v = float(offer_value_override)
    elif state.current_offer_to_self is not None:
        self_v = offer_value(state.current_offer_to_self, state.valuations_self)
    else:
        return False
    # M4: never accept below BATNA.
    if accept_candidate and self_v < state.batna_self:
        return False
    # M5: in the final round, never walk away from above-BATNA.
    if (
        not accept_candidate
        and state.round >= state.max_rounds
        and self_v > state.batna_self
    ):
        return True
    return accept_candidate