"""
Optional RL proposer layer.

When USE_RL=true and an NFSP or RNAD checkpoint is present at the expected
path (or RL_CHECKPOINT_PATH), this module loads the checkpoint via OpenSpiel
and produces a candidate allocation. The negotiator compares it against the
heuristic baseline and picks whichever has higher self-value (both sanitised).

When USE_RL=false (default) or the checkpoint is missing or the dependencies
(open_spiel, torch) are not installed, this module degrades silently to
returning None — the negotiator falls back to the heuristic baseline.

This module is INTENTIONALLY a stub-with-an-integration-point. Wiring up the
real OpenSpiel state -> allocation conversion is non-trivial because OpenSpiel
bargaining uses an integer action space over a "keep vector" encoding. The
green's pyspiel_runner.py uses `state.action_to_string(...)` to decode; we
would need to do the inverse here. That's tractable but requires the
checkpoint and the matching OpenSpiel game spec at hand.

To enable for real:
  1. Set USE_RL=true in the environment.
  2. Provide RL_CHECKPOINT_PATH=/path/to/nfsp_bg5.pt (or rnad_bg5.pkl).
  3. Set RL_ALGO=nfsp or RL_ALGO=rnad accordingly.
  4. Add `open_spiel` and `torch` to requirements.txt (CPU build is fine).
  5. Add the checkpoint file to the Docker image (large; consider a separate
     image tag like `:rl` so the default image stays small).
  6. Implement _propose_from_checkpoint() body — see TODO comment below.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from strategy import GameState

logger = logging.getLogger("purple_mae.rl")

USE_RL = os.environ.get("USE_RL", "false").lower() in ("1", "true", "yes")
ALGO = os.environ.get("RL_ALGO", "nfsp").lower()
CHECKPOINT_PATH = os.environ.get("RL_CHECKPOINT_PATH", "")


_checkpoint_loaded = False
_checkpoint_obj = None  # placeholder; populated by _try_load()


def is_enabled() -> bool:
    """Cheap check: should the negotiator even call us?"""
    return USE_RL


def _try_load() -> bool:
    """Lazily load the checkpoint. Returns True if usable."""
    global _checkpoint_loaded, _checkpoint_obj
    if _checkpoint_loaded:
        return _checkpoint_obj is not None
    _checkpoint_loaded = True

    if not USE_RL:
        return False
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        logger.info(
            "RL enabled but no checkpoint at %r; RL layer is no-op.",
            CHECKPOINT_PATH,
        )
        return False

    try:
        # TODO(rl-integration): load the checkpoint. Sketch:
        #
        #   if ALGO == "nfsp":
        #       import torch
        #       _checkpoint_obj = torch.load(CHECKPOINT_PATH, map_location="cpu")
        #   elif ALGO == "rnad":
        #       import pickle
        #       with open(CHECKPOINT_PATH, "rb") as f:
        #           _checkpoint_obj = pickle.load(f)
        #
        # Then verify the OpenSpiel game spec matches:
        #   import pyspiel
        #   game = pyspiel.load_game("bargaining", { ... })
        # and stash a reference for use in _propose_from_checkpoint().
        logger.info("RL stub: would load %s checkpoint from %s", ALGO, CHECKPOINT_PATH)
        _checkpoint_obj = None  # stub: never actually loaded
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("RL checkpoint load failed: %s; degrading to heuristic.", exc)
        _checkpoint_obj = None
        return False


def maybe_propose(state: GameState) -> Optional[tuple[list[int], list[int]]]:
    """
    Return a candidate (alloc_self, alloc_other) from the RL policy, or None
    if RL is unavailable. The caller MUST still pass our return through
    strategy.sanitise_proposal — the RL policy is not bound by M1-M5.
    """
    if not _try_load():
        return None
    # TODO(rl-integration): real proposer:
    #
    #   1. Build OpenSpiel state from (state.valuations_self, state.batna_self,
    #      state.quantities, state.discount, state.max_rounds) — note that the
    #      opponent's private info is unknown, so we either marginalise or
    #      use a self-play surrogate.
    #   2. Run a forward pass through _checkpoint_obj to get action logits.
    #   3. Mask to legal non-terminal actions (skip ACCEPT/WALK; we're proposing).
    #   4. Pick argmax (or sample).
    #   5. Decode the integer action via state.action_to_string(...) into a
    #      "keep vector" allocation_self.
    #   6. Derive allocation_other as quantities - allocation_self.
    #   7. Return (allocation_self, allocation_other).
    logger.debug("RL proposer is a stub; returning None.")
    return None
