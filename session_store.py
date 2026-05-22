"""
Per-game memory for the Purple MAE agent.

Each call to our agent contains the green's full observation, which includes
`pair` (e.g. "challenger__vs__soft") and `game_index` (0..games-1). We key
the session store on (pair, game_index) so:

  * Cross-round M1 anchor works reliably within one game.
  * State resets automatically between games (different game_index).
  * No dependence on A2A context_id semantics (which can be flaky).

The store is an in-memory dict with TTL eviction; one bargaining session is
short-lived (seconds to a minute), and the green serialises calls per game.
"""

from __future__ import annotations

import threading
import time

_TTL_SECONDS = 60 * 30


class SessionStore:
    """Thread-safe TTL dict for per-(pair, game_index) state."""

    def __init__(self, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        # key -> (value, inserted_at)
        self._data: dict[str, tuple[float, float]] = {}

    @staticmethod
    def make_key(pair: str | None, game_index: int | None) -> str | None:
        if pair is None or game_index is None:
            return None
        return f"{pair}#{int(game_index)}"

    def get_last_self_value(self, key: str | None) -> float | None:
        if not key:
            return None
        with self._lock:
            self._evict_expired_locked()
            entry = self._data.get(key)
            return entry[0] if entry else None

    def set_last_self_value(self, key: str | None, value: float) -> None:
        if not key:
            return
        with self._lock:
            self._data[key] = (value, time.time())
            self._evict_expired_locked()

    def _evict_expired_locked(self) -> None:
        cutoff = time.time() - self._ttl
        expired = [k for k, (_, t) in self._data.items() if t < cutoff]
        for k in expired:
            self._data.pop(k, None)


session_store = SessionStore()
