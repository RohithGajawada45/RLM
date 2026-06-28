"""
cache_persist.py — Shared persistence helpers for all in-memory caches.

Handles:
  • Periodic auto-save (background thread, every AUTOSAVE_INTERVAL seconds)
  • On-demand save (called from shutdown hook)
  • Load on startup
  • atexit registration so even SIGTERM / Ctrl+C saves what it can

Usage (in each module that owns a cache):
    from cache_persist import CachePersister

    _persister = CachePersister(
        path="cache/my_cache.pkl",
        get_cache=lambda: _MY_CACHE,       # callable → current OrderedDict
        set_cache=lambda d: _set(_MY_CACHE, d),  # callable → replaces contents
        label="my-cache",
    )
    _persister.start()      # begin background autosave + register atexit
    _persister.load()       # call once at import / startup
"""

import atexit
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional


AUTOSAVE_INTERVAL = int(60)   # seconds between background saves


class CachePersister:
    """
    Wraps an in-memory OrderedDict cache with load/save/autosave logic.

    Parameters
    ----------
    path        : File to persist to/from (pickle).
    get_cache   : Zero-arg callable that returns the live cache dict.
    set_cache   : One-arg callable that *replaces* the live cache contents
                  with the supplied OrderedDict (used during load).
    label       : Short name used in log lines.
    interval    : Autosave period in seconds (default: AUTOSAVE_INTERVAL).
    """

    def __init__(
        self,
        path: str,
        get_cache: Callable[[], OrderedDict],
        set_cache: Callable[[OrderedDict], None],
        label: str,
        interval: int = AUTOSAVE_INTERVAL,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._get = get_cache
        self._set = set_cache
        self.label = label
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────

    def load(self) -> int:
        """Load cache from disk into memory. Returns number of entries loaded."""
        if not self.path.exists():
            print(f"[{self.label}] No persisted cache at {self.path} — starting fresh.")
            return 0

        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, OrderedDict):
                print(f"[{self.label}] Cache file has unexpected type {type(data)} — ignored.")
                return 0

            self._set(data)
            n = len(data)
            print(f"[{self.label}] Loaded {n} entries from {self.path}.")
            return n

        except Exception as exc:
            print(f"[{self.label}] Failed to load cache: {exc}")
            return 0

    def save(self) -> int:
        """Snapshot the live cache to disk. Returns number of entries saved."""
        try:
            snapshot = OrderedDict(self._get())          # shallow copy under caller's lock
            with open(self.path, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            n = len(snapshot)
            print(f"[{self.label}] Saved {n} entries to {self.path}.")
            return n
        except Exception as exc:
            print(f"[{self.label}] Failed to save cache: {exc}")
            return 0

    def start(self) -> None:
        """Start the background autosave thread and register atexit handler."""
        atexit.register(self._atexit_save)

        self._thread = threading.Thread(
            target=self._autosave_loop,
            name=f"autosave-{self.label}",
            daemon=True,          # won't block process exit
        )
        self._thread.start()
        print(
            f"[{self.label}] Autosave thread started "
            f"(interval={self.interval}s, file={self.path})."
        )

    def stop(self) -> None:
        """Signal the autosave thread to exit and do a final save."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.save()

    # ── Internals ─────────────────────────────────────────────────────────

    def _autosave_loop(self) -> None:
        while not self._stop.wait(timeout=self.interval):
            self.save()

    def _atexit_save(self) -> None:
        """Called by Python's atexit machinery on any clean-ish exit
        (SIGTERM, Ctrl+C SIGINT, normal sys.exit). SIGKILL (kill -9)
        cannot be caught — use the periodic autosave as the safety net."""
        print(f"[{self.label}] atexit: saving cache before exit…")
        self.save()