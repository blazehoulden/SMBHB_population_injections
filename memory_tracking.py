#!/usr/bin/env python3
"""
memory_tracking.py — opt-in RSS memory tracker shared by stage1_setup.py
and stage2_inject.py.

Purpose
───────
Both stage scripts run as Slurm array tasks / subprocesses, and the usual
question after a run ("how much --mem should I actually request?") is hard
to answer from stdout logs alone. This module adds a `MemoryTracker` that,
when explicitly enabled via each script's --track-memory flag, records RSS
at named checkpoints throughout a run, optionally samples in a background
thread to catch short-lived peaks a checkpoint-only trace would miss, and
writes a small JSON report at the end.

Default is OFF everywhere. When disabled, every method is a cheap no-op
(a single `if not self.enabled: return`) — zero measurable overhead, zero
behavior change, unless a script explicitly turns it on.

Two backends
────────────
- psutil (preferred, if installed): per-call current RSS, and a background
  thread can sample it periodically to catch peaks between checkpoints.
- stdlib fallback (resource.getrusage): no extra dependency, but only
  gives peak-RSS-so-far (ru_maxrss), not an instantaneous "right now"
  value — so per-checkpoint deltas aren't meaningful with this backend,
  though the overall peak still is. Background sampling is skipped in
  this mode since ru_maxrss already tracks the running peak on its own.
"""

import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional


class MemoryTracker:
    """
    Opt-in (default OFF) RSS memory tracker for HPC job-sizing.

    Usage:
        mem = MemoryTracker(enabled=args.track_memory, label='stage2 baseline')
        mem.start_background_sampling()
        ...
        mem.checkpoint('after loading pulsars')
        ...
        mem.checkpoint('after CGW SNR computation')
        mem.stop_background_sampling()
        mem.write_report('metadata/memory_profile/stage2_baseline.json')
    """

    def __init__(self, enabled: bool = False, label: str = '',
                 sample_interval: float = 5.0):
        self.enabled = enabled
        self.label = label
        self.sample_interval = max(0.5, sample_interval)
        self.records: List[dict] = []
        self._peak_rss_mb = 0.0
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._use_psutil = False
        self._proc = None
        self._t_start = time.time()

        if self.enabled:
            try:
                import psutil
                self._proc = psutil.Process(os.getpid())
                self._use_psutil = True
            except Exception:
                self._use_psutil = False

    # ── internals ──────────────────────────────────────────────────────────

    def _current_rss_mb(self) -> Optional[float]:
        if self._use_psutil:
            try:
                return self._proc.memory_info().rss / 1024**2
            except Exception:
                return None
        try:
            import resource
            ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is KB on Linux, bytes on macOS.
            return ru_maxrss / 1024**2 if sys.platform == 'darwin' else ru_maxrss / 1024.0
        except Exception:
            return None

    # ── public API ─────────────────────────────────────────────────────────

    def checkpoint(self, tag: str) -> None:
        """Record current RSS under a named checkpoint. No-op if disabled."""
        if not self.enabled:
            return
        rss_mb = self._current_rss_mb()
        if rss_mb is None:
            return
        self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
        self.records.append({
            't_sec':  round(time.time() - self._t_start, 1),
            'tag':    tag,
            'rss_mb': round(rss_mb, 1),
        })
        prefix = f' {self.label}' if self.label else ''
        print(f'  [mem{prefix}] {tag}: {rss_mb:,.0f} MB RSS')

    def start_background_sampling(self) -> None:
        """
        Start a daemon thread that samples RSS every `sample_interval`
        seconds, so peaks that occur *between* checkpoints (e.g. inside a
        single long NUFFT or PTA-build call) still show up in the report.

        Only meaningful with the psutil backend — the stdlib fallback
        (ru_maxrss) already tracks the running peak on its own without
        needing a sampling thread, so this is a no-op in that case.
        """
        if not self.enabled or not self._use_psutil or self._thread is not None:
            return
        self._stop_event = threading.Event()

        def _sample_loop():
            while not self._stop_event.is_set():
                rss_mb = self._current_rss_mb()
                if rss_mb is not None:
                    self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
                self._stop_event.wait(self.sample_interval)

        self._thread = threading.Thread(target=_sample_loop, daemon=True)
        self._thread.start()

    def stop_background_sampling(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval + 1)
        self._stop_event = None
        self._thread = None

    def write_report(self, path: str, extra: Optional[dict] = None) -> None:
        """Dump checkpoints + peak RSS to a JSON file. No-op if disabled."""
        if not self.enabled:
            return
        # Always fold in one last reading of the stdlib peak-so-far, even
        # on the psutil backend, as a cheap sanity cross-check.
        try:
            import resource
            ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            stdlib_peak_mb = (ru_maxrss / 1024**2 if sys.platform == 'darwin'
                               else ru_maxrss / 1024.0)
        except Exception:
            stdlib_peak_mb = None

        report = {
            'label':                 self.label,
            'backend':               'psutil' if self._use_psutil else 'resource.getrusage',
            'peak_rss_mb':           round(self._peak_rss_mb, 1),
            'stdlib_peak_rss_mb':    (round(stdlib_peak_mb, 1)
                                       if stdlib_peak_mb is not None else None),
            'wall_time_sec':         round(time.time() - self._t_start, 1),
            'checkpoints':           self.records,
        }
        if extra:
            report.update(extra)

        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(report, fh, indent=2)

        prefix = f' {self.label}' if self.label else ''
        print(f'  📊 [mem{prefix}] peak={self._peak_rss_mb:,.0f} MB → wrote {path}')