"""
log.py — how much the programs say while they run.

    say(...)     the headline, always printed: one line per stage.
    detail(...)  the working — per-device counts, paths, full reports.
                 Printed only when VERBOSE is on.

Nothing is lost by staying quiet: every report printed at detail level is also
written to a file (dataset_summary.txt, results.txt, comparison.txt).  Turn the
working on with DQD_VERBOSE=1, or log.VERBOSE = True before the program starts.
"""
import os
import sys

VERBOSE = os.environ.get("DQD_VERBOSE", "").lower() in ("1", "true", "yes")


def say(*args, **kwargs):
    """A headline: always printed."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def detail(*args, **kwargs):
    """The working: printed only when VERBOSE."""
    if VERBOSE:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def warn(*args, **kwargs):
    """Something the reader must see even when quiet."""
    kwargs.setdefault("flush", True)
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)
