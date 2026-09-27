"""
capacitance_config.py — the capacitance parameter space every device is drawn
from.

An interval is either [lo, hi] or a union [[lo1, hi1], [lo2, hi2], ...].  The
union form supports a provably non-overlapping train/test split (train from
the odd bands, test from the even), which the study no longer uses: the split
is made on device IDs (study/device_split.py).

WHY THESE RANGES.  The honeycomb only appears when each dot is driven mainly
by its own plunger gate —

    dot-1 lines   slope dV2/dV1 = -d1g1 / d1g2   (steep)
    dot-2 lines   slope dV2/dV1 = -d2g1 / d2g2   (shallow)

— so the two families stay distinct while max(cross) < min(primary), i.e.
0.60 < 0.80.  That holds for every band here, but nothing enforces it.

WHY THEY ARE WIDE.  Narrower ranges gave a training set whose members all
looked alike.  These widen the three quantities that change what a diagram
looks like:

    d1g1, d2g2   0.8 - 6.0   honeycomb period      ~1.6 to ~12 cells/window
    d1g2, d2g1   0.05 - 0.6  line slope            ~0.5 to ~37 deg from axis
    d1d2         0.2 - 2.0   interdot anticrossing

No draw is checked, filtered or redrawn: the pool is a plain uniform sample of
this box, unusual draws included.
"""
from typing import Dict, Iterable, List, Sequence, Tuple, Union


# One band, [lo, hi], or a union of them.
IntervalSpec = Union[Sequence[float], Sequence[Sequence[float]]]


# ── The parameter space ───────────────────────────────────────────────────
#
# Every entry is drawn independently and uniformly per device, so a device is
# a point in this 14-dimensional box, not a perturbation of a template.

DEFAULT_INTERVALS: Dict[str, Dict[str, IntervalSpec]] = {
    "Cdd": {
        # Dot self-capacitances.  Set the charging energy; they move the
        # honeycomb only weakly, which is why they are not split (below).
        "d1d1": [0.30, 1.40],
        "d2d2": [0.30, 1.40],
        # Interdot (mutual) capacitance — THE double-dot parameter.  Small:
        # the two dots are nearly independent and the line families simply
        # cross.  Large: the crossings open into long interdot segments and
        # the triple points separate.  Wide on purpose, and not trimmed:
        # draws where the dots merge into one effective dot are kept too.
        "d1d2": [0.20, 2.00],
    },
    "Cgd": {
        # Primary gate capacitances — one per dot, drawn INDEPENDENTLY so
        # asymmetric devices (a fine honeycomb in one direction, a coarse one
        # in the other) occur as often as symmetric ones.
        "d1g1": [0.80, 6.00],
        "d2g2": [0.80, 6.00],
        # Cross gate capacitances — the slope of each line family.  Bounded
        # by 0.60 < 0.80 = min(primary), which is condition (*) above.
        "d1g2": [0.05, 0.60],
        "d2g1": [0.05, 0.60],
        # Third gate: a weak residual coupling, same range for both dots.
        "d1g3": [0.01, 0.10],
        "d2g3": [0.01, 0.10],
    },
    "Cds": {
        # Dot-to-sensor: how strongly each dot's charge shifts the sensor.
        "s1d1": [0.02, 0.15],
        "s1d2": [0.02, 0.15],
    },
    "Cgs": {
        # Gate-to-sensor cross-talk.  s1g2 used to be the single point
        # [0.05, 0.05] — a constant, and therefore one more reason every
        # sensor image looked like every other one.
        "s1g1": [0.02, 0.10],
        "s1g2": [0.02, 0.10],
        "s1g3": [0.10, 1.00],
    },
}

DEFAULT_LABELS_Cdd = [
    ["d1d1", "d1d2"],
    ["d1d2", "d2d2"],
]

DEFAULT_LABELS_Cgd = [
    ["d1g1", "d1g2", "d1g3"],
    ["d2g1", "d2g2", "d2g3"],
]

DEFAULT_LABELS_Cds = [["s1d1", "s1d2"]]

DEFAULT_LABELS_Cgs = [["s1g1", "s1g2", "s1g3"]]

# The entries that decide what the honeycomb LOOKS like.  These are the ones
# a train/test split has to separate; everything else (sensor coupling, self
# capacitance) can be shared without weakening the claim, because it does not
# move a single transition line.
GEOMETRY_KEYS: Dict[str, List[str]] = {
    "Cdd": ["d1d2"],
    "Cgd": ["d1g1", "d1g2", "d2g1", "d2g2"],
}


# ── Working with band specifications ──────────────────────────────────────

def as_bands(spec: IntervalSpec) -> List[List[float]]:
    """
    Normalise an interval specification to a list of [lo, hi] bands.

        [0.1, 1.0]                 -> [[0.1, 1.0]]
        [[0.1, 0.4], [0.6, 1.0]]   -> [[0.1, 0.4], [0.6, 1.0]]
    """
    if len(spec) and isinstance(spec[0], (list, tuple)):
        return [[float(a), float(b)] for a, b in spec]
    lo, hi = spec
    return [[float(lo), float(hi)]]


def bounds(spec: IntervalSpec) -> Tuple[float, float]:
    """(smallest value, largest value) a specification can produce."""
    bands = as_bands(spec)
    return min(b[0] for b in bands), max(b[1] for b in bands)


def sample(rng, spec: IntervalSpec) -> float:
    """
    Draw one value uniformly from a specification.

    With several bands the draw is uniform over their UNION — a band twice as
    wide is chosen twice as often — so banding a parameter changes which
    values are reachable, never the shape of the distribution over the values
    that are.
    """
    bands = as_bands(spec)
    if len(bands) == 1:
        return rng.uniform(bands[0][0], bands[0][1])
    widths = [b[1] - b[0] for b in bands]
    total = sum(widths)
    if total <= 0:                       # every band degenerate
        return bands[0][0]
    x = rng.uniform(0.0, total)
    for (lo, hi), w in zip(bands, widths):
        if x <= w:
            return lo + x
        x -= w
    return bands[-1][1]


def overlaps(a: IntervalSpec, b: IntervalSpec, atol: float = 0.0) -> bool:
    """True if the two specifications can ever produce the same value."""
    for lo1, hi1 in as_bands(a):
        for lo2, hi2 in as_bands(b):
            if lo1 - atol <= hi2 and lo2 - atol <= hi1:
                return True
    return False


# ── splitting the space into two non-overlapping halves ──────────────────

def split_band(spec: IntervalSpec, gap: float = 0.10
               ) -> Tuple[List[List[float]], List[List[float]]]:
    """Cut one parameter's range into a lower (train) and upper (test) band,
    separated by a dead zone `gap` wide as a fraction of the full range.

        [0.80, 6.00], gap=0.10  ->  train [0.80, 3.14]  test [3.66, 6.00]

    The two bands cannot produce the same value — not "unlikely to", cannot.
    """
    if not 0.0 <= gap < 1.0:
        raise ValueError(f"gap must be in [0, 1), got {gap}")
    lo, hi = bounds(spec)
    width = hi - lo
    edge = 0.5 * (1.0 - gap) * width
    return [[lo, lo + edge]], [[hi - edge, hi]]


def split_by_interval(intervals: Dict[str, Dict[str, IntervalSpec]],
                      gap: float = 0.10
                      ) -> Tuple[Dict[str, Dict[str, IntervalSpec]],
                                 Dict[str, Dict[str, IntervalSpec]]]:
    """(train space, test space) with every parameter cut in two.

    Train draws from the lower band of all 14 capacitances, test from the
    upper, so a test device is made of values the training set never
    contained: a model that scores on it has extrapolated.  Condition (*)
    happens to survive the cut in both directions, since the cross-gate bands
    stay inside [0.05, 0.60] and the primary-gate bands inside [0.80, 6.00].
    Nothing checks it.
    """
    train: Dict[str, Dict[str, IntervalSpec]] = {}
    test: Dict[str, Dict[str, IntervalSpec]] = {}
    for matrix in intervals:
        train[matrix], test[matrix] = {}, {}
        for key, spec in intervals[matrix].items():
            lower, upper = split_band(spec, gap)
            train[matrix][key], test[matrix][key] = lower, upper
    return train, test


def disjoint_report(a: Dict[str, Dict[str, IntervalSpec]],
                    b: Dict[str, Dict[str, IntervalSpec]]) -> Dict:
    """
    Per-parameter proof that two spaces cannot produce the same value.

    ``all_disjoint`` is the single boolean the extrapolation claim rests on;
    ``rows`` carries the two bands and the size of the gap between them, so
    the paper can print the table rather than assert the conclusion.
    """
    rows, all_ok = [], True
    for matrix, key, spec in _flatten(a):
        other = b.get(matrix, {}).get(key)
        if other is None:
            continue
        hit = overlaps(spec, other)
        all_ok = all_ok and not hit
        a_lo, a_hi = bounds(spec)
        b_lo, b_hi = bounds(other)
        rows.append({
            "parameter": f"{matrix}.{key}",
            "train": [a_lo, a_hi], "test": [b_lo, b_hi],
            "gap": float(max(b_lo - a_hi, a_lo - b_hi)),
            "disjoint": not hit,
        })
    return {"all_disjoint": all_ok, "n_parameters": len(rows), "rows": rows}


def format_spec(spec: IntervalSpec) -> str:
    """A band union as one readable string, e.g. '[0.80, 2.10] U [2.70, 4.00]'."""
    return " U ".join(f"[{lo:.3g}, {hi:.3g}]" for lo, hi in as_bands(spec))


def _flatten(intervals: Dict[str, Dict[str, IntervalSpec]]) -> Iterable:
    for matrix in sorted(intervals):
        for key in sorted(intervals[matrix]):
            yield matrix, key, intervals[matrix][key]


def fingerprint(intervals: Dict[str, Dict[str, IntervalSpec]]) -> str:
    """
    Short hash of a whole parameter space.

    It goes into the device-pool folder name, so devices simulated under one
    set of intervals can never be silently reused after the intervals change
    — the single most expensive mistake available in this project.
    """
    import hashlib
    text = ";".join(f"{m}.{k}={as_bands(s)}" for m, k, s in _flatten(intervals))
    return hashlib.sha1(text.encode()).hexdigest()[:8]


class CapacitanceConfig:
    """
    A parameter space: the interval specification of every capacitance entry,
    plus the label layout that says which entry sits where in each matrix.
    """

    def __init__(
        self,
        intervals: Dict[str, Dict[str, IntervalSpec]] = None,
        labels_Cdd: List[List[str]] = None,
        labels_Cgd: List[List[str]] = None,
        labels_Cds: List[List[str]] = None,
        labels_Cgs: List[List[str]] = None,
        name: str = "full",
    ):
        self.intervals = intervals if intervals is not None else DEFAULT_INTERVALS
        self.labels_Cdd = labels_Cdd if labels_Cdd is not None else DEFAULT_LABELS_Cdd
        self.labels_Cgd = labels_Cgd if labels_Cgd is not None else DEFAULT_LABELS_Cgd
        self.labels_Cds = labels_Cds if labels_Cds is not None else DEFAULT_LABELS_Cds
        self.labels_Cgs = labels_Cgs if labels_Cgs is not None else DEFAULT_LABELS_Cgs
        self.name = name

    # ------------------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.intervals)

    def rows(self) -> List[Tuple[str, str, str]]:
        """[(matrix, key, formatted spec)] — for tables and reports."""
        return [(m, k, format_spec(s)) for m, k, s in _flatten(self.intervals)]

