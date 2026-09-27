"""
config.py — StudyConfig: one configuration of the study, and the folder it owns.

A configuration is one measurement budget (rays, points per ray, training-set
size) and gets one folder named after those three numbers, holding config.json,
train/test .npz, the dataset summary, figures/, model/ and evaluation/.

The simulated devices are NOT in that folder.  They do not depend on the ray
budget, so they live in a shared pool under data/_device_pools/ and every
configuration reuses them; the pool name carries a fingerprint of the
capacitance intervals, so changing the parameter space builds a new pool
instead of quietly reusing the old one.  The train/test split lives with the
pool too (study/device_split.py), so a device is on the same side in every
cell of every sweep.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

from ..config import paths
from ..config import log
from ..config.capacitance_config import (CapacitanceConfig, DEFAULT_INTERVALS,
                                         disjoint_report, split_by_interval)
from ..simulation import device_factory
from . import sampling
from .device_figures import DEFAULT_DEVICE_FIGURES, normalise_devices

DATASET_FILES = {"train": "train.npz", "test": "test.npz"}
CONFIG_JSON = "config.json"
SUMMARY_JSON = "dataset_summary.json"
SUMMARY_TXT = "dataset_summary.txt"
FIGURES_DIR = "figures"
MODEL_DIR = "model"
EVAL_DIR = "evaluation"
POOLS_DIRNAME = "_device_pools"

SPLIT_MODES = ("device", "interval")


@dataclass
class StudyConfig:
    """Every knob of one configuration.

    run_1 writes these to config.json; run_2 and run_3 read that file back, so
    they cannot silently disagree with the dataset they are pointed at.
    """

    n_rays: int = 3
    n_points: int = 50

    # Where those n_rays x n_points points are put.  "rays" is the real
    # experiment; "grid" and "random" spend the same budget on scattered
    # points (study/sampling.py).  The geometry ablation that used them was
    # deleted on 2026-09-20, but a folder on disk may still carry one.
    sampling: str = sampling.DEFAULT

    n_train: int = 100
    n_test: int = 30

    # "device"   one pool split by device ID: train and test share every
    #            interval, so the held-out score is interpolation across
    #            devices from one population.
    # "interval" two pools: every capacitance range cut in half with a dead
    #            zone between, train from the lower bands and test from the
    #            upper.  Extrapolation, and a harder question — expect lower
    #            scores than the device split gives.
    split_mode: str = "device"
    interval_gap: float = 0.10      # dead zone, as a fraction of each range

    resolution: int = 100
    # The split is made on device IDs once and stored with the pool; changing
    # this seed reassigns the same devices, it does not redraw them.
    split_seed: int = 12345
    voltage_window: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    offset_scale: float = 0.35      # random per-device window offset
    coulomb_peak_width: float = 0.01
    temperature: float = 0.00001
    seed: int = 0                   # same seed = the same devices, every time

    epochs: int = 40

    # Initial weights and batch order only: nothing about the data depends on
    # it, so two configurations differing only here see the same measurements
    # and differ only in where the optimisation landed.  One run per arm is
    # therefore one roll of the dice — re-run at several train_seeds and
    # compare the clusters, not the single numbers.  model/ and evaluation/
    # carry the seed in their names so the runs do not overwrite each other.
    train_seed: int = 0

    # Figures change nothing about the data or the results, so they are safe
    # to turn on and off at any time.
    save_device_figures: bool = True
    device_figures: Dict[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_DEVICE_FIGURES))
    # Per split: "ALL", "NONE", or [1, 3, 7] (1-based, matching sample_<i>).
    # "ALL" on a large pool writes a lot of files; the count is printed first.
    figure_devices: Dict[str, object] = field(
        default_factory=lambda: {"train": [1, 2, 3], "test": [1, 2, 3]})

    figure_dpi: int = 200           # 300 for the paper, 150 for a quick look
    figure_size_in: float = 8.0
    figure_cell_grid: bool = True   # voltage-grid cells behind the binary maps

    def __post_init__(self):
        # Set by load(): the folder this configuration was read from.
        self._dir = None
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(
                f"split_mode={self.split_mode!r} is not a mode; available: "
                + ", ".join(SPLIT_MODES))
        if self.sampling not in sampling.STRATEGIES:
            raise ValueError(
                f"sampling={self.sampling!r} is not a strategy; available: "
                + ", ".join(sampling.STRATEGIES))
        self.voltage_window = tuple(float(v) for v in self.voltage_window)
        # Any figure kind left unmentioned is off, so a settings block that
        # lists only the pictures it wants behaves the way it reads.
        self.device_figures = {k: bool(self.device_figures.get(k, False))
                               for k in DEFAULT_DEVICE_FIGURES}
        # Copied, not shared: dataclasses.replace() hands the same dict to
        # every configuration built from one template, and mutating it here
        # would edit them all.  Checked now so a typo like "AL" fails on the
        # first line of run_1, not after twenty minutes of simulation.
        devices = dict(self.figure_devices)
        for split in ("train", "test"):
            devices.setdefault(split, "ALL")
            normalise_devices(devices[split])
        self.figure_devices = devices

    @property
    def name(self) -> str:
        """The folder name, e.g. '3_rays_50_points_100_samples'.

        A non-default sampling strategy or the interval split adds a suffix,
        so those are separate folders rather than overwriting the budget
        study's.
        """
        base = (f"{self.n_rays}_rays_{self.n_points}_points_"
                f"{self.n_train}_samples")
        if self.sampling != sampling.DEFAULT:
            base = f"{base}_{self.sampling}"
        if self.split_mode == "interval":
            base = f"{base}_disjoint"
        return base

    @property
    def dir(self) -> str:
        # CONFIG_ROOT, not data/, so run_0 can put a whole sweep inside one
        # folder under results/.  It still defaults to data/.
        return self._dir or paths.config_root(self.name)

    def path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    @property
    def train_npz(self) -> str:
        return self.path(DATASET_FILES["train"])

    @property
    def test_npz(self) -> str:
        return self.path(DATASET_FILES["test"])

    @property
    def figures_dir(self) -> str:
        return self.path(FIGURES_DIR)

    @property
    def _seed_suffix(self) -> str:
        # Empty at train_seed 0 on purpose: every model already on disk was
        # trained at that seed and keeps the folder it is in.
        return "" if self.train_seed == 0 else f"_seed{self.train_seed}"

    @property
    def model_dir(self) -> str:
        return self.path(MODEL_DIR + self._seed_suffix)

    @property
    def eval_dir(self) -> str:
        return self.path(EVAL_DIR + self._seed_suffix)

    @property
    def checkpoint(self) -> str:
        return os.path.join(self.model_dir, "unet.pt")

    @property
    def n_devices(self) -> int:
        """Devices in the pool: train and test together, from one draw."""
        return self.n_train + self.n_test

    def capacitance_config(self, side: str = None) -> CapacitanceConfig:
        """The distribution a pool is drawn from.

        In "device" mode there is one space and `side` is ignored.  In
        "interval" mode `side` picks the lower ("train") or upper ("test")
        half of every range; side=None is an error rather than a silent
        fallback, since there is no "the" distribution once the space is cut.
        """
        if self.split_mode != "interval":
            return CapacitanceConfig()
        if side not in ("train", "test"):
            raise ValueError(
                "split_mode='interval' has two capacitance spaces; ask for "
                f"side='train' or side='test', not {side!r}")
        train, test = split_by_interval(DEFAULT_INTERVALS, self.interval_gap)
        space = train if side == "train" else test
        return CapacitanceConfig(space, name=f"{side}_half")

    def interval_report(self) -> Dict:
        """Per-parameter proof the two halves cannot produce the same value."""
        if self.split_mode != "interval":
            return {"applicable": False}
        train, test = split_by_interval(DEFAULT_INTERVALS, self.interval_gap)
        out = disjoint_report(train, test)
        out["applicable"] = True
        out["gap_fraction"] = self.interval_gap
        return out

    def pool_dir(self, side: str = None) -> str:
        """Where this configuration's devices live.

        A pure function of the settings, shared by every configuration that
        asks for the same devices — which is why changing only n_rays or
        n_points costs no simulation, and why every cell of a sweep sees the
        same devices under the same split.
        """
        if self.split_mode == "interval":
            # Two pools, each fingerprinted by its own half of the space, so a
            # lower-band device can never be reused as an upper-band one.
            n = self.n_train if side == "train" else self.n_test
            return os.path.join(
                paths.data_path(POOLS_DIRNAME),
                device_factory.pool_name(n, self.resolution,
                                         self.capacitance_config(side)))
        return os.path.join(
            paths.data_path(POOLS_DIRNAME),
            device_factory.pool_name(self.n_devices, self.resolution,
                                     self.capacitance_config()))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["voltage_window"] = list(self.voltage_window)
        d["name"] = self.name
        return d

    def save(self) -> str:
        os.makedirs(self.dir, exist_ok=True)
        path = self.path(CONFIG_JSON)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, folder: str) -> "StudyConfig":
        """Read back the config.json run_1 wrote into a configuration folder."""
        path = os.path.join(folder, CONFIG_JSON)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{os.path.abspath(path)} is missing — run "
                f"run_1_generate_dataset.py for this configuration first")
        with open(path) as f:
            d = json.load(f)
        d.pop("name", None)
        fields = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in d.items() if k in fields})
        # A configuration keeps the folder it was found in: run_0 puts its
        # cells under results/<sweep>/ and the later stages must write back
        # into that same folder.
        cfg._dir = os.path.abspath(folder)
        return cfg

    def differences(self, other: "StudyConfig") -> Dict[str, Tuple]:
        """Settings on which two configurations disagree — for the warnings."""
        a, b = self.to_dict(), other.to_dict()
        return {k: (a[k], b[k]) for k in a if a[k] != b.get(k)}

    def _split_line(self) -> str:
        if self.split_mode == "interval":
            return (f"DISJOINT INTERVALS — all 14 capacitance ranges cut in "
                    f"half, {100 * self.interval_gap:.0f}% dead zone")
        return f"by device ID, seed {self.split_seed}"

    def describe(self) -> str:
        return (f"{self.name}\n"
                f"  measurement : {self.n_rays} rays x {self.n_points} points\n"
                f"  devices     : {self.n_train} train / {self.n_test} test, "
                f"{self.resolution} x {self.resolution} px\n"
                f"  split       : {self._split_line()}\n"
                f"  folder      : {os.path.abspath(self.dir)}")


def _candidate_folders() -> List[str]:
    """Every folder holding a config.json, sorted by name.

    Looked for in CONFIG_ROOT, data/ and results/, and one level down inside
    each, because run_0 gives every sweep its own folder with the
    configuration folders inside it while the hand-run programs write
    straight into data/.
    """
    roots, found = [paths.CONFIG_ROOT, paths.DATA_ROOT, paths.RESULTS], []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder) or name == POOLS_DIRNAME:
                continue
            if os.path.isfile(os.path.join(folder, CONFIG_JSON)):
                found.append(os.path.abspath(folder))
                continue
            for sub in sorted(os.listdir(folder)):
                inner = os.path.join(folder, sub)
                if os.path.isfile(os.path.join(inner, CONFIG_JSON)):
                    found.append(os.path.abspath(inner))
    # One folder per name, CONFIG_ROOT first: a budget left over in data/ from
    # an earlier hand run must not become a second row for the same
    # configuration.
    out, names = [], set()
    for folder in found:
        name = os.path.basename(folder)
        if name not in names:
            names.add(name)
            out.append(folder)
    return sorted(out, key=os.path.basename)


def existing_configs(every_sampling: bool = False) -> list:
    """Every configuration folder under data/, oldest name first.

    Discovery ignores the sampling arms: a 'grid' or 'random' arm sits at the
    same budget as the ray arm beside it, so including it would put two
    different measurements on the same point of the budget figure.  They are
    found only by name, or with every_sampling=True.
    """
    out, seen = [], set()
    for folder in _candidate_folders():
        if folder in seen:
            continue
        seen.add(folder)
        try:
            cfg = StudyConfig.load(folder)
        except Exception as exc:
            log.detail(f"[skip] {os.path.basename(folder)}: {exc}")
            continue
        if every_sampling or cfg.sampling == sampling.DEFAULT:
            out.append(cfg)
    return out


ALL = "ALL"


def resolve_configs(names) -> List[StudyConfig]:
    """Turn a settings-block entry into a list of configurations.

    "ALL" or None gives every budget of the study; a list gives those folders
    in the order written, which is the order the tables and figures come out
    in.  A missing name raises with the available names listed, rather than
    being skipped: a comparison silently missing a configuration says
    something untrue.
    """
    if names is None or (isinstance(names, str) and names.strip().upper() == ALL):
        return existing_configs()
    if isinstance(names, str):
        names = [names]
    # Named explicitly, a sampling arm resolves like any other folder.
    by_name = {c.name: c for c in existing_configs(every_sampling=True)}
    picked, missing = [], []
    for name in names:
        if name in by_name:
            picked.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        listing = "\n  ".join(sorted(by_name)) or "(none — run run_1 first)"
        raise KeyError(
            f"no configuration folder(s) {missing} under "
            f"{os.path.abspath(paths.CONFIG_ROOT)}\navailable:\n  {listing}")
    return picked
