"""
paths.py — the project's directory layout, in one place.

Every path is anchored to this file, not to the working directory, so a
program behaves the same wherever it is started from.

    probabilistic-csd-reconstruction/    PROJECT_ROOT
      src/csdrecon/                      the library
      scripts/                           the programs you run
      data/                              DATA_ROOT — simulated device pools
      results/                           RESULTS   — one folder per run
"""
import os

# .../src/csdrecon/config/paths.py -> config -> csdrecon -> src -> root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# The simulated devices.  Expensive to make, independent of the measurement
# budget, and therefore shared by every run instead of copied into each one.
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# What a run produces: datasets, models, evaluations, tables and figures.
RESULTS = os.path.join(PROJECT_ROOT, "results")


# Where the configuration folders (one per budget) are written.  Overridable,
# because run_0 gives each sweep its own folder under results/ with the
# configuration folders inside it.  The device pools are deliberately not
# moved: they are the expensive part, do not depend on the budget, and stay
# shared in data/.
CONFIG_ROOT = DATA_ROOT


def set_config_root(path: str) -> str:
    """Point the configuration folders at *path* (run_0 does this)."""
    global CONFIG_ROOT
    CONFIG_ROOT = path
    os.makedirs(CONFIG_ROOT, exist_ok=True)
    return CONFIG_ROOT


def config_root(*parts: str) -> str:
    """Path to a configuration folder under the current CONFIG_ROOT."""
    return os.path.join(CONFIG_ROOT, *parts)


def data_path(*parts: str) -> str:
    """Path inside data/, e.g. data_path("_device_pools")."""
    return os.path.join(DATA_ROOT, *parts)
