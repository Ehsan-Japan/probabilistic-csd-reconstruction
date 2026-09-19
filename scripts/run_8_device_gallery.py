"""
CONTACT SHEETS — every simulated device on a page, train and test.

    python scripts/run_8_device_gallery.py

For each device: the raw charge-sensor image, the same image with the smooth
gate background differenced away (where the honeycomb is actually visible),
and the binary ground truth the network is scored against.  Many devices per
page, so a whole split can be checked at a glance — what kind of diagrams the
simulator produced, and whether train and test look like the same population.

Writes <configuration>/figures/gallery/<split>_page_NN.png, plus an
index.txt saying which device is on which page.  Nothing else is touched:
this only ever adds .png files.
"""
import _common
from csdrecon.study import device_gallery
from csdrecon.config import log

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Which configuration folders to draw, in this order.  "ALL" is every one.
CONFIG_NAMES = "ALL"

# Which splits.  ("train", "test") is both; ("test",) is the held-out
# devices alone.
SPLITS = ("train", "test")

# The panels drawn for each device, in this order.  Drop one to fit more
# devices on a page.
PANELS = ("charge_sensor", "charge_sensor_gradient", "stability_diagram")

# Page layout.  15 devices on a 3-wide page is a sheet that still reads at
# 150 dpi; raise PER_PAGE for fewer, denser pages.
PER_PAGE = 15
COLUMNS = 3
DPI = 150

# The voltage-grid cells behind the ground truth.  Off here: at thumbnail
# size the lattice hides the transition lines it is supposed to sit behind.
CELL_GRID = False

# ══════════════════════════════════════════════════════════════════════════


def main():
    _common.banner("CONTACT SHEETS — the devices, as they were simulated")
    cfgs = _common.configs(CONFIG_NAMES)

    def draw(cfg):
        return device_gallery.render_config(
            cfg, splits=SPLITS, panels=PANELS, per_page=PER_PAGE,
            columns=COLUMNS, dpi=DPI, cell_grid=CELL_GRID)

    done, failed = _common.for_each(cfgs, draw)
    _common.table([("configuration", 40), ("pages", 8)],
                  [(cfg.name, str(len(files))) for cfg, files in done],
                  failed)
    log.say("the pages are in <configuration>/figures/gallery/")


if __name__ == "__main__":
    main()
