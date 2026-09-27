"""
CapacitanceMatrixGenerator — generates random capacitance matrices
from interval specifications.  Absorbs the old matrix_utils.py.
"""
import random
from typing import Dict, List, Optional

from ..config.capacitance_config import sample as sample_interval
from ..config import log


class CapacitanceMatrixGenerator:
    """
    Generates random capacitance matrices (symmetric and general) from
    interval specifications, with optional seed-based reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        # ONE stream for the whole lifetime of the generator.  Each generate_*
        # call used to build its own random.Random(self.seed): with a seed set
        # that made every matrix and every sample identical, so a seeded run
        # produced N copies of one device.  Reuse one stream and a seed means
        # what it should — reproducible, but still varied.
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_symmetric(
        self,
        intervals: Dict[str, List[float]],
        size: int,
        labels: List[List[str]],
    ) -> List[List[float]]:
        """Symmetric square matrix with entries drawn uniformly from `intervals`
        ({label: [min, max]}), side `size`, labelled by the size x size
        `labels` grid whose upper triangle mirrors the lower.
        """
        rng = self._rng
        matrix = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i, size):
                label = labels[i][j]
                spec = intervals.get(label, [0, 1])
                value = round(sample_interval(rng, spec), 4)
                matrix[i][j] = matrix[j][i] = value
        return matrix

    def generate_general(
        self,
        intervals: Dict[str, List[float]],
        shape: List[int],
        labels: List[List[str]],
    ) -> List[List[float]]:
        """
        Generate a general (non-symmetric) matrix.

        Parameters
        ----------
        intervals : dict       {label: [min, max]}
        shape     : [rows, cols]
        labels    : list       rows×cols label grid
        """
        rng = self._rng
        rows, cols = shape
        matrix = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                label = labels[i][j]
                spec = intervals.get(label, [0, 1])
                matrix[i][j] = round(sample_interval(rng, spec), 4)
        return matrix

    # ------------------------------------------------------------------
    # Convenience: generate all four DQD matrices at once
    # ------------------------------------------------------------------

    def generate_all(self, config) -> Dict[str, List]:
        """
        Generate Cdd, Cgd, Cds, Cgs from a CapacitanceConfig instance.

        Returns a dict with keys "Cdd", "Cgd", "Cds", "Cgs".
        """
        intervals = config.intervals
        return {
            "Cdd": self.generate_symmetric(
                intervals["Cdd"], size=2, labels=config.labels_Cdd
            ),
            "Cgd": self.generate_general(
                intervals["Cgd"], shape=[2, 3], labels=config.labels_Cgd
            ),
            "Cds": self.generate_general(
                intervals["Cds"], shape=[1, 2], labels=config.labels_Cds
            ),
            "Cgs": self.generate_general(
                intervals["Cgs"], shape=[1, 3], labels=config.labels_Cgs
            ),
        }

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    @staticmethod
    def display(matrix: List[List[float]], labels: List[List[str]], name: str) -> None:
        """Pretty-print a matrix with its labels."""
        log.detail(f"\nGenerated {name} Matrix:")
        for i, row in enumerate(matrix):
            row_strs = [f"{labels[i][j]}={matrix[i][j]}" for j in range(len(row))]
            log.detail("[ " + ", ".join(row_strs) + " ]")
