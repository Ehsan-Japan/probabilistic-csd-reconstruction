"""
OverlayRenderer — overlays scanned cells, peaks, and rays on charge-sensor images.
Absorbs the old plot_overlay.py, utility4.py, and utility6.py.
"""
import numpy as np
from typing import List, Tuple


Coords = List[Tuple[float, float]]


def _load_grid(npy_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a [Vx, Vy, z] .npy file and return (z_2d, Vx_values, Vy_values)."""
    array = np.load(npy_file)
    Vx_values = np.unique(array[:, 0])
    Vy_values = np.unique(array[:, 1])
    z_2d = array[:, 2].reshape((len(Vy_values), len(Vx_values)))
    return z_2d, Vx_values, Vy_values


class OverlayRenderer:
    """
    Generates overlay images that highlight scanned cells, detected peaks,
    and ray paths on top of charge-sensor or double-dot background images.

    All methods are stateless; pass required parameters explicitly.
    """

    # ------------------------------------------------------------------
    # Voltage-coordinate file  (shared parser)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Single-file voltage-coordinate overlay  (from old plot_overlay.py)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # All-rays overlay  (from old plot_overlay.py)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sample-wide scanned/peaks summary txt  (utility4.py)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Ground-truth binary array  (utility4.py)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_ground_truth_array(
        data_path: str,
        output_npy_path: str,
    ) -> np.ndarray:
        """Build a binary ground-truth array from double_dot_data.npy and save it
        as a .npy file, shape (num_rows, num_cols), dtype uint8.

        The z column is already an exact binary transition label (1 = charge
        state change), which preserves interdot lines that have no measurable
        charge-sensor response.
        """
        z_2d, _, _ = _load_grid(data_path)
        binary_array = (z_2d > 0.5).astype(np.uint8)
        np.save(output_npy_path, binary_array)
        return binary_array

    # ------------------------------------------------------------------
    # 2-D ground-truth grid visualisations
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

