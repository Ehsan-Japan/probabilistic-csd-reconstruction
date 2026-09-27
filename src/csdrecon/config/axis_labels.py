"""
axis_labels.py — one place that decides what the x and y axes are called.

Every figure plots the same two gate voltages, so they must all carry the same
labels.  Set them once at the start of the program with set_axis_labels() and
every plot picks them up:

    ax.set_xlabel(x_label())      # -> "P1 (mV)"

Call the functions at plot time; a default argument is evaluated at import
time, before set_axis_labels() has run.
"""
from dataclasses import dataclass

DEFAULT_X_NAME = "P1"
DEFAULT_Y_NAME = "P2"
DEFAULT_X_UNIT = "mV"
DEFAULT_Y_UNIT = "mV"


@dataclass
class AxisLabels:
    """Axis names and units shared by every figure."""

    x_name: str = DEFAULT_X_NAME
    y_name: str = DEFAULT_Y_NAME
    x_unit: str = DEFAULT_X_UNIT
    y_unit: str = DEFAULT_Y_UNIT

    @property
    def x_label(self) -> str:
        return f"{self.x_name} ({self.x_unit})" if self.x_unit else self.x_name

    @property
    def y_label(self) -> str:
        return f"{self.y_name} ({self.y_unit})" if self.y_unit else self.y_name

_ACTIVE = AxisLabels()


def set_axis_labels(
    x_name: str = None,
    y_name: str = None,
    x_unit: str = None,
    y_unit: str = None,
) -> AxisLabels:
    """Set the axis names / units used by every plot. None = leave unchanged."""
    if x_name is not None:
        _ACTIVE.x_name = x_name
    if y_name is not None:
        _ACTIVE.y_name = y_name
    if x_unit is not None:
        _ACTIVE.x_unit = x_unit
    if y_unit is not None:
        _ACTIVE.y_unit = y_unit
    return _ACTIVE


def get_axis_labels() -> AxisLabels:
    """The active AxisLabels object."""
    return _ACTIVE


def x_label() -> str:
    """Full x-axis label, e.g. 'P1 (mV)'."""
    return _ACTIVE.x_label


def y_label() -> str:
    """Full y-axis label, e.g. 'P2 (mV)'."""
    return _ACTIVE.y_label
