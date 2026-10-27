"""
Built-in racetrack presets: Oval, Figure-8, and S-Curve.
"""
import numpy as np
from car_racer.tracks.circle import CircularTrack
from car_racer.tracks.parametric import ParametricClosedTrack

class OvalTrack(ParametricClosedTrack):
    """
    Oval / Stadium racetrack with long straights and curved corners.
    """
    def __init__(self, size: int = 24, half_w: float = 2.2):
        c = (size - 1) / 2.0
        # Elliptical racetrack with major axis 8.5 and minor axis 5.5
        def curve(u: float) -> tuple[float, float]:
            theta = 2.0 * np.pi * u
            x = c + 8.5 * np.cos(theta)
            y = c + 5.5 * np.sin(theta)
            return float(x), float(y)

        super().__init__(name="oval", curve_fn=curve, num_samples=1000, size=size, half_w=half_w)


class Figure8Track(ParametricClosedTrack):
    """
    Figure-8 (infinity shape) racetrack testing alternating left and right turns.
    """
    def __init__(self, size: int = 24, half_w: float = 2.0):
        c = (size - 1) / 2.0
        def curve(u: float) -> tuple[float, float]:
            theta = 2.0 * np.pi * u
            x = c + 7.5 * np.cos(theta)
            y = c + 5.5 * np.sin(2.0 * theta)
            return float(x), float(y)

        super().__init__(name="figure8", curve_fn=curve, num_samples=1000, size=size, half_w=half_w)


class SCurveTrack(ParametricClosedTrack):
    """
    Serpentine / wavy 3-lobe racetrack with undulating curves.
    """
    def __init__(self, size: int = 24, half_w: float = 2.0):
        c = (size - 1) / 2.0
        def curve(u: float) -> tuple[float, float]:
            theta = 2.0 * np.pi * u
            r = 7.5 + 2.0 * np.cos(3.0 * theta)
            x = c + r * np.cos(theta)
            y = c + r * np.sin(theta)
            return float(x), float(y)

        super().__init__(name="s_curve", curve_fn=curve, num_samples=1000, size=size, half_w=half_w)
