"""
Circular ring road implementation of Track.
"""
import numpy as np
from car_racer.tracks.base import Track

class CircularTrack(Track):
    """
    Standard circular ring road (default baseline).
    Radius R = 8.0, Half-width = 2.2, Size = 24.
    """
    def __init__(self, size: int = 24, r: float = 8.0, half_w: float = 2.2):
        super().__init__(name="circle", size=size, half_w=half_w)
        self.R = r
        self._c = (size - 1) / 2.0
        self._length = 2.0 * np.pi * self.R

    @property
    def length(self) -> float:
        return self._length

    def centerline_point(self, s: float) -> tuple[float, float]:
        theta = (s / self.R) % (2.0 * np.pi)
        x = self._c + self.R * np.cos(theta)
        y = self._c + self.R * np.sin(theta)
        return float(x), float(y)

    def tangent_heading(self, s: float) -> float:
        theta = (s / self.R) % (2.0 * np.pi)
        # Tangent vector is (-sin(theta), cos(theta))
        return float(theta + np.pi / 2.0)

    def normal_vector(self, s: float) -> tuple[float, float]:
        theta = (s / self.R) % (2.0 * np.pi)
        # Normal points outward along radial direction: (cos(theta), sin(theta))
        return float(np.cos(theta)), float(np.sin(theta))

    def curvature(self, s: float) -> float:
        return 1.0 / self.R

    def render_canvas(self) -> np.ndarray:
        c = self._c
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)

        canvas = np.tile(self.GRASS, (self.size, self.size, 1)).astype(np.float32)
        canvas[np.abs(rad - self.R) <= self.half_w] = self.ROAD
        canvas[np.abs(np.abs(rad - self.R) - self.half_w) <= 0.45] = self.KERB
        return canvas
