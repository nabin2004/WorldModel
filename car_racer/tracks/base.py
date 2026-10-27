"""
Base Track interface for MiniRacer.
"""
from abc import ABC, abstractmethod
import numpy as np

class Track(ABC):
    """
    Abstract base class for all track layouts.
    Coordinates are defined in a normalized grid of dimensions size x size (default 24x24).
    """
    GRASS = np.array([0.16, 0.34, 0.16], dtype=np.float32)
    ROAD = np.array([0.44, 0.44, 0.46], dtype=np.float32)
    KERB = np.array([0.75, 0.72, 0.62], dtype=np.float32)

    def __init__(self, name: str, size: int = 24, half_w: float = 2.2):
        self.name = name
        self.size = size
        self.half_w = half_w
        self._cached_canvas = None

    @property
    @abstractmethod
    def length(self) -> float:
        """Total closed-loop arc length L."""
        pass

    @abstractmethod
    def centerline_point(self, s: float) -> tuple[float, float]:
        """Returns (x, y) coordinates of centerline at arc-length distance s."""
        pass

    @abstractmethod
    def tangent_heading(self, s: float) -> float:
        """Returns local tangent angle psi(s) in radians in [-pi, pi)."""
        pass

    @abstractmethod
    def curvature(self, s: float) -> float:
        """Returns local signed curvature kappa(s) = d(psi)/ds."""
        pass

    def normal_vector(self, s: float) -> tuple[float, float]:
        """
        Returns unit normal vector (nx, ny) pointing in direction of positive lateral offset (+d).
        """
        psi = self.tangent_heading(s)
        return float(-np.sin(psi)), float(np.cos(psi))

    def position(self, s: float, d: float) -> tuple[float, float]:
        """Returns (x, y) for Frenet coordinates (s, d)."""
        cx, cy = self.centerline_point(s)
        nx, ny = self.normal_vector(s)
        return cx + d * nx, cy + d * ny

    @abstractmethod
    def render_canvas(self) -> np.ndarray:
        """Renders static 2D track canvas (size, size, 3) float32 in [0, 1]."""
        pass

    def get_canvas(self) -> np.ndarray:
        if self._cached_canvas is None:
            self._cached_canvas = self.render_canvas()
        return self._cached_canvas.copy()

    def sample_spawn(self, rng: np.random.Generator) -> tuple[float, float, float, int]:
        """
        Samples an initial (s, d, phi, direction) for episode reset.
        Returns:
            s: initial arc distance [0, L)
            d: lateral offset [-1.0, 1.0]
            phi: heading error [-0.2, 0.2]
            direction: +1 (forward) or -1 (reverse)
        """
        s = float(rng.uniform(0, self.length))
        d = float(rng.uniform(-1.0, 1.0))
        phi = float(rng.uniform(-0.2, 0.2))
        direction = int(rng.choice([-1, 1]))
        return s, d, phi, direction
