"""
Parametric closed track implementation.
Enables any 2D closed curve to act as a racetrack.
"""
from typing import Callable
import numpy as np
from car_racer.tracks.base import Track

class ParametricClosedTrack(Track):
    """
    Constructs a closed racetrack from any parametric curve (x(u), y(u)) for u in [0, 1].
    Discretizes the curve into N equidistant arc-length segments.
    """
    def __init__(self,
                 name: str,
                 curve_fn: Callable[[float], tuple[float, float]],
                 num_samples: int = 1000,
                 size: int = 24,
                 half_w: float = 2.2):
        super().__init__(name=name, size=size, half_w=half_w)
        self.curve_fn = curve_fn
        self.num_samples = num_samples

        # Discretize curve
        u_vals = np.linspace(0.0, 1.0, num_samples, endpoint=False)
        pts = np.array([curve_fn(u) for u in u_vals], dtype=np.float32)
        # Append start point to close loop
        pts_closed = np.vstack([pts, pts[0:1]])

        diffs = np.diff(pts_closed, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))

        self.cum_s = np.zeros(num_samples + 1, dtype=np.float32)
        self.cum_s[1:] = np.cumsum(seg_lengths)
        self._length = float(self.cum_s[-1])

        # Normalize arc length parameterization
        self.pts_closed = pts_closed
        self.diffs = diffs
        self.seg_lengths = np.maximum(seg_lengths, 1e-6)

        # Precompute tangents and headings
        tangents = diffs / self.seg_lengths[:, None]
        self.headings = np.arctan2(tangents[:, 1], tangents[:, 0])

        # Precompute curvature d(heading)/ds
        d_headings = np.diff(np.unwrap(np.hstack([self.headings, self.headings[0:1]])))
        self.curvatures = d_headings / self.seg_lengths

    @property
    def length(self) -> float:
        return self._length

    def _get_index(self, s: float) -> tuple[int, float]:
        s_wrapped = float(s % self._length)
        idx = int(np.searchsorted(self.cum_s, s_wrapped, side="right") - 1)
        idx = max(0, min(idx, self.num_samples - 1))
        ds = s_wrapped - self.cum_s[idx]
        frac = ds / self.seg_lengths[idx]
        return idx, float(np.clip(frac, 0.0, 1.0))

    def centerline_point(self, s: float) -> tuple[float, float]:
        idx, frac = self._get_index(s)
        p0 = self.pts_closed[idx]
        p1 = self.pts_closed[idx + 1]
        pt = p0 + frac * (p1 - p0)
        return float(pt[0]), float(pt[1])

    def tangent_heading(self, s: float) -> float:
        idx, _ = self._get_index(s)
        return float(self.headings[idx])

    def curvature(self, s: float) -> float:
        idx, _ = self._get_index(s)
        return float(self.curvatures[idx])

    def render_canvas(self) -> np.ndarray:
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (576, 2)

        # Distance from each grid point to dense centerline points
        # grid_pts[:, None, :] shape (576, 1, 2)
        # pts_closed[None, :, :] shape (1, 1001, 2)
        dists = np.min(np.sqrt(np.sum((grid_pts[:, None, :] - self.pts_closed[None, :, :]) ** 2, axis=2)), axis=1)
        dist_grid = dists.reshape(self.size, self.size)

        canvas = np.tile(self.GRASS, (self.size, self.size, 1)).astype(np.float32)
        canvas[dist_grid <= self.half_w] = self.ROAD
        canvas[(dist_grid > self.half_w) & (dist_grid <= self.half_w + 0.45)] = self.KERB
        return canvas
