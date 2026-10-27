"""
MiniRacer Environment: A 24x24 miniature top-down racing simulation.
"""
import numpy as np
from car_racer.tracks.base import Track
from car_racer.tracks.circle import CircularTrack

class MiniRacer:
    """
    MiniRacer is a top-down track racing environment.
    - Observation: 24x24x3 RGB image (normalized [0, 1]).
    - Actions: 0 = steer left, 1 = straight, 2 = steer right.
    - Reward: 1.0 - |distance from center line| / half_width.
    - Hidden state: Direction (forward vs reverse) decided at reset().
    """
    SIZE = 24
    ACTIONS = 3
    R = 8.0
    HALF_W = 2.2
    SPEED = 1.5

    GRASS = Track.GRASS
    ROAD = Track.ROAD
    KERB = Track.KERB
    CAR = np.array([0.92, 0.20, 0.15], dtype=np.float32)

    def __init__(self, track=None, seed=None):
        self.rng = np.random.default_rng(seed)
        if track is None or (isinstance(track, str) and track == "circle"):
            self.track_obj = CircularTrack(size=self.SIZE, r=self.R, half_w=self.HALF_W)
        elif isinstance(track, Track):
            self.track_obj = track
        elif isinstance(track, str):
            from car_racer.tracks.registry import get_track
            self.track_obj = get_track(track, size=self.SIZE, half_w=self.HALF_W)
        else:
            raise ValueError(f"Unknown track parameter: {track}")

        self.HALF_W = self.track_obj.half_w
        self.track = self.track_obj.get_canvas()
        self._c = (self.SIZE - 1) / 2.0

        self.s = 0.0
        self.d = 0.0
        self.phi = 0.0
        self.direction = 1
        self.t = 0
        self.reset()

    @property
    def theta(self) -> float:
        """Backward-compatible theta property for circular track."""
        return (self.s / self.R) % (2.0 * np.pi)

    def reset(self):
        self.s, self.d, self.phi, self.direction = self.track_obj.sample_spawn(self.rng)
        self.t = 0
        return self.observe()

    @property
    def state(self):
        return np.array([self.theta, self.d, self.phi, self.direction], dtype=np.float32)

    def observe(self):
        img = self.track.copy()
        x, y = self.track_obj.position(self.s, self.d)
        xi, yi = int(round(x)), int(round(y))
        for dx in (0, 1):
            for dy in (0, 1):
                px, py = xi + dx - 1, yi + dy - 1
                if 0 <= px < self.SIZE and 0 <= py < self.SIZE:
                    img[py, px] = self.CAR
        return img

    def step(self, action):
        """
        Executes one timestep in the environment.
        action: 0 (left), 1 (straight), 2 (right)
        Returns: (observation, reward, done, info)
        """
        self.phi = float(np.clip(self.phi + 0.16 * (action - 1), -0.7, 0.7))
        self.d = float(self.d + 1.1 * np.sin(self.phi))

        # Scraping the kerb damping
        if abs(self.d) > self.HALF_W:
            self.d = float(np.clip(self.d, -self.HALF_W, self.HALF_W))
            self.phi *= 0.4

        kappa = self.track_obj.curvature(self.s)
        curvature_factor = max(0.2, 1.0 + self.d * kappa)
        ds = self.direction * (self.SPEED / curvature_factor)
        self.s = (self.s + ds) % self.track_obj.length

        reward = 1.0 - abs(self.d) / self.HALF_W
        self.t += 1
        done = (self.t >= 150)
        info = {
            "theta": self.theta,
            "s": self.s,
            "d": self.d,
            "phi": self.phi,
            "direction": self.direction,
            "step": self.t,
            "track": self.track_obj.name
        }
        return self.observe(), float(reward), done, info

    def render_upscaled(self, scale=16):
        """
        Returns a high-resolution version of the observation for clear display (e.g. 384x384).
        """
        obs = self.observe()
        upscaled = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1)
        return upscaled
