"""
MiniRacer Environment: A 24x24 miniature top-down racing simulation.
"""
import numpy as np

class MiniRacer:
    """
    MiniRacer is a ring road on grass.
    - Observation: 24x24x3 RGB image (normalized [0, 1]).
    - Actions: 0 = steer left, 1 = straight, 2 = steer right.
    - Reward: 1.0 - |distance from center line| / half_width.
    - Hidden state: Clockwise vs Counter-Clockwise direction decided at reset().
    """
    SIZE = 24
    ACTIONS = 3
    R = 8.0
    HALF_W = 2.2
    SPEED = 1.5

    GRASS = np.array([0.16, 0.34, 0.16], dtype=np.float32)
    ROAD = np.array([0.44, 0.44, 0.46], dtype=np.float32)
    KERB = np.array([0.75, 0.72, 0.62], dtype=np.float32)
    CAR = np.array([0.92, 0.20, 0.15], dtype=np.float32)

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        c = (self.SIZE - 1) / 2
        yy, xx = np.mgrid[0:self.SIZE, 0:self.SIZE]
        rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)

        self.track = np.tile(self.GRASS, (self.SIZE, self.SIZE, 1)).astype(np.float32)
        self.track[np.abs(rad - self.R) <= self.HALF_W] = self.ROAD
        self.track[np.abs(np.abs(rad - self.R) - self.HALF_W) <= 0.45] = self.KERB
        self._c = c

        self.theta = 0.0
        self.d = 0.0
        self.phi = 0.0
        self.direction = 1
        self.t = 0
        self.reset()

    def reset(self):
        self.theta = float(self.rng.uniform(0, 2 * np.pi))
        self.d = float(self.rng.uniform(-1.0, 1.0))
        self.phi = float(self.rng.uniform(-0.2, 0.2))
        self.direction = int(self.rng.choice([-1, 1]))  # Hidden state: -1 CCW, +1 CW
        self.t = 0
        return self.observe()

    @property
    def state(self):
        return np.array([self.theta, self.d, self.phi, self.direction], dtype=np.float32)

    def observe(self):
        img = self.track.copy()
        r = self.R + self.d
        x = self._c + r * np.cos(self.theta)
        y = self._c + r * np.sin(self.theta)
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

        self.theta = (self.theta + self.direction * self.SPEED / (self.R + self.d)) % (2 * np.pi)
        reward = 1.0 - abs(self.d) / self.HALF_W
        self.t += 1
        done = (self.t >= 150)
        info = {
            "theta": self.theta,
            "d": self.d,
            "phi": self.phi,
            "direction": self.direction,
            "step": self.t
        }
        return self.observe(), float(reward), done, info

    def render_upscaled(self, scale=16):
        """
        Returns a high-resolution version of the observation for clear display (e.g. 384x384).
        """
        obs = self.observe()
        # Scale nearest-neighbor for sharp pixel car or bilinear
        upscaled = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1)
        return upscaled
