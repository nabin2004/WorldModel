"""
Registry for looking up, registering, and listing available tracks.
"""
from typing import Callable, Dict, List
from car_racer.tracks.base import Track
from car_racer.tracks.circle import CircularTrack
from car_racer.tracks.presets import OvalTrack, Figure8Track, SCurveTrack

TRACK_REGISTRY: Dict[str, Callable[..., Track]] = {
    "circle": lambda **kw: CircularTrack(**kw),
    "oval": lambda **kw: OvalTrack(**kw),
    "figure8": lambda **kw: Figure8Track(**kw),
    "s_curve": lambda **kw: SCurveTrack(**kw),
    "wavy": lambda **kw: SCurveTrack(**kw),
}

def register_track(name: str, factory: Callable[..., Track]) -> None:
    """Registers a new track factory under a given name."""
    TRACK_REGISTRY[name.lower()] = factory

def get_track(name: str, **kwargs) -> Track:
    """
    Retrieves and instantiates a track by name.
    Raises ValueError if track is not found.
    """
    key = name.lower()
    if key not in TRACK_REGISTRY:
        available = ", ".join(list_tracks())
        raise ValueError(f"Unknown track '{name}'. Available tracks: {available}")
    return TRACK_REGISTRY[key](**kwargs)

def list_tracks() -> List[str]:
    """Returns a list of all registered track names (unique canonical names)."""
    return ["circle", "oval", "figure8", "s_curve"]
