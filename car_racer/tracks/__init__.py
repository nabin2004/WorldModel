"""
Track module for Mini World Models.
"""
from car_racer.tracks.base import Track
from car_racer.tracks.circle import CircularTrack
from car_racer.tracks.parametric import ParametricClosedTrack
from car_racer.tracks.presets import OvalTrack, Figure8Track, SCurveTrack
from car_racer.tracks.registry import get_track, list_tracks, register_track

__all__ = [
    "Track",
    "CircularTrack",
    "ParametricClosedTrack",
    "OvalTrack",
    "Figure8Track",
    "SCurveTrack",
    "get_track",
    "list_tracks",
    "register_track",
]
