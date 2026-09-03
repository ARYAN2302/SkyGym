"""SkyGym - a Gymnasium playground for counter-UAS recon data generation.

obs  = corrupted sensor detections (radar / EO / RF) about a flying drone
info = hidden ground truth ("witness channel") for labels and evaluation
"""
from .config import (EnvCfg, FlightCfg, RadarCfg, EOCfg, RFCfg, SensorRig,
                     ScenarioCfg, CLASSES)
from .env import SkyGymEnv
from .multidrone import MultiDroneEnv, sample_fleet
from .sensors import Detection, RadarSensor, EOSensor, RFSensor
from . import world

__version__ = "0.4.0"

__all__ = [
    "SkyGymEnv", "MultiDroneEnv", "sample_fleet",
    "EnvCfg", "FlightCfg", "RadarCfg", "EOCfg", "RFCfg",
    "SensorRig", "ScenarioCfg", "CLASSES",
    "Detection", "RadarSensor", "EOSensor", "RFSensor",
    "world", "__version__",
]

# Gymnasium registry (id available via gym.make, though direct
# construction with a custom EnvCfg is the primary usage pattern).
try:
    import gymnasium as gym
    gym.register(
        id="SkyGym-QuadTarget-v0",
        entry_point="skygym.env:SkyGymEnv",
    )
    gym.register(
        id="SkyGym-MultiDrone-v0",
        entry_point="skygym.multidrone:MultiDroneEnv",
    )
except Exception:  # pragma: no cover
    pass
