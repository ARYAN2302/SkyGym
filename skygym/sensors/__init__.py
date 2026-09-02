"""Recon sensor channel models (detection-level).

Each channel consumes the TRUE drone state + site geometry and emits a list
of Detection records - the corrupted, partial, latency-shifted view that a
real sensor would report. Ground truth NEVER passes through a sensor.

Shared machinery lives in base.py: the Detection record, the class-confusion
posterior generator, and the base Sensor class handling rate gating.
"""
from .base import Detection, CLASSES, Sensor, class_posterior
from .radar import RadarSensor
from .eo import EOSensor
from .rf import RFSensor

SENSORS = {"radar": RadarSensor, "eo": EOSensor, "rf": RFSensor}

__all__ = [
    "Detection", "CLASSES", "Sensor", "class_posterior",
    "RadarSensor", "EOSensor", "RFSensor", "SENSORS",
]
