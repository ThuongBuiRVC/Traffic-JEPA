"""WTS VQA label spaces: event phases and question categories (selectors).

These are the label spaces used by the model and the cached dataset. Kept in one small
module so nothing else has to be imported for them.
"""
from __future__ import annotations

SELECTORS = [
    "pedestrian_orientation", "pedestrian_position", "pedestrian_behavior",
    "vehicle_orientation", "vehicle_position", "vehicle_behavior",
    "pedestrian_ego_relative", "vehicle_ego_relative",
    "pedestrian_vehicle_relative", "pedestrian_speed", "vehicle_speed",
    "pedestrian_ego_distance", "vehicle_ego_distance",
    "pedestrian_vehicle_distance", "pedestrian_gaze", "pedestrian_count",
    "environment",
]

PHASES = [
    "environment",
    "prerecognition",
    "recognition",
    "judgement",
    "action",
    "avoidance",
    "unknown",
]
