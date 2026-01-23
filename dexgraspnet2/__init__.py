"""
DexGraspNet2 - Production-grade dexterous grasping inference pipeline.

This package provides a clean, hand-agnostic interface for dexterous grasp
pose generation using the DexGraspNet 2.0 method (CoRL 2024).

Key classes:
    - GraspPredictor: Main inference class for predicting grasp poses
    - HandConfig: Configuration class for different dexterous hands
    - GraspResult: Container for grasp predictions
    - GraspPose: Single grasp pose representation
    - GraspGenerator: Grasp label generation using physics simulation

Example:
    >>> from dexgraspnet2 import GraspPredictor, HandConfig
    >>> hand_config = HandConfig.leap_hand()
    >>> predictor = GraspPredictor(
    ...     checkpoint_path="checkpoints/model.pth",
    ...     hand_config=hand_config
    ... )
    >>> result = predictor.predict(point_cloud, segmentation)
    >>> best_grasp = result.best()
"""

__version__ = "0.1.0"

from dexgraspnet2.inference.grasp_predictor import GraspPredictor
from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.data.grasp_result import GraspResult, GraspPose
from dexgraspnet2.generation.grasp_generator import GraspGenerator

__all__ = [
    "GraspPredictor",
    "HandConfig",
    "GraspResult",
    "GraspPose",
    "GraspGenerator",
]
