"""
Grasp label generation module for DexGraspNet2.

This module provides tools for generating grasp labels for dexterous hands
using physics simulation (Isaac Gym).

Note:
    GraspSimulator requires explicit import due to Isaac Gym import order constraints:
        from dexgraspnet2.generation.grasp_simulator import GraspSimulator
"""

from dexgraspnet2.generation.grasp_generator import (
    GraspCandidate,
    GraspGenerator,
    GraspLabel,
)

# GraspSimulator is NOT auto-imported due to Isaac Gym requiring
# import before torch. Import explicitly when needed:
#   from dexgraspnet2.generation.grasp_simulator import GraspSimulator

__all__ = [
    "GraspCandidate",
    "GraspGenerator",
    "GraspLabel",
]
