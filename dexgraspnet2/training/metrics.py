"""
Metrics tracking for DexGraspNet2 training.

This module provides utilities for tracking and aggregating
training metrics over time.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and aggregate training metrics.

    Provides running averages, min/max values, and history
    for training metrics.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update({"loss": 0.5, "accuracy": 0.8})
        >>> tracker.update({"loss": 0.4, "accuracy": 0.85})
        >>> tracker.get_average("loss")
        0.45
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize the metrics tracker.

        Args:
            window_size: Size of the sliding window for running averages.
        """
        self._window_size = window_size
        self._values: Dict[str, List[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)
        self._sums: Dict[str, float] = defaultdict(float)

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric name to value.
        """
        for key, value in metrics.items():
            self._values[key].append(value)
            self._counts[key] += 1
            self._sums[key] += value

            # Keep only last window_size values
            if len(self._values[key]) > self._window_size:
                old_value = self._values[key].pop(0)
                self._sums[key] -= old_value

    def get_average(self, key: str) -> Optional[float]:
        """
        Get the running average for a metric.

        Args:
            key: Metric name.

        Returns:
            Running average, or None if metric not tracked.
        """
        if key not in self._values or len(self._values[key]) == 0:
            return None
        return self._sums[key] / len(self._values[key])

    def get_last(self, key: str) -> Optional[float]:
        """
        Get the most recent value for a metric.

        Args:
            key: Metric name.

        Returns:
            Last value, or None if metric not tracked.
        """
        if key not in self._values or len(self._values[key]) == 0:
            return None
        return self._values[key][-1]

    def get_min(self, key: str) -> Optional[float]:
        """
        Get the minimum value for a metric in the window.

        Args:
            key: Metric name.

        Returns:
            Minimum value, or None if metric not tracked.
        """
        if key not in self._values or len(self._values[key]) == 0:
            return None
        return min(self._values[key])

    def get_max(self, key: str) -> Optional[float]:
        """
        Get the maximum value for a metric in the window.

        Args:
            key: Metric name.

        Returns:
            Maximum value, or None if metric not tracked.
        """
        if key not in self._values or len(self._values[key]) == 0:
            return None
        return max(self._values[key])

    def get_all_averages(self) -> Dict[str, float]:
        """
        Get running averages for all tracked metrics.

        Returns:
            Dictionary of metric names to averages.
        """
        return {key: self.get_average(key) for key in self._values}

    def get_count(self, key: str) -> int:
        """
        Get the total number of updates for a metric.

        Args:
            key: Metric name.

        Returns:
            Total update count.
        """
        return self._counts.get(key, 0)

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._values.clear()
        self._counts.clear()
        self._sums.clear()

    def keys(self) -> List[str]:
        """Get list of tracked metric names."""
        return list(self._values.keys())


class EMAMetrics:
    """
    Exponential moving average metrics tracker.

    Uses exponential smoothing for a smoother view of metrics.

    Args:
        decay: EMA decay factor (higher = smoother).
    """

    def __init__(self, decay: float = 0.99):
        """Initialize the EMA tracker."""
        self._decay = decay
        self._values: Dict[str, float] = {}
        self._counts: Dict[str, int] = defaultdict(int)

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric name to value.
        """
        for key, value in metrics.items():
            if key in self._values:
                self._values[key] = (
                    self._decay * self._values[key] + (1 - self._decay) * value
                )
            else:
                self._values[key] = value
            self._counts[key] += 1

    def get(self, key: str) -> Optional[float]:
        """
        Get the EMA value for a metric.

        Args:
            key: Metric name.

        Returns:
            EMA value, or None if metric not tracked.
        """
        return self._values.get(key)

    def get_all(self) -> Dict[str, float]:
        """Get all EMA values."""
        return self._values.copy()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._values.clear()
        self._counts.clear()
