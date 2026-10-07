from abc import ABC, abstractmethod

import numpy as np


class PoseModel(ABC):
    name: str
    label: str
    keypoints: dict[str, int]
    threshold: float

    @abstractmethod
    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return keypoints (N, K, 2) and scores (N, K) for a BGR frame."""

    @abstractmethod
    def draw(self, frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Draw the skeleton and return the annotated frame."""

    def close(self) -> None:
        pass
