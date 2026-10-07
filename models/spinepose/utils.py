from unittest.mock import patch

import numpy as np
from spinepose import SpinePoseEstimator
from spinepose.tools import base_solution
from spinepose.metainfo import metainfo
from spinepose.tools.visualization import draw_skeleton

from . import config


class FullFrameDetector:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        height, width = image.shape[:2]
        return np.array([[0, 0, width, height]], dtype=np.float32)


def build_estimator():
    def create():
        return SpinePoseEstimator(
            mode=config.MODE,
            detector=config.DETECTOR,
            hardware_acceleration=config.HARDWARE_ACCELERATION,
        )

    if config.USE_DETECTOR:
        return create()

    with patch.object(base_solution, config.DETECTOR.upper(), FullFrameDetector):
        return create()


def draw_pose(frame, keypoints, scores):
    scale = frame.shape[1] / 800

    return draw_skeleton(
        frame,
        keypoints,
        scores,
        metainfo,
        kpt_thr=config.THRESHOLD,
        radius=max(int(4 * scale), 2),
        line_width=max(int(2 * scale), 1),
    )
