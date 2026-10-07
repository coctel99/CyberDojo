from ..base import PoseModel
from . import config
from .utils import build_estimator, draw_pose


class SpinePoseModel(PoseModel):
    name = "spinepose"
    label = config.LABEL
    keypoints = config.KEYPOINTS
    threshold = config.THRESHOLD

    def __init__(self):
        self.estimator = build_estimator()

    def predict(self, frame):
        return self.estimator(frame)

    def draw(self, frame, keypoints, scores):
        return draw_pose(frame, keypoints, scores)

    def close(self):
        self.estimator.close()
