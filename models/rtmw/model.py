from ..base import PoseModel
from . import config
from .utils import build_pose_model, draw_pose


class RTMWModel(PoseModel):
    name = "rtmw"
    label = config.LABEL
    keypoints = config.KEYPOINTS
    threshold = config.THRESHOLD

    def __init__(self):
        self.pose_model = build_pose_model()

    def predict(self, frame):
        return self.pose_model(frame)

    def draw(self, frame, keypoints, scores):
        return draw_pose(frame, keypoints, scores)
