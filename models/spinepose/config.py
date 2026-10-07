from spinepose.metainfo import metainfo

MODE = "medium"
DETECTOR = "rfdetr"
USE_DETECTOR = True
HARDWARE_ACCELERATION = True
THRESHOLD = 0.4

LABEL = "SpinePose"

KEYPOINTS = {info["name"]: index for index, info in metainfo["keypoint_info"].items()}
