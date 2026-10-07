from rtmlib import RTMPose, Wholebody, draw_skeleton

from . import config


def build_pose_model():
    mode_config = Wholebody.MODE[config.MODE]

    return RTMPose(
        mode_config["pose"],
        model_input_size=mode_config["pose_input_size"],
        to_openpose=False,
        backend=config.BACKEND,
        device=config.DEVICE,
    )


def draw_pose(frame, keypoints, scores):
    return draw_skeleton(
        frame,
        keypoints,
        scores,
        openpose_skeleton=False,
        kpt_thr=config.THRESHOLD,
    )
