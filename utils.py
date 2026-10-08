import math
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def output_path(name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name


METRIC_COLUMNS = [
    "shoulder_center_x", "shoulder_center_y",
    "pelvis_center_x", "pelvis_center_y",
    "left_elbow_angle", "right_elbow_angle",
    "left_hip_angle", "right_hip_angle",
    "left_knee_angle", "right_knee_angle",
    "left_foot_angle", "right_foot_angle",
    "shoulder_angle", "pelvis_angle",
    "shoulder_pelvis_angle_difference",
    "torso_angle",
    "shoulder_width_px",
    "wrist_distance_px", "wrist_distance_normalized",
    "ankle_distance_px", "ankle_distance_normalized",
    "heel_distance_px", "heel_distance_normalized",
    "pelvis_height_normalized",
]


def resize_for_preview(frame, max_width=900, max_height=900):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale == 1.0:
        return frame

    return cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def get_keypoint(keypoints, scores, index, threshold):
    score = float(scores[index])

    if score < threshold:
        return None, score

    return np.array(keypoints[index], dtype=np.float32), score


def extract_points(keypoints, scores, keypoint_map, threshold):
    points = {}
    point_scores = {}

    for name, index in keypoint_map.items():
        points[name], point_scores[name] = get_keypoint(keypoints, scores, index, threshold)

    return points, point_scores


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None

    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_line_angle(a, b):
    if a is None or b is None:
        return None

    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def calculate_distance(a, b):
    if a is None or b is None:
        return None

    return float(np.linalg.norm(a - b))


def midpoint(a, b):
    if a is None or b is None:
        return None

    return (a + b) / 2.0


def calculate_torso_angle(shoulder_center, pelvis_center):
    if shoulder_center is None or pelvis_center is None:
        return None

    dx = shoulder_center[0] - pelvis_center[0]
    dy = pelvis_center[1] - shoulder_center[1]
    return math.degrees(math.atan2(dx, dy))


def normalize_angle(angle):
    if angle is None:
        return None

    return (angle + 180.0) % 360.0 - 180.0


def safe_value(value):
    if value is None:
        return ""

    return round(float(value), 4)


def point_value(point, index):
    if point is None:
        return ""

    return safe_value(point[index])


def normalized(value, scale):
    if value is None or scale is None or scale <= 0:
        return None

    return value / scale


def compute_metrics(points):
    shoulder_center = midpoint(points["left_shoulder"], points["right_shoulder"])
    pelvis_center = midpoint(points["left_hip"], points["right_hip"])

    shoulder_angle = calculate_line_angle(points["left_shoulder"], points["right_shoulder"])
    pelvis_angle = calculate_line_angle(points["left_hip"], points["right_hip"])

    shoulder_pelvis_difference = None
    if shoulder_angle is not None and pelvis_angle is not None:
        shoulder_pelvis_difference = normalize_angle(shoulder_angle - pelvis_angle)

    shoulder_width = calculate_distance(points["left_shoulder"], points["right_shoulder"])
    wrist_distance = calculate_distance(points["left_wrist"], points["right_wrist"])
    ankle_distance = calculate_distance(points["left_ankle"], points["right_ankle"])
    heel_distance = calculate_distance(points["left_heel"], points["right_heel"])

    pelvis_y = pelvis_center[1] if pelvis_center is not None else None

    values = {
        "shoulder_center_x": point_value(shoulder_center, 0),
        "shoulder_center_y": point_value(shoulder_center, 1),
        "pelvis_center_x": point_value(pelvis_center, 0),
        "pelvis_center_y": point_value(pelvis_center, 1),
        "left_elbow_angle": safe_value(calculate_angle(
            points["left_shoulder"], points["left_elbow"], points["left_wrist"])),
        "right_elbow_angle": safe_value(calculate_angle(
            points["right_shoulder"], points["right_elbow"], points["right_wrist"])),
        "left_hip_angle": safe_value(calculate_angle(
            points["left_shoulder"], points["left_hip"], points["left_knee"])),
        "right_hip_angle": safe_value(calculate_angle(
            points["right_shoulder"], points["right_hip"], points["right_knee"])),
        "left_knee_angle": safe_value(calculate_angle(
            points["left_hip"], points["left_knee"], points["left_ankle"])),
        "right_knee_angle": safe_value(calculate_angle(
            points["right_hip"], points["right_knee"], points["right_ankle"])),
        "left_foot_angle": safe_value(calculate_line_angle(
            points["left_heel"], points["left_big_toe"])),
        "right_foot_angle": safe_value(calculate_line_angle(
            points["right_heel"], points["right_big_toe"])),
        "shoulder_angle": safe_value(shoulder_angle),
        "pelvis_angle": safe_value(pelvis_angle),
        "shoulder_pelvis_angle_difference": safe_value(shoulder_pelvis_difference),
        "torso_angle": safe_value(calculate_torso_angle(shoulder_center, pelvis_center)),
        "shoulder_width_px": safe_value(shoulder_width),
        "wrist_distance_px": safe_value(wrist_distance),
        "wrist_distance_normalized": safe_value(normalized(wrist_distance, shoulder_width)),
        "ankle_distance_px": safe_value(ankle_distance),
        "ankle_distance_normalized": safe_value(normalized(ankle_distance, shoulder_width)),
        "heel_distance_px": safe_value(heel_distance),
        "heel_distance_normalized": safe_value(normalized(heel_distance, shoulder_width)),
        "pelvis_height_normalized": safe_value(normalized(pelvis_y, shoulder_width)),
    }

    return values, {"pelvis_center": pelvis_center}


def draw_text(frame, text, position, scale=0.7, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_angle_text(frame, point, label, angle):
    if point is None or angle == "":
        return

    draw_text(
        frame,
        f"{label}: {angle:.1f}",
        (int(point[0]) + 10, int(point[1]) - 10),
        scale=0.5,
    )


def draw_point(frame, point, label, radius=6):
    if point is None:
        return

    x, y = int(point[0]), int(point[1])
    cv2.circle(frame, (x, y), radius, (255, 255, 255), -1)
    draw_text(frame, label, (x + 8, y - 8), scale=0.45, thickness=1)
