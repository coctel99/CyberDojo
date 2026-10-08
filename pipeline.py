from time import perf_counter

from utils import (
    METRIC_COLUMNS,
    compute_metrics,
    draw_angle_text,
    draw_point,
    draw_text,
    extract_points,
    point_value,
    safe_value,
)


class FramePipeline:
    def __init__(self, model):
        self.model = model

    @property
    def csv_columns(self):
        joint_columns = [
            f"{name}_{suffix}"
            for name in self.model.keypoints
            for suffix in ("x", "y", "score")
        ]
        return ["frame", "time_s", *joint_columns, *METRIC_COLUMNS]

    def process(self, frame, frame_index, time_s):
        """Return (annotated frame, csv row or None, pose inference time in seconds)."""
        pose_start = perf_counter()
        keypoints, scores = self.model.predict(frame)
        pose_time = perf_counter() - pose_start

        annotated = frame.copy()

        if len(keypoints) == 0:
            return annotated, None, pose_time

        points, point_scores = extract_points(
            keypoints[0],
            scores[0],
            self.model.keypoints,
            self.model.threshold,
        )

        metrics, derived = compute_metrics(points)

        row = {
            "frame": frame_index,
            "time_s": safe_value(time_s),
            **metrics,
        }

        for name, point in points.items():
            row[f"{name}_x"] = point_value(point, 0)
            row[f"{name}_y"] = point_value(point, 1)
            row[f"{name}_score"] = safe_value(point_scores[name])

        annotated = self.model.draw(annotated, keypoints, scores)

        draw_angle_text(annotated, points["left_elbow"], "L elbow", metrics["left_elbow_angle"])
        draw_angle_text(annotated, points["right_elbow"], "R elbow", metrics["right_elbow_angle"])
        draw_angle_text(annotated, points["left_knee"], "L knee", metrics["left_knee_angle"])
        draw_angle_text(annotated, points["right_knee"], "R knee", metrics["right_knee_angle"])

        draw_point(annotated, derived["pelvis_center"], "pelvis")

        return annotated, row, pose_time

    def draw_stats(self, frame, header, total_time, pose_time):
        total_fps = 1.0 / total_time if total_time > 0 else 0
        pose_fps = 1.0 / pose_time if pose_time > 0 else 0

        draw_text(frame, header, (20, 40))
        draw_text(frame, f"Total: {total_fps:.1f} FPS", (20, 70))
        draw_text(frame, f"{self.model.label}: {pose_fps:.1f} FPS", (20, 100))
