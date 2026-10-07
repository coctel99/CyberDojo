import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2

from models.registry import MODELS, create_model
from utils import (
    METRIC_COLUMNS,
    compute_metrics,
    draw_angle_text,
    draw_point,
    draw_text,
    extract_points,
    point_value,
    resize_for_preview,
    safe_value,
)

VIDEO_NAME = "video.mp4"
DEFAULT_MODEL = MODELS.SPINEPOSE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--video", default=VIDEO_NAME)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    video_path = script_dir / args.video
    output_path = script_dir / f"pose_output_{args.model}.mp4"
    data_output_path = script_dir / f"motion_data_{args.model}.csv"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f} sec")
    print()

    print(f"Initializing model: {args.model}")
    model = create_model(args.model)
    print("Model initialized.")
    print()

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    joint_columns = [
        f"{name}_{suffix}"
        for name in model.keypoints
        for suffix in ("x", "y", "score")
    ]
    csv_columns = ["frame", "time_s", *joint_columns, *METRIC_COLUMNS]

    frame_index = 0
    processing_start = perf_counter()
    total_pose_time = 0.0

    with open(data_output_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        csv_writer.writeheader()

        while True:
            ret, frame = cap.read()

            if not ret or (args.max_frames and frame_index >= args.max_frames):
                break

            frame_start = perf_counter()

            pose_start = perf_counter()
            keypoints, scores = model.predict(frame)
            pose_time = perf_counter() - pose_start
            total_pose_time += pose_time

            annotated = frame.copy()

            if len(keypoints) > 0:
                points, point_scores = extract_points(
                    keypoints[0],
                    scores[0],
                    model.keypoints,
                    model.threshold,
                )

                metrics, derived = compute_metrics(points)

                row = {
                    "frame": frame_index,
                    "time_s": safe_value(frame_index / fps),
                    **metrics,
                }

                for name, point in points.items():
                    row[f"{name}_x"] = point_value(point, 0)
                    row[f"{name}_y"] = point_value(point, 1)
                    row[f"{name}_score"] = safe_value(point_scores[name])

                csv_writer.writerow(row)

                annotated = model.draw(annotated, keypoints, scores)

                draw_angle_text(annotated, points["left_elbow"], "L elbow", metrics["left_elbow_angle"])
                draw_angle_text(annotated, points["right_elbow"], "R elbow", metrics["right_elbow_angle"])
                draw_angle_text(annotated, points["left_knee"], "L knee", metrics["left_knee_angle"])
                draw_angle_text(annotated, points["right_knee"], "R knee", metrics["right_knee_angle"])

                draw_point(annotated, derived["pelvis_center"], "pelvis")

            frame_time = perf_counter() - frame_start
            total_fps = 1.0 / frame_time if frame_time > 0 else 0
            pose_fps = 1.0 / pose_time if pose_time > 0 else 0

            draw_text(annotated, f"Frame {frame_index + 1}/{frame_count}", (20, 40))
            draw_text(annotated, f"Total: {total_fps:.1f} FPS", (20, 70))
            draw_text(annotated, f"{model.label}: {pose_fps:.1f} FPS", (20, 100))

            writer.write(annotated)

            frame_index += 1

            if not args.no_preview:
                cv2.imshow(f"Fencing Coach - {model.label}", resize_for_preview(annotated))

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    processing_time = perf_counter() - processing_start

    cap.release()
    writer.release()
    model.close()
    cv2.destroyAllWindows()

    average_fps = frame_index / processing_time if processing_time > 0 else 0
    avg_pose_ms = total_pose_time / frame_index * 1000 if frame_index > 0 else 0

    print()
    print("Finished.")
    print(f"Processed frames: {frame_index}")
    print(f"Processing time: {processing_time:.2f} sec")
    print(f"Average total FPS: {average_fps:.2f}")
    print(f"Average {model.label} time: {avg_pose_ms:.1f} ms")
    print()
    print(f"Video output: {output_path}")
    print(f"Motion data: {data_output_path}")


if __name__ == "__main__":
    main()
