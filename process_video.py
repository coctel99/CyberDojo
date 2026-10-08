import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2

from models.registry import MODELS, create_model
from pipeline import FramePipeline
from utils import output_path as outputs_file, resize_for_preview

VIDEO_NAME = "data/maki_uchi_1080p_60fps.mp4"
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
    output_path = outputs_file(f"{video_path.stem}_{args.model}.mp4")
    data_output_path = outputs_file(f"{video_path.stem}_{args.model}.csv")

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

    pipeline = FramePipeline(model)

    frame_index = 0
    processing_start = perf_counter()
    total_pose_time = 0.0

    with open(data_output_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=pipeline.csv_columns)
        csv_writer.writeheader()

        while True:
            ret, frame = cap.read()

            if not ret or (args.max_frames and frame_index >= args.max_frames):
                break

            frame_start = perf_counter()

            annotated, row, pose_time = pipeline.process(
                frame,
                frame_index,
                frame_index / fps,
            )
            total_pose_time += pose_time

            if row is not None:
                csv_writer.writerow(row)

            pipeline.draw_stats(
                annotated,
                f"Frame {frame_index + 1}/{frame_count}",
                perf_counter() - frame_start,
                pose_time,
            )

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
