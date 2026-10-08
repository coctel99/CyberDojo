import argparse
import csv
from datetime import datetime
from time import perf_counter

import cv2

from models.registry import MODELS, create_model
from pipeline import FramePipeline
from utils import output_path, resize_for_preview

DEFAULT_MODEL = MODELS.SPINEPOSE
DEFAULT_CAMERA = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Live camera pose analysis")
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA)
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {args.camera}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = output_path(f"live_{stamp}_{args.model}.mp4")
    data_file = output_path(f"live_{stamp}_{args.model}.csv")

    print(f"Initializing model: {args.model}")
    model = create_model(args.model)
    pipeline = FramePipeline(model)
    print("Model initialized. Press q or Esc to quit.")

    writer = cv2.VideoWriter(
        str(video_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    csv_file = open(data_file, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=pipeline.csv_columns)
    csv_writer.writeheader()

    frame_index = 0
    start = perf_counter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_start = perf_counter()

            annotated, row, pose_time = pipeline.process(
                frame,
                frame_index,
                perf_counter() - start,
            )

            if row is not None:
                csv_writer.writerow(row)

            pipeline.draw_stats(
                annotated,
                f"Frame {frame_index}",
                perf_counter() - frame_start,
                pose_time,
            )

            writer.write(annotated)
            frame_index += 1

            cv2.imshow(f"Fencing Coach Live - {model.label}", resize_for_preview(annotated))

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        writer.release()
        csv_file.close()
        model.close()
        cv2.destroyAllWindows()
        print(f"Video output: {video_file}")
        print(f"Motion data: {data_file}")


if __name__ == "__main__":
    main()
