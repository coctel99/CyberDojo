from pathlib import Path
from time import perf_counter

import cv2

from rtmlib import RTMPose, Wholebody, draw_skeleton
# from rtmlib import YOLOX

VIDEO_NAME = "video.mp4"
OUTPUT_NAME = "pose_output.mp4"

BACKEND = "onnxruntime"

POSE_DEVICE = "mps"
# DETECTOR_DEVICE = "cpu"

MODEL_MODE = "balanced"
KEYPOINT_THRESHOLD = 0.4


def resize_for_preview(frame, max_width=900, max_height=900):
    height, width = frame.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale == 1.0:
        return frame

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def main():
    script_dir = Path(__file__).resolve().parent
    video_path = script_dir / VIDEO_NAME
    output_path = script_dir / OUTPUT_NAME

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

    config = Wholebody.MODE[MODEL_MODE]

    # print("Initializing YOLOX detector...")
    # detector = YOLOX(
    #     config["det"],
    #     model_input_size=config["det_input_size"],
    #     backend=BACKEND,
    #     device=DETECTOR_DEVICE,
    # )
    # print("YOLOX initialized.")
    # print()

    print("Initializing RTMW pose model...")
    print(f"Pose device: {POSE_DEVICE}")

    pose_model = RTMPose(
        config["pose"],
        model_input_size=config["pose_input_size"],
        to_openpose=False,
        backend=BACKEND,
        device=POSE_DEVICE,
    )

    print("RTMW initialized.")
    print()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_index = 0
    processing_start = perf_counter()
    total_pose_time = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_start = perf_counter()

        # detection_start = perf_counter()
        # bboxes = detector(frame)
        # detection_time = perf_counter() - detection_start

        pose_start = perf_counter()
        keypoints, scores = pose_model(frame)

        # keypoints, scores = pose_model(
        #     frame,
        #     bboxes=bboxes,
        # )

        pose_time = perf_counter() - pose_start
        total_pose_time += pose_time

        annotated = frame.copy()
        annotated = draw_skeleton(
            annotated,
            keypoints,
            scores,
            openpose_skeleton=False,
            kpt_thr=KEYPOINT_THRESHOLD,
        )

        frame_time = perf_counter() - frame_start
        total_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        pose_fps = 1.0 / pose_time if pose_time > 0 else 0.0

        cv2.putText(
            annotated,
            f"Frame {frame_index + 1}/{frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Total: {total_fps:.1f} FPS",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"RTMW CoreML: {pose_fps:.1f} FPS",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)
        preview = resize_for_preview(annotated)

        cv2.imshow(
            "Fencing Coach - RTMW",
            preview,
        )

        frame_index += 1
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

    processing_time = perf_counter() - processing_start

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    average_fps = frame_index / processing_time if processing_time > 0 else 0
    avg_pose_ms = total_pose_time / frame_index * 1000 if frame_index > 0 else 0

    print()
    print("Finished.")
    print(f"Processed frames: {frame_index}")
    print(f"Processing time: {processing_time:.2f} sec")
    print(f"Average total FPS: {average_fps:.2f}")
    print(f"Average RTMW time: {avg_pose_ms:.1f} ms")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()