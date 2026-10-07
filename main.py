from pathlib import Path

import cv2


def main():
    # Directory where main.py is located
    script_dir = Path(__file__).resolve().parent

    video_path = script_dir / "video.mp4"

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

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f} sec")

    frame_index = 0

    while True:
        ret, frame = cap.read()


        if not ret:
            break

        frame_index += 1

        # Optional frame counter overlay
        cv2.putText(
            frame,
            f"Frame: {frame_index}/{frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Fencing Coach", frame)

        # Try to reproduce the video's original frame rate.
        delay_ms = max(1, int(1000 / fps)) if fps > 0 else 1

        key = cv2.waitKey(delay_ms) & 0xFF

        # Press Q or ESC to quit
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()