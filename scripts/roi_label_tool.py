#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Simple ROI labeling tool")
    p.add_argument("--frames_dir", required=True, help="Root directory of frames")
    p.add_argument("--output_json", required=True, help="Output JSON path")
    p.add_argument("--episodes", required=True, help="Episode range, e.g. 0-29")
    p.add_argument("--cameras", required=True, help="Comma-separated camera names")
    return p.parse_args()


def ep_label(ep):
    if 0 <= ep <= 9:
        return "banana"
    if 10 <= ep <= 19:
        return "socks"
    if 20 <= ep <= 29:
        return "strawberry"
    return "unknown"


def find_frames(frames_dir: Path, ep: int, camera: str):
    ep_dir = frames_dir / f"ep{ep:02d}"
    candidates = [
        ep_dir / camera,
        ep_dir / f"observation.images.{camera}",
        frames_dir / camera / f"ep{ep:02d}",
    ]
    frames = []
    for c in candidates:
        if c.exists():
            frames += sorted(c.glob("img_*.png"))
            frames += sorted(c.glob("attn_*.png"))
    # Flat naming scheme: episode_00_camera1.png
    flat = frames_dir / f"episode_{ep:02d}_{camera}.png"
    if flat.exists():
        frames.append(flat)
    return frames


def load_existing(path: Path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {item["path"]: item for item in data}


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    output_json = Path(args.output_json)
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    ep_start, ep_end = [int(x) for x in args.episodes.split("-")]

    existing = load_existing(output_json)
    results = list(existing.values())

    current_rect = None
    drawing = False
    ix = iy = -1

    def on_mouse(event, x, y, flags, param):
        nonlocal ix, iy, drawing, current_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            current_rect = (ix, iy, ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_rect = (ix, iy, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_rect = (ix, iy, x, y)

    for ep in range(ep_start, ep_end + 1):
        for cam in cams:
            frames = find_frames(frames_dir, ep, cam)
            if not frames:
                print(f"no frames for ep{ep:02d} {cam}")
                continue
            for f in frames:
                f_str = str(f)
                if f_str in existing:
                    continue
                img = cv2.imread(f_str)
                if img is None:
                    continue
                current_rect = None
                cv2.namedWindow("roi_label", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("roi_label", on_mouse)
                while True:
                    disp = img.copy()
                    if current_rect is not None:
                        x1, y1, x2, y2 = current_rect
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        disp,
                        f"ep{ep:02d} {cam} {ep_label(ep)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("roi_label", disp)
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("r"):
                        current_rect = None
                    elif key == ord("s"):
                        if current_rect is not None:
                            x1, y1, x2, y2 = current_rect
                            x1, x2 = sorted([x1, x2])
                            y1, y2 = sorted([y1, y2])
                            item = {
                                "episode": ep,
                                "camera": cam,
                                "label": ep_label(ep),
                                "path": f_str,
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            }
                            results.append(item)
                            existing[f_str] = item
                            output_json.write_text(json.dumps(results, indent=2))
                            break
                    elif key == ord("n"):
                        # skip frame without saving
                        break
                    elif key == ord("q"):
                        output_json.write_text(json.dumps(results, indent=2))
                        cv2.destroyAllWindows()
                        return
    output_json.write_text(json.dumps(results, indent=2))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
