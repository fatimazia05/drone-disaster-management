"""
Generates a handful of synthetic aerial-style frames so the detection
pipeline and dashboard can be demoed end-to-end without needing a real
drone or a licensed disaster-imagery dataset.

Run:  python create_sample_data.py
Creates: sample_data/frames/frame_00.jpg ... frame_05.jpg
"""

import os
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(__file__), "frames")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 640, 480


def base_terrain():
    """Green/brown terrain background to simulate an aerial view."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = (60, 110, 60)  # BGR greenish
    noise = np.random.randint(-15, 15, (H, W, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def add_blob(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1)
    return img


def frame_normal():
    return base_terrain()


def frame_fire():
    img = base_terrain()
    img = add_blob(img, (420, 260), 55, (0, 60, 230))   # red-orange fire
    img = add_blob(img, (440, 230), 35, (0, 120, 255))
    return img


def frame_smoke():
    img = base_terrain()
    img = add_blob(img, (200, 150), 70, (190, 190, 190))  # grey smoke
    img = add_blob(img, (240, 130), 50, (170, 170, 170))
    return img


def frame_flood():
    img = base_terrain()
    cv2.rectangle(img, (0, 300), (W, H), (180, 90, 30), -1)  # blue-ish water
    return img


def frame_stranded_person():
    img = base_terrain()
    cv2.rectangle(img, (300, 200), (340, 280), (40, 40, 40), -1)  # person silhouette
    cv2.circle(img, (320, 190), 12, (40, 40, 40), -1)
    return img


def frame_multi():
    img = base_terrain()
    img = add_blob(img, (450, 200), 45, (0, 60, 230))
    cv2.rectangle(img, (0, 340), (200, H), (180, 90, 30), -1)
    cv2.rectangle(img, (300, 220), (330, 290), (40, 40, 40), -1)
    return img


if __name__ == "__main__":
    generators = [
        frame_normal, frame_fire, frame_smoke,
        frame_flood, frame_stranded_person, frame_multi,
    ]
    for i, gen in enumerate(generators):
        path = os.path.join(OUT_DIR, f"frame_{i:02d}.jpg")
        cv2.imwrite(path, gen())
        print("wrote", path)
