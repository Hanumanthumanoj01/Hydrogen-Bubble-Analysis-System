"""
generate_pseudo_labels.py
=========================
Bootstraps a YOLO-format dataset by running the classical detector
(bubble_analysis.BubbleAnalyzer.detect_bubbles) on a folder of images and
writing out YOLO bounding-box labels.

IMPORTANT — read this before trusting the output:
This produces PSEUDO-labels, not ground truth. A model trained only on
these inherits every blind spot and false positive of the classical
detector — it will not exceed it. This script exists to:
  (a) prove the training pipeline runs end-to-end, and
  (b) give you a fast starting point to CORRECT rather than create from
      scratch (open the generated .txt files in a tool like Roboflow,
      CVAT, or LabelImg and fix them against the actual images — that
      human-corrected set is what will actually teach the model
      something the classical detector doesn't already know).

Usage:
    python generate_pseudo_labels.py <images_dir> <output_labels_dir>
"""
import sys
import os
import cv2
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from bubble_analysis import BubbleAnalyzer  # noqa: E402

analyzer = BubbleAnalyzer()


def generate(images_dir: str, labels_dir: str, sensitivity: str = "medium"):
    os.makedirs(labels_dir, exist_ok=True)
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    total = 0
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  skip (unreadable): {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bubbles = analyzer.detect_bubbles(gray, sensitivity=sensitivity)

        stem = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        with open(label_path, "w") as f:
            for x, y, r in bubbles:
                # YOLO format: class x_center y_center width height (all normalized 0-1)
                x_c, y_c = x / w, y / h
                box_w, box_h = (2 * r) / w, (2 * r) / h
                f.write(f"0 {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}\n")
        total += len(bubbles)
        print(f"  {os.path.basename(path)}: {len(bubbles)} pseudo-labels -> {label_path}")

    print(f"\nDone. {len(image_paths)} images, {total} total pseudo-labels.")
    print("These are STARTING POINTS for human correction, not ground truth.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    generate(sys.argv[1], sys.argv[2])
