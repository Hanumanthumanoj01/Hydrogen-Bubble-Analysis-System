"""
calibrate.py
============
Interactive calibration tool for the Hough Circle bubble detector.
Run this script directly to tune detection parameters on your sample images.

Usage:
  python calibrate.py --image data/samples/bubble_density_100mA.jpg
  python calibrate.py --image data/samples/bubble_density_200mA.jpg --sensitivity high

The script renders an OpenCV window with sliders for:
  - Top-hat kernel size (background suppression)
  - Median blur kernel
  - Hough dp
  - Hough minDist
  - Hough param1 (Canny high threshold)
  - Hough param2 (accumulator threshold)
  - minRadius, maxRadius

Press 's' to save the current parameters to calibration_params.json.
Press 'q' to quit.
"""

import argparse
import json
import cv2
import numpy as np
import os

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "data/calibration_params.json")


def nothing(_): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to sample image")
    ap.add_argument("--sensitivity", default="medium")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: cannot read {args.image}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image: {w}×{h}  |  Press 's' to save params, 'q' to quit")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", min(w * 2, 1600), min(h, 800))

    # Create trackbars
    cv2.createTrackbar("TopHat kernel", "Calibration", 31, 101, nothing)
    cv2.createTrackbar("Median blur", "Calibration", 5, 15, nothing)
    cv2.createTrackbar("Hough dp x10", "Calibration", 12, 30, nothing)
    cv2.createTrackbar("minDist", "Calibration", 18, 100, nothing)
    cv2.createTrackbar("param1", "Calibration", 50, 200, nothing)
    cv2.createTrackbar("param2", "Calibration", 28, 100, nothing)
    cv2.createTrackbar("minRadius", "Calibration", 4, 40, nothing)
    cv2.createTrackbar("maxRadius", "Calibration", 80, 200, nothing)

    while True:
        th_k = max(3, cv2.getTrackbarPos("TopHat kernel", "Calibration") | 1)
        med_k = max(3, cv2.getTrackbarPos("Median blur", "Calibration") | 1)
        dp = max(0.1, cv2.getTrackbarPos("Hough dp x10", "Calibration") / 10)
        minDist = max(1, cv2.getTrackbarPos("minDist", "Calibration"))
        p1 = max(1, cv2.getTrackbarPos("param1", "Calibration"))
        p2 = max(1, cv2.getTrackbarPos("param2", "Calibration"))
        minR = max(1, cv2.getTrackbarPos("minRadius", "Calibration"))
        maxR = max(minR + 1, cv2.getTrackbarPos("maxRadius", "Calibration"))

        # Top-hat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_k, th_k))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blurred = cv2.medianBlur(tophat, med_k)

        # Hough
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp,
                                    minDist=minDist, param1=p1, param2=p2,
                                    minRadius=minR, maxRadius=maxR)

        vis = img.copy()
        count = 0
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for x, y, r in circles:
                cv2.circle(vis, (x, y), r, (0, 0, 255), 1)
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            count = len(circles)

        # Info overlay
        cv2.putText(vis, f"Detected: {count} bubbles", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f"dp={dp:.1f} minDist={minDist} p1={p1} p2={p2} r=[{minR},{maxR}]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show tophat next to result
        tophat_color = cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([tophat_color, vis])
        scale = min(1600 / combined.shape[1], 800 / combined.shape[0])
        combined = cv2.resize(combined, None, fx=scale, fy=scale)
        cv2.imshow("Calibration", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            params = {"tophat_kernel": th_k, "median_blur": med_k, "dp": dp,
                      "minDist": minDist, "param1": p1, "param2": p2,
                      "minRadius": minR, "maxRadius": maxR}
            os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
            with open(PARAMS_FILE, "w") as f:
                json.dump(params, f, indent=2)
            print(f"Saved to {PARAMS_FILE}: {params}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
