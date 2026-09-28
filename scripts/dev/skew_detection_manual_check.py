# Print the measured skew angle and tilt verdict for every .jpg photo in a folder.
#
# A manual check of app.pipeline.skew_detection against phone photos. The tilt threshold is
# 8 degrees. Nothing is written.
#
# Usage: backend/.venv/Scripts/python.exe scripts/dev/skew_detection_manual_check.py <photos_dir>
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from app.pipeline.skew_detection import is_photo_too_tilted, measure_skew_angle  # noqa: E402

photos_dir = Path(sys.argv[1])
for p in sorted(photos_dir.glob("*.jpg")):
    img = cv2.imread(str(p))
    angle = measure_skew_angle(img)
    too_tilted, angle2 = is_photo_too_tilted(img)
    print(f"{p.name:50s} angle={angle} too_tilted(threshold=8deg)={too_tilted}")
