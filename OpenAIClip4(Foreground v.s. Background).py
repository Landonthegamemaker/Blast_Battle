import os
import shutil
import cv2
import numpy as np

BASE_DIR = r"C:\Users\YourName\Desktop\Blast_Attack\Assets"

def is_simple_background(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))  # normalize size

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Flatten pixels
        pixels = hsv.reshape(-1, 3)

        # Measure color variance
        variance = np.var(pixels, axis=0).mean()

        # Measure edge density
        edges = cv2.Canny(img, 100, 200)
        edge_ratio = np.sum(edges > 0) / (128 * 128)

        # 🔥 Heuristic thresholds (tune if needed)
        if variance < 500 and edge_ratio < 0.08:
            return True  # likely simple background (DEFENSE)
        else:
            return False  # likely scene (CHARACTER)

    except:
        return False

defense_dir = os.path.join(BASE_DIR, "Defense")
character_dir = os.path.join(BASE_DIR, "Characters")

for file in os.listdir(defense_dir):
    path = os.path.join(defense_dir, file)

    if not os.path.isfile(path):
        continue

    simple_bg = is_simple_background(path)

    if not simple_bg:
        # Complex scene → move to Characters
        shutil.move(path, os.path.join(character_dir, file))
        print(f"Moved to Characters (scene): {file}")

print("\n✅ Background-based cleanup complete.")