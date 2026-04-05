import os
import cv2
import numpy as np
import shutil

# === CONFIG ===
BASE_DIR = r"C:/Users/dierk/OneDrive/Blast_Attack"
CHARACTER_DIR = os.path.join(BASE_DIR, "Characters")
DEFENSE_DIR = os.path.join(BASE_DIR, "Defense")

# Create Character folder if it doesn't exist
os.makedirs(CHARACTER_DIR, exist_ok=True)
# Create Defense folder if it doesn't exist
os.makedirs(DEFENSE_DIR, exist_ok=True)

# === THRESHOLDS (TUNE THESE IF NEEDED) ===
LOW_EDGE = 0.09
CENTER_EDGE_THRESHOLD = 0.12

# === PROCESS ===
for file in os.listdir(CHARACTER_DIR):
    file_lower = file.lower()

    # Only process images
    if not file_lower.endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(CHARACTER_DIR, file)

    # Read image
    img = cv2.imread(path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === METRIC 1: Variance ===
    variance = np.var(gray)

    # === METRIC 2: Edge Ratio ===
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.mean(edges > 0)

    # === METRIC 3: Center Edge Density ===
    h, w = edges.shape
    center = edges[h//4:3*h//4, w//4:3*w//4]
    center_edge_ratio = np.mean(center > 0)

    # === DEBUG OUTPUT ===
    print(f"{file} → var={variance:.0f}, edges={edge_ratio:.3f}, center={center_edge_ratio:.3f}")

    # === CLASSIFICATION LOGIC ===
    if edge_ratio < LOW_EDGE and center_edge_ratio < CENTER_EDGE_THRESHOLD:
        label = "Defense"
    else:
        label = "Character"

    # === ACTION ===
    if label == "Defense":
        new_path = os.path.join(DEFENSE_DIR, file)
        shutil.move(path, new_path)
        print(f"✅ Moved to Defense: {file}")
    else:
        print(f"➡️ Kept in Character: {file}")

print("\n🎯 Iteration 10 complete.")