import os
import cv2
import numpy as np
import shutil

# === CONFIG ===
BASE_DIR = r"C:/Users/dierk/OneDrive/Blast_Attack"
CHAR_DIR = os.path.join(BASE_DIR, "Characters")
DEF_DIR = os.path.join(BASE_DIR, "Defense")

os.makedirs(CHAR_DIR, exist_ok=True)
os.makedirs(DEF_DIR, exist_ok=True)

# === THRESHOLDS ===
# Strict = cleaner Defense
LOW_EDGE = 0.042
CENTER_EDGE_THRESHOLD = 0.09
VARR_THRESHOLD = 1000

# === STORE MOVES (no conflicts) ===
moves = []

def classify_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variance = np.var(gray)

    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.mean(edges > 0)

    h, w = edges.shape
    center = edges[h//4:3*h//4, w//4:3*w//4]
    center_edge_ratio = np.mean(center > 0)

    print(f"{os.path.basename(path)} → var={variance:.0f}, edges={edge_ratio:.3f}, center={center_edge_ratio:.3f}")

    # === CLASSIFICATION ===
    if variance > VARR_THRESHOLD and edge_ratio < LOW_EDGE and center_edge_ratio < CENTER_EDGE_THRESHOLD:
        return "Defense"
    else:
        return "Character"

# === PASS 1: Analyze Characters Folder ===
for file in os.listdir(CHAR_DIR):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(CHAR_DIR, file)
    label = classify_image(path)

    if label == "Defense":
        target = os.path.join(DEF_DIR, file)
        moves.append((path, target))
        print(f"➡️ Queue move: Characters → Defense | {file}")
    else:
        print(f"✅ Stay: Characters | {file}")

# === PASS 2: Analyze Defense Folder ===
for file in os.listdir(DEF_DIR):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(DEF_DIR, file)
    label = classify_image(path)

    if label == "Character":
        target = os.path.join(CHAR_DIR, file)
        moves.append((path, target))
        print(f"➡️ Queue move: Defense → Characters | {file}")
    else:
        print(f"✅ Stay: Defense | {file}")

# === PASS 3: EXECUTE MOVES (safe) ===
print("\n🚀 Executing Moves...\n")

for src, dst in moves:
    # Prevent overwrite conflicts
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        counter = 1
        new_dst = f"{base}_{counter}{ext}"
        while os.path.exists(new_dst):
            counter += 1
            new_dst = f"{base}_{counter}{ext}"
        dst = new_dst

    shutil.move(src, dst)
    print(f"✅ Moved: {os.path.basename(src)}")

print("\n🎯 Bidirectional sorting complete.")