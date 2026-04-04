import os
import shutil
import torch
import clip
from PIL import Image

# ===== PATHS =====
# Path to your messy folder
SOURCE_DIR = "C:/Users/dierk/OneDrive/Blast_Attack/Unsorted"
# Destination base folder
DEST_DIR = "C:/Users/dierk/OneDrive/Blast_Attack"

# ===== LOAD CLIP =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ===== LABELS =====
labels = ["a weapon", "a character", "armor or protective gear", "a game map or zone"]
text = clip.tokenize(labels).to(device)

# ===== COUNTERS FOR RENAMING =====
counters = {
    "Blasters": 1,
    "Characters": 1,
    "Defense": 1,
    "Zones": 1,
    "Unsorted": 1
}

# ===== CREATE FOLDERS =====
for folder in counters.keys():
    os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

# ===== CLASSIFY FUNCTION =====
def classify_image(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        return labels[probs.argmax()]
    
    except:
        return "unknown"

# ===== MAIN LOOP =====
for filename in os.listdir(SOURCE_DIR):
    file_path = os.path.join(SOURCE_DIR, filename)
    if not os.path.isfile(file_path):
        continue

    ext = filename.lower().split('.')[-1]

    # 🔥 STEP 1: Quick filter by file type
    if ext == "png":
        category = "Blasters"

    elif ext in ["jpg", "jpeg"]:
        result = classify_image(file_path)

        if "weapon" in result:
            category = "Blasters"
        elif "character" in result:
            category = "Characters"
        elif "armor" in result:
            category = "Defense"
        elif "map" in result or "zone" in result:
            category = "Zones"
        else:
            category = "Unsorted"
    else:
        category = "Unsorted"

    # ===== AUTO-RENAME =====
    count = counters[category]
    new_name = f"{category[:2].upper()}_{count}.png"
    counters[category] += 1

    dest_path = os.path.join(DEST_DIR, category, new_name)

    shutil.move(file_path, dest_path)

    print(f"Moved → {category}: {new_name}")

print("\n✅ AI Sorting + Renaming Complete.")