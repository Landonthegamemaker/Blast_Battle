import os
import shutil
import torch
import clip
from PIL import Image

BASE_DIR = r"C:/Users/dierk/OneDrive/Blast_Attack"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 🔥 More specific labels (this is the upgrade)
labels = [
    "a person or character",
    "a weapon only",
    "armor or protective gear",
    "a character holding a weapon"
]

text = clip.tokenize(labels).to(device)

def classify(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        return labels[probs.argmax()]
    except:
        return "unknown"

# ===== BLASTERS CLEANUP =====
blaster_dir = os.path.join(BASE_DIR, "Blasters")

for file in os.listdir(blaster_dir):
    path = os.path.join(blaster_dir, file)

    result = classify(path)

    if "character holding a weapon" in result:
        dest = os.path.join(BASE_DIR, "Characters")
        shutil.move(path, os.path.join(dest, file))
        print(f"Moved to Characters: {file}")

# ===== DEFENSE CLEANUP =====
defense_dir = os.path.join(BASE_DIR, "Defense")

for file in os.listdir(defense_dir):
    path = os.path.join(defense_dir, file)

    result = classify(path)

    if "person or character" in result:
        dest = os.path.join(BASE_DIR, "Characters")
        shutil.move(path, os.path.join(dest, file))
        print(f"Moved to Characters: {file}")