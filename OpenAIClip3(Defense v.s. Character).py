import os
import shutil
import torch
import clip
from PIL import Image

BASE_DIR = r"C:/Users/dierk/OneDrive/Blast_Attack"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 🔥 Two separate classifiers
person_labels = ["a person", "no person"]
gear_labels = ["armor or protective gear", "not armor"]

person_text = clip.tokenize(person_labels).to(device)
gear_text = clip.tokenize(gear_labels).to(device)

def classify_dual(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            # Person detection
            logits_person, _ = model(image, person_text)
            person_probs = logits_person.softmax(dim=-1).cpu().numpy()[0]

            # Gear detection
            logits_gear, _ = model(image, gear_text)
            gear_probs = logits_gear.softmax(dim=-1).cpu().numpy()[0]

        has_person = person_probs[0] > person_probs[1]
        is_gear = gear_probs[0] > gear_probs[1]

        return has_person, is_gear

    except:
        return False, False

defense_dir = os.path.join(BASE_DIR, "Defense")
character_dir = os.path.join(BASE_DIR, "Characters")

for file in os.listdir(defense_dir):
    path = os.path.join(defense_dir, file)

    if not os.path.isfile(path):
        continue

    has_person, is_gear = classify_dual(path)

    # 🔥 KEY LOGIC
    if has_person:
        # If a person is present → it's a character (even if armored)
        shutil.move(path, os.path.join(character_dir, file))
        print(f"Moved to Characters: {file}")

print("\n✅ Defense cleanup complete.")