import os
import shutil

# Path to your messy folder
SOURCE_DIR = "C:/Users/dierk/Blast_Battle"

# Destination base folder
DEST_DIR = "C:/Users/dierk/OneDrive/Blast_Attack"

# Keyword rules
rules = {
    "Blasters": ["pistol", "rifle", "shotgun", "smg", "sniper", "machine gun"],
    "Defense": ["armor", "helmet", "shield", "vest", "boots"],
    "Characters": ["hero", "villain", "agent", "soldier", "him", "her", "they", "character"],
    "Zones": ["zone", "map", "arena", "range"]
}

def move_file(file, category):
    dest_folder = os.path.join(DEST_DIR, category)
    os.makedirs(dest_folder, exist_ok=True)
    shutil.move(file, os.path.join(dest_folder, os.path.basename(file)))

for filename in os.listdir(SOURCE_DIR):
    file_path = os.path.join(SOURCE_DIR, filename)

    if not os.path.isfile(file_path):
        continue

    lower_name = filename.lower()

    moved = False
    for category, keywords in rules.items():
        if any(keyword in lower_name for keyword in keywords):
            move_file(file_path, category)
            moved = True
            break

    if not moved:
        move_file(file_path, "Unsorted")

print("Sorting complete.")
