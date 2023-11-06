import os
import pickle
from tqdm import tqdm
from dfcon import Directory

FPS = 25.0
IN_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features"

print("building directory structure...", end="")
dirc = Directory(IN_PATH, empty=True).build_structure()
terminal_dir = dirc.get_terminal_instances(serialize=True)
print("done")

for head_dir in tqdm(terminal_dir, position=0, desc="patching"):
    max_idx = 0
    for head_file in os.listdir(head_dir.path):
        idx = int(head_file.split("_")[1].split(".")[0])
        max_idx = max(max_idx, idx)
    for idx in range(max_idx):
        target_name = os.path.join(head_dir.path, f"{head_dir.name}_{idx:05d}.head")
        if os.path.isfile(target_name):
            continue

        with open(target_name, "wb") as f:
            pickle.dump((idx, None), f)
