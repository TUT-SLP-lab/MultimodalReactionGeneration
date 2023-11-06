import os
import pickle
import shutil
from typing import List, Optional, Tuple
from tqdm import tqdm
from dfcon import Directory

from mr_gen.utils.tools.adapter import FaceAdapter
from mr_gen.utils.io import compute_statistics


FPS = 25.0
IN_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features"
OUT_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features_"

if not os.path.isdir(OUT_PATH):
    os.makedirs(OUT_PATH)

# ffilter = FileFilter().include_extention(["head", "wav"])
print("building directory structure...", end="")
dirc = Directory(IN_PATH, empty=True).build_structure()
terminal_dir = dirc.get_terminal_instances(serialize=True)
print("done")

for head_dir in tqdm(terminal_dir, position=0, desc="patching"):
    # make directory if not exist
    if not os.path.isdir(head_dir.path.replace(IN_PATH, OUT_PATH)):
        os.makedirs(head_dir.path.replace(IN_PATH, OUT_PATH))

    # copy wav file
    wav_file = head_dir.path + ".wav"
    shutil.copy2(wav_file, wav_file.replace(IN_PATH, OUT_PATH))

    results: List[Optional[FaceAdapter]] = []
    face_path: List[Tuple[Tuple[int, Optional[FaceAdapter]], str]] = []
    for head_file in os.listdir(head_dir.path):
        head_file = head_dir.path + "/" + head_file
        with open(head_file, "rb") as f:
            face_info: Tuple[int, Optional[FaceAdapter]] = pickle.load(f)
            idx, face = face_info
        results.append(face)
        face_path.append((face_info, head_file))

    stats = compute_statistics(results)
    for (idx, face), path in face_path:
        if face is None:
            continue
        face.angle_mean = stats["angle_mean"]
        face.angle_std = stats["angle_std"]
        face.centroid_mean = stats["centroid_mean"]
        face.centroid_std = stats["centroid_std"]

        out_path = path.replace(IN_PATH, OUT_PATH)
        with open(out_path, "wb") as f:
            pickle.dump((idx, face), f)
