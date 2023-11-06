import os
import pickle
from tqdm import tqdm
from dfcon import Directory, FileFilter


FPS = 25.0
IN_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features"
OUT_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features_"

ffilter = FileFilter().include_extention("head")
dirc = Directory(IN_PATH).build_structure(ffilter)
file_path_list = dirc.get_file_path(serialize=True)

for fpath in tqdm(file_path_list, position=0, desc="patching"):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    new_data = []
    for i, d in enumerate(
        tqdm(data, position=1, desc=os.path.basename(fpath), leave=False)
    ):
        if d is None:
            continue
        d.time = i / FPS
        d.frame_no = i
        d.fps = FPS
        new_data.append(d)
    output_path = fpath.replace(IN_PATH, OUT_PATH)
    with open(output_path, "wb") as f:
        pickle.dump(new_data, f)
