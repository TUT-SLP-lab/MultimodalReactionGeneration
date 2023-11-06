import os
import pickle
import shutil
from tqdm import tqdm
from dfcon import Directory, FileFilter

from mr_gen.utils.io import ZERO_PADDING


FPS = 25.0
IN_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features"
OUT_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features_"

ffilter = FileFilter().include_extention(["head", "wav"])
dirc = Directory(IN_PATH).build_structure(ffilter)
file_path_list = dirc.get_file_path(serialize=True)

for fpath in tqdm(file_path_list, position=0, desc="patching"):
    # make directory if not exist
    if not os.path.isdir(os.path.dirname(fpath.replace(IN_PATH, OUT_PATH))):
        os.makedirs(os.path.dirname(fpath.replace(IN_PATH, OUT_PATH)))

    # copy wav file
    if fpath.endswith("wav"):
        shutil.copy2(fpath, fpath.replace(IN_PATH, OUT_PATH))
        continue

    # make directory for head file
    base_path, base_file = os.path.split(fpath.replace(IN_PATH, OUT_PATH))
    base_name, ext = base_file.rsplit(".", maxsplit=1)
    head_dir_path = os.path.join(base_path, base_name)
    if not os.path.isdir(head_dir_path):
        os.mkdir(head_dir_path)

    with open(fpath, "rb") as f:
        data = pickle.load(f)
    for i, d in enumerate(
        tqdm(data, position=1, desc=os.path.basename(fpath), leave=False)
    ):
        output_path = os.path.join(
            head_dir_path, base_name + "_" + str(i).zfill(ZERO_PADDING) + "." + ext
        )
        with open(output_path, "wb") as f:
            pickle.dump((i, d), f)
