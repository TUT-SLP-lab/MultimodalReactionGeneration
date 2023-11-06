import os
import shutil
from tqdm import tqdm
from dfcon import Directory, FileFilter


IN_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_formd"
OUT_PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features"

ffilter = FileFilter().include_extention("wav").contained(["host", "comp"])
dirc = Directory(IN_PATH).build_structure(ffilter)
file_path_list = dirc.get_file_path(serialize=True)

for fpath in tqdm(file_path_list, position=0, desc="patching"):
    copy_path = fpath.replace(IN_PATH, OUT_PATH)
    if os.path.isfile(copy_path):
        continue
    shutil.copy(fpath, copy_path)
