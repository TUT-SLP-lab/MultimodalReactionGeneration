import os
import json
from tqdm import tqdm


FPS = 25.0
IN_PATH = "./data/temp/multimodal_dialogue_features_20231107185420"


correct_duration = 0
flg = False

for fpath in tqdm(os.listdir(IN_PATH), position=0, desc="patching"):
    if not fpath.endswith(".json"):
        continue
    fpath = os.path.join(IN_PATH, fpath)

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not flg:
        correct_duration = data["audio"]["end"] - data["audio"]["start"]
        flg = True
    else:
        duration = data["audio"]["end"] - data["audio"]["start"]
        if duration != correct_duration:
            print(fpath)
            print(duration, correct_duration)
            print("")
