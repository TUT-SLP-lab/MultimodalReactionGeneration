import os
import pickle
import json
from dfcon import Directory, FileFilter
from tqdm import tqdm


# IN_PATH = "./data/multimodal_dialogue_features"
IN_PATH = "./data/multimodal_dialogue_features"
RES_PARH = "./data/anomaly_search_result.json"

ffilter = FileFilter().include_extention("head")
dirc = Directory(IN_PATH).build_structure(ffilter)
file_path_list = dirc.get_file_path(serialize=True)

if os.path.isfile(RES_PARH):
    os.remove(RES_PARH)

for fpath in tqdm(file_path_list, "search anomaly ..."):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    with open(RES_PARH, "a", encoding="utf-8") as f:
        jdic = {"path": fpath, "count": 0, "anomaly": []}
        for i, d in enumerate(data):
            if d is None:
                jdic["count"] += 1
                jdic["anomaly"].append(i)
        json_line = json.dumps(jdic, ensure_ascii=False)
        f.write(json_line + "\n")
