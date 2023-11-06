import os
import pickle
from typing import List
import numpy as np
from dfcon import Directory, FileFilter
from tqdm import tqdm
from matplotlib import pyplot as plt
from mr_gen.utils.tools.adapter import FaceAdapter


# IN_PATH = "./data/multimodal_dialogue_features"
IN_PATH = "./data/multimodal_dialogue_features"
RES_PARH = "./mr_gen/scripts/anomaly_search_result.json"

ffilter = FileFilter().include_extention("head")
dirc = Directory(IN_PATH).build_structure(ffilter)
file_path_list = dirc.get_file_path(serialize=True)

if os.path.isfile(RES_PARH):
    os.remove(RES_PARH)

for fpath in tqdm(file_path_list, "search anomaly ..."):
    with open(fpath, "rb") as f:
        data: List[FaceAdapter] = pickle.load(f)
    with open(RES_PARH, "a", encoding="utf-8") as f:
        angle_seq = []
        for d in data:
            if d is not None:
                angle_seq = [data[0].angle]
                break

        jdic = {"path": fpath, "count": 0, "anomaly": []}
        for i, d in enumerate(data):
            if d is None:
                continue
            angle_seq.append(d.angle)
        angle_seq = np.stack(angle_seq)
        dist_seq: np.ndarray = np.linalg.norm(angle_seq[1:] - angle_seq[:-1], axis=1)

        dist_mean = np.mean(dist_seq)
        dist_std = np.std(dist_seq)

        print(fpath)
        for i, d in enumerate(dist_seq):
            if d > 8:
                print(i, end=", ")
        print()

        plt.hist(dist_seq, bins=50)
        if os.path.basename(fpath) == "host.head":
            plt.ylim(0, 100)
            plt.savefig(f"./mr_gen/scripts/{os.path.basename(fpath)}.png")
            plt.clf()

            input()
