import argparse
import os
from typing import Dict, Union, List
import pickle
import numpy as np
from tqdm import tqdm

from mr_gen.utils.io import ZERO_PADDING
from mr_gen.utils.tools import FaceAdapter

PATH = "/home/mikawa/lab/MultimodalReactionGeneration/data/multimodal_dialogue_features"
OUTPUT_PATH = "/home/mikawa/lab/MultimodalReactionGeneration2/data/movements"

FIX_SIZE = 10
MIN_DATA_SIZE = 100


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    return parser.parse_args()


def collect_path(root):
    path_list = {}
    # multimodal_dialogue_features/dataXXX
    for dirs_1st in tqdm(os.listdir(root), desc="collect path"):
        data_path = os.path.join(root, dirs_1st)
        children = []
        # multimodal_dialogue_features/dataXXX/{comp or host}
        for dirs_2nd in os.listdir(data_path):
            dirs_2nd_path = os.path.join(data_path, dirs_2nd)
            if os.path.isdir(dirs_2nd_path):
                if dirs_2nd == "comp" or dirs_2nd == "host":
                    children.append(dirs_2nd)
        path_list[dirs_1st] = children
    return path_list


def collect_head_seq(
    comp_host_path, p_type
) -> Dict[str, Union[np.ndarray, List[Union[np.ndarray, None]]]]:
    index = 0
    angle_centroid = {
        "angle": [],
        "centroid": [],
        "angle_mean": [],
        "angle_std": [],
        "centroid_mean": [],
        "centroid_std": [],
    }
    while True:
        file_name = p_type + "_" + str(index).zfill(ZERO_PADDING) + ".head"
        file_path = os.path.join(comp_host_path, file_name)
        if not os.path.isfile(file_path):
            break

        # initialize
        angle_mean = None
        angle_std = None
        centroid_mean = None
        centroid_std = None

        with open(file_path, "rb") as f:
            _, head_pose = pickle.load(f)
            head_pose: FaceAdapter

            if head_pose is not None:
                # statistics
                angle_mean = head_pose.angle_mean
                centroid_mean = head_pose.centroid_mean
                angle_std = head_pose.angle_std
                centroid_std = head_pose.centroid_std

                # standardization
                angle = (head_pose.angle - angle_mean) / angle_std
                centroid = (head_pose.centroid - centroid_mean) / centroid_std
            else:
                angle = None
                centroid = None

            # record data
            angle_centroid["angle"].append(angle)
            angle_centroid["centroid"].append(centroid)
            angle_centroid["angle_mean"] = angle_mean
            angle_centroid["angle_std"] = angle_std
            angle_centroid["centroid_mean"] = centroid_mean
            angle_centroid["centroid_std"] = centroid_std

        index += 1

    return angle_centroid


def fix_data(angle_centroid):
    angle = angle_centroid["angle"]
    centroid = angle_centroid["centroid"]

    none_count = 0
    for i, (a, c) in enumerate(zip(angle, centroid)):
        if (a is None and c is not None) or (a is not None and c is None):
            raise ValueError("Invalid data")
        if a is None or c is None:
            none_count += 1
            continue
        if i - none_count == 0:
            none_count = 0
            continue
        if none_count == 0:
            continue
        if none_count < FIX_SIZE:
            dif_a = a - angle[i - none_count - 1]
            dif_c = c - centroid[i - none_count - 1]

            divider = np.linspace(0, 1, 1 + none_count + 1)[1:-1]

            ip_a = np.dot(divider.reshape((-1, 1)), dif_a.reshape((1, -1))) + a
            ip_c = np.dot(divider.reshape((-1, 1)), dif_c.reshape((1, -1))) + c

            angle[i - none_count : i] = [a for a in ip_a]
            centroid[i - none_count : i] = [c for c in ip_c]
        none_count = 0

    angle_centroid["angle"] = angle
    angle_centroid["centroid"] = centroid

    return angle_centroid


def output_npz(base_output_name, angle_centroid):
    start = 0
    stop = 1
    max_idx = len(angle_centroid["angle"]) - 1
    for i, element in enumerate(angle_centroid["angle"]):
        if element is None or i == max_idx:
            if stop - start < MIN_DATA_SIZE:
                start = i + 1
                continue

            start_idx = str(start).zfill(ZERO_PADDING)
            stop_idx = str(stop).zfill(ZERO_PADDING)
            save_name = base_output_name + "_" + start_idx + "_" + stop_idx + ".npz"
            angle = np.stack(angle_centroid["angle"][start:stop])
            centroid = np.stack(angle_centroid["centroid"][start:stop])

            np.savez(
                save_name,
                angle=angle,
                centroid=centroid,
                angle_mean=angle_centroid["angle_mean"],
                angle_std=angle_centroid["angle_std"],
                centroid_mean=angle_centroid["centroid_mean"],
                centroid_std=angle_centroid["centroid_std"],
                section=np.array([start, stop]),
            )

            start = i + 1
            continue
        stop = i + 1


def copy_wav(original_path, output_path):
    original_path = original_path + ".wav"
    output_path = output_path + ".wav"
    os.system("cp {} {}".format(original_path, output_path))


def collect_angle_cent(root, out_root, path_list):
    if not os.path.isdir(out_root):
        os.makedirs(out_root, exist_ok=True)

    for dirs_1st in tqdm(path_list, desc="collect angle & centroid"):
        if not os.path.isdir(os.path.join(out_root, dirs_1st)):
            os.makedirs(os.path.join(out_root, dirs_1st), exist_ok=True)

        data_path = os.path.join(root, dirs_1st)

        for dirs_2nd in path_list[dirs_1st]:
            comp_host_path = os.path.join(data_path, dirs_2nd)
            angle_centroid = collect_head_seq(comp_host_path, dirs_2nd)
            angle_centroid = fix_data(angle_centroid)

            output_base_path = os.path.join(out_root, dirs_1st, dirs_2nd)
            output_npz(output_base_path, angle_centroid)

            copy_wav(comp_host_path, output_base_path)


if __name__ == "__main__":
    args = get_args()
    _path_list = collect_path(args.path)
    collect_angle_cent(args.path, args.output, _path_list)
