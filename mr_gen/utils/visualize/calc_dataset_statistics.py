import os
import pickle
from tqdm import tqdm
import numpy as np

from mr_gen.utils.tools.adapter import FaceAdapter


def calc_face_statistics(dataset_path, **kwargs):
    total_angl_std = np.zeros(3)
    total_cent_std = np.zeros(3)

    member_1st = os.listdir(dataset_path)
    for mem_1st in tqdm(member_1st, desc="calc face statistics", **kwargs):
        if os.path.isfile(os.path.join(dataset_path, mem_1st)):
            continue

        member_2nd = os.listdir(os.path.join(dataset_path, mem_1st))
        for mem_2nd in member_2nd:
            if os.path.isfile(os.path.join(dataset_path, mem_1st, mem_2nd)):
                continue

            member_3rd = os.listdir(os.path.join(dataset_path, mem_1st, mem_2nd))
            for mem_3rd in member_3rd:
                trgt = os.path.isfile(
                    os.path.join(dataset_path, mem_1st, mem_2nd, mem_3rd)
                )

                with open(trgt, "rb") as f:
                    head: FaceAdapter = pickle.load(f)[1]  # (idx, head)

                total_angl_std += head.angle_std
                total_cent_std += head.centroid_std
