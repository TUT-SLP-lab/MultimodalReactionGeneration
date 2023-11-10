"""Collection of I/O utilities."""
import os
import wave
import pickle
from typing import List, Optional, Dict, Tuple
from numpy import ndarray
import numpy as np

from mr_gen.utils.tools import FaceAdapter

ZERO_PADDING = 5


def compute_statistics(
    result: List[Tuple[int, Optional[FaceAdapter]]]
) -> Dict[str, ndarray]:
    # compute mean
    angle_sum = np.zeros(3)
    centroid_sum = np.zeros(3)
    for _, face in result:
        if face is None:
            continue
        angle_sum += face.angle
        centroid_sum += face.centroid
    angle_mean = angle_sum / len(result)
    centroid_mean = centroid_sum / len(result)

    # compute std
    angle_err = np.zeros(3)
    centroid_err = np.zeros(3)
    for _, face in result:
        if face is None:
            continue
        angle_err += (face.angle - angle_mean) ** 2
        centroid_err += (face.centroid - centroid_mean) ** 2
    angle_std = np.sqrt(angle_err / len(result))
    centroid_std = np.sqrt(centroid_err / len(result))

    return {
        "angle_mean": angle_mean,
        "angle_std": angle_std,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
    }


def write_head_pose(path: str, result: List[Tuple[int, Optional[FaceAdapter]]]) -> None:
    """Write head pose result to file.

    Args:
        path (str): output path
        result (List[Optional[FaceAdapter]]): head pose result
    """
    # check type
    if not isinstance(result, list):
        raise TypeError("result must be list.")
    if not isinstance(result[0][1], (FaceAdapter, type(None))):
        raise TypeError("result must be list of index & FaceAdapter or NoneType.")

    # make directory if not exist
    if os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # compute statistics
    statis = compute_statistics(result)

    # save as pickle each FaceAdapter
    for i, a_face in result:
        base_path, ext = path.rsplit(".", maxsplit=1)
        output_path = base_path + "_" + str(i).zfill(ZERO_PADDING) + "." + ext

        if a_face is not None:
            a_face.angle_mean = statis["angle_mean"]
            a_face.angle_std = statis["angle_std"]
            a_face.centroid_mean = statis["centroid_mean"]
            a_face.centroid_std = statis["centroid_std"]

        with open(output_path, "wb") as f:
            pickle.dump((i, a_face), f)


N_BYTE = 2
SCALE = 1.0 / float(1 << ((8 * N_BYTE) - 1))


def load_wav(filepath: str) -> ndarray:
    with wave.open(filepath, "r") as f:
        wav_buffer = f.readframes(f.getnframes())
        data = np.frombuffer(wav_buffer, dtype=np.int16).astype(np.float32)

    wav = data * SCALE

    return wav
