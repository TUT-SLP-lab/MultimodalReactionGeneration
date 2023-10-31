"""Collection of I/O utilities."""
import wave
import pickle
from typing import List, Optional
from numpy import ndarray
import numpy as np

from mr_gen.utils.tools import FaceAdapter


def write_head_pose(path: str, result: List[Optional[FaceAdapter]]) -> None:
    """Write head pose result to file.

    Args:
        path (str): output path
        result (List[Optional[FaceAdapter]]): head pose result
    """
    if not isinstance(result, list):
        raise TypeError("result must be list.")
    if not isinstance(result[0], (FaceAdapter, type(None))):
        raise TypeError("result must be list of FaceAdapter or NoneType.")

    with open(path, "wb") as f:
        pickle.dump(result, f)


N_BYTE = 2
SCALE = 1.0 / float(1 << ((8 * N_BYTE) - 1))


def load_wav(filepath: str) -> ndarray:
    with wave.open(filepath, "r") as f:
        wav_buffer = f.readframes(f.getnframes())
        data = np.frombuffer(wav_buffer, dtype=np.int16).astype(np.float32)

    wav = data * SCALE

    return wav
