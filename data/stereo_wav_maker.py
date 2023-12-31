import argparse
import os
import wave
import numpy as np
from numpy import ndarray
from tqdm import tqdm

# from mr_gen.utils.io import ZERO_PADDING
# from mr_gen.utils.tools import FaceAdapter

PATH = "/home/mikawa/lab/MultimodalReactionGeneration/data/unprocess/multimodal_dialogue_formd"
OUTPUT_PATH = "/home/mikawa/lab/MultimodalReactionGeneration2/data/annotations"

N_BYTE = 2
SCALE = 1.0 / float(1 << ((8 * N_BYTE) - 1))

CH = 2
WIDTH = 2
FS = 16000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    return parser.parse_args()


def load_wav(filepath: str) -> ndarray:
    with wave.open(filepath, "r") as f:
        wav_buffer = f.readframes(f.getnframes())
        data = np.frombuffer(wav_buffer, dtype=np.int16).astype(np.float32)

    wav = data * SCALE

    return wav


def output_wav(path: str, wav1: ndarray, wav2: ndarray):
    wav1 = (wav1 / SCALE).astype(np.int16)
    wav2 = (wav2 / SCALE).astype(np.int16)

    if len(wav1) != len(wav2):
        raise ValueError("wav1 and wav2 must be same length.")
    N = len(wav1)

    with wave.open(path, "wb") as wf:
        wf.setframerate(FS)
        wf.setnchannels(CH)
        wf.setsampwidth(WIDTH)

        stereo = np.array([[wav1[i], wav2[i]] for i in range(int(N))]).flatten()
        wf.writeframes(stereo.tobytes())


def to_stereo(comp_path: str, host_path: str, outpath: str):
    comp_wav = load_wav(comp_path)
    host_wav = load_wav(host_path)
    output_wav(outpath, comp_wav, host_wav)


def copy_movie(movie_path: str, outpath: str):
    os.system("cp {} {}".format(movie_path, outpath))


def collect_path(root):
    path_list = {}
    # multimodal_dialogue_features/dataXXX
    for dirs_1st in tqdm(os.listdir(root), desc="collect path"):
        data_path = os.path.join(root, dirs_1st)
        if not os.path.isdir(data_path):
            continue
        children = []
        # multimodal_dialogue_features/dataXXX/{comp or host}
        for dirs_2nd in os.listdir(data_path):
            dirs_2nd_path = os.path.join(data_path, dirs_2nd)
            if os.path.isfile(dirs_2nd_path):
                if (
                    dirs_2nd == "comp.wav"
                    or dirs_2nd == "host.wav"
                    or dirs_2nd == "movie.mp4"
                ):
                    children.append(dirs_2nd)
        path_list[dirs_1st] = children
    return path_list


def make_annotation_data(root, out_root, path_list):
    if not os.path.isdir(out_root):
        os.makedirs(out_root, exist_ok=True)

    for data_name, children in tqdm(path_list.items(), desc="make annotation data"):
        data_path = os.path.join(out_root, data_name)
        if not os.path.isdir(data_path):
            os.makedirs(data_path, exist_ok=True)

        if len(children) != 3:
            raise ValueError("Invalid data")
        if (
            "comp.wav" not in children
            or "host.wav" not in children
            or "movie.mp4" not in children
        ):
            raise ValueError("Invalid data")

        comp_path = os.path.join(root, data_name, "comp.wav")
        host_path = os.path.join(root, data_name, "host.wav")
        movie_path = os.path.join(root, data_name, "movie.mp4")

        out_stereo_path = os.path.join(data_path, "stereo.wav")
        out_movie_path = os.path.join(data_path, "movie.mp4")

        to_stereo(comp_path, host_path, out_stereo_path)
        copy_movie(movie_path, out_movie_path)


if __name__ == "__main__":
    args = get_args()
    _path_list = collect_path(args.path)
    make_annotation_data(args.path, args.output, _path_list)
