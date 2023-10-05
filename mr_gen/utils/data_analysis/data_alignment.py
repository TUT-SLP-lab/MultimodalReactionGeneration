import os
import argparse
from typing import List, Tuple
import wave
import shutil
import audioread as ar
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from joblib import Parallel, delayed

from mr_gen.utils.video import Video


N_BYTE = 2
SCALE = 1.0 / float(1 << ((8 * N_BYTE) - 1))

CH = 1
WIDTH = 2
FS = 16000
CTYPE = "NONE"
CNAME = "not compressed"

WINDOW_SIZE = 5  # [s]
STD_SP = 1e-4  # [Pa] standard sound pressure


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)


def get_args():
    parser = argparse.ArgumentParser()
    set_args(parser)

    args = parser.parse_args()

    return args


def load_m4a(filepath: str) -> ndarray:
    with ar.audio_open(filepath) as f:
        frame_list = [
            SCALE * np.frombuffer(frame, f"<i{N_BYTE}").astype(np.float32)
            for frame in f
        ]
        wav = np.hstack(frame_list)
        wav = wav[::2]  # 32k -> 16k Down Sampling

    return wav


def load_wav(filepath: str) -> ndarray:
    with wave.open(filepath, "r") as f:
        wav_buffer = f.readframes(f.getnframes())
        data = np.frombuffer(wav_buffer, dtype=np.int16).astype(np.float32)

    wav = data * SCALE

    return wav


def shaping_grid(grid_size, target0: ndarray) -> ndarray:
    rest_length0 = len(target0) % grid_size
    rest_length0 = None if rest_length0 == 0 else -rest_length0
    grid0 = target0.copy()[:rest_length0].reshape((-1, grid_size))

    return grid0


def make_grid_single(grid_size, target0: ndarray) -> ndarray:
    grid0 = shaping_grid(grid_size, target0)

    grid0[np.abs(grid0) < 0.1] = STD_SP
    grid0: ndarray = np.sqrt(np.mean(np.square(grid0), axis=-1))
    grid0 = 20 * np.log10(grid0 / STD_SP)
    grid0 /= np.mean(grid0)

    return grid0


def make_grid(grid_size, target0: ndarray, target1: ndarray) -> Tuple[ndarray, ndarray]:
    return (
        make_grid_single(grid_size, target0),
        make_grid_single(grid_size, target1),
    )


def grid_matching(
    target0: ndarray,
    target1: ndarray,
    div: List[int],
    use_tqdm=False,
) -> Tuple[int, Tuple[int, int]]:
    assert div[1] > div[0]

    # First Grid Process ########################################
    grid_size = FS // div[0]

    grid0, grid1 = make_grid(grid_size, target0, target1)

    mrgn = len(grid1) // 3

    obj = grid1[mrgn:-mrgn]
    trg = np.zeros((2 * (len(obj) - 1) + len(grid0)))
    trg[len(obj) - 1 : len(grid0) + (len(obj) - 1)] = grid0
    res = []

    if use_tqdm:
        iterator = tqdm(range(len(grid0) + (len(obj) - 1)), desc="1st Grid")
    else:
        iterator = range(len(grid0) + (len(obj) - 1))

    for t in iterator:
        res.append(np.dot(trg[t : t + len(obj)], obj))
    shift = np.argmax(res) - mrgn - (len(obj) - 1)

    if shift >= 0:
        new_trg0 = target0.copy()
        new_trg1 = np.zeros((shift * grid_size + len(target1)))
        new_trg1[shift * grid_size :] = target1.copy()
    else:
        new_trg0 = target0.copy()
        new_trg1 = target1[abs(shift) * grid_size :].copy()

    first_shift = shift * grid_size
    #############################################################

    # Second Grid Process #######################################
    grid_size = FS // div[1]
    prob = int(1.5 * (FS // div[0]) / grid_size)
    w_size = WINDOW_SIZE * div[1]

    grid0, grid1 = make_grid(grid_size, new_trg0, new_trg1)

    mrgn = len(grid1) // 3

    obj = grid1[mrgn:-mrgn]
    trg = grid0[mrgn - prob : mrgn + prob + (len(obj) - 1)]
    res = []

    if use_tqdm:
        iterator = tqdm(range(2 * prob), desc="2nd Grid")
    else:
        iterator = range(2 * prob)

    for t in iterator:
        res.append(np.dot(trg[t : t + len(obj)], obj))
    shift = np.argmax(res) - prob

    second_shift = shift * grid_size
    total_shift = first_shift + second_shift

    if total_shift >= 0:
        new_trg0 = target0.copy()
        new_trg1 = np.zeros((total_shift + len(target1)))
        new_trg1[total_shift:] = target1.copy()
    else:
        new_trg0 = target0.copy()
        new_trg1 = target1[abs(total_shift) :].copy()

    #############################################################

    area_start = max(total_shift, 0)
    candidate_area = new_trg1[area_start : len(new_trg0)].copy()

    windows = shaping_grid(w_size, candidate_area)
    windows[np.abs(windows) < 0.1] = 0
    act_idx = np.argmax(np.sum(np.abs(windows) > 0, axis=-1))
    act_idx = act_idx * w_size + abs(min(total_shift, 0))

    return (total_shift, (act_idx, int(act_idx + WINDOW_SIZE * FS)))


def get_time_shift(target0: ndarray, target1: ndarray, use_tqdm=False) -> int:
    div = [10, 100]
    shift, (obj_st, obj_ed) = grid_matching(target0, target1, div, use_tqdm)

    trg_st = obj_st + shift
    trg_ed = obj_ed + shift
    prob = int(FS // div[1] * 10)

    obj = target1[obj_st:obj_ed].copy()
    trg = target0[trg_st - prob : trg_ed + prob].copy()

    obj /= max(obj)
    trg /= max(trg)

    res = []

    if use_tqdm:
        iterator = tqdm(range(len(trg) - len(obj)))
    else:
        iterator = range(len(trg) - len(obj))

    for t in iterator:
        res.append(np.dot(trg[t : t + len(obj)], obj))
    shift += np.argmax(res) - prob

    return shift


def alignment(
    target0: str,
    target1: str,
    target2: str,
    mp4: str,
    out_site: str,
    use_tqdm=False,
) -> List[Tuple[ndarray, int, int]]:
    """alignment audio files.

    Args:
        target0 (str): m4a file path.
        target1 (str): wav file which for person A path.
        target2 (str): wav file which for person B path.

    """

    org_wav0 = load_m4a(target0)
    org_wav1 = load_wav(target1)
    org_wav2 = load_wav(target2)

    shift_width1 = get_time_shift(org_wav0, org_wav1, use_tqdm)
    shift_width2 = get_time_shift(org_wav0, org_wav2, use_tqdm)

    std_f = max(0, shift_width1, shift_width2)

    wav0_cut_f = abs(0 - std_f)
    wav1_cut_f = abs(shift_width1 - std_f)
    wav2_cut_f = abs(shift_width2 - std_f)

    std_b = min(
        len(org_wav0) - wav0_cut_f,
        len(org_wav1) - wav1_cut_f,
        len(org_wav2) - wav2_cut_f,
    )

    start_time = wav0_cut_f / FS
    stop_time = (wav0_cut_f + std_b) / FS
    video = Video(mp4, codec="mp4v")

    formed_wav0 = org_wav0[wav0_cut_f : wav0_cut_f + std_b]
    formed_wav1 = org_wav1[wav1_cut_f : wav1_cut_f + std_b]
    formed_wav2 = org_wav2[wav2_cut_f : wav2_cut_f + std_b]

    wav_target0 = target0[:-3] + "wav"

    out_mp4 = os.path.join(out_site, os.path.basename(mp4))
    out_wav0 = os.path.join(out_site, os.path.basename(wav_target0))
    out_wav1 = os.path.join(out_site, os.path.basename(target1))
    out_wav2 = os.path.join(out_site, os.path.basename(target2))

    video.trime_time(start_time, stop_time, out_mp4, use_tqdm)
    output_wav(out_wav0, formed_wav0)
    output_wav(out_wav1, formed_wav1)
    output_wav(out_wav2, formed_wav2)


def output_wav(path: str, wav: ndarray):
    wav = (wav / SCALE).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setframerate(FS)
        wf.setnchannels(CH)
        wf.setsampwidth(WIDTH)

        wf.writeframes(wav.tobytes())


def process():
    _args = get_args()

    if os.path.isdir(_args.output):
        if _args.overwrite:
            shutil.rmtree(_args.output)
            os.mkdir(_args.output)
        else:
            raise ValueError("output path directory is already exists.")
    else:
        os.mkdir(_args.output)

    job_num = 8
    arg_set = []

    total_times = 0.0

    dircs: list = os.listdir(_args.target)
    dircs.sort()

    for dirc in tqdm(dircs, desc="initialize output site"):
        dpath = os.path.join(_args.target, dirc)
        out_dpath = os.path.join(_args.output, dirc)
        arg_dict = {
            "target0": None,
            "target1": None,
            "target2": None,
            "mp4": None,
            "out_site": out_dpath,
            "use_tqdm": len(arg_set) % job_num == 0,
        }
        arg_set.append(arg_dict)

        os.mkdir(out_dpath)

        for mem in os.listdir(dpath):
            fpath = os.path.join(dpath, mem)

            if mem[-3:] == "m4a":
                arg_set[-1]["target0"] = fpath
            elif mem[-3:] == "wav":
                if arg_set[-1]["target1"] is None:
                    arg_set[-1]["target1"] = fpath
                    dt = load_wav(fpath)
                    total_times += len(dt) / FS
                else:
                    arg_set[-1]["target2"] = fpath
            elif mem[-3:] == "mp4":
                arg_set[-1]["mp4"] = fpath
            else:
                print("warn :", fpath)

    print("Data Amount (before Time-Shift-Modification)")
    print(round(total_times, 2), "[s]")
    print(round(total_times / 3600, 2), "[h]")

    for step in range(0, len(arg_set), job_num):
        print(step, "/", len(arg_set))
        Parallel(n_jobs=job_num, verbose=0)(
            [delayed(alignment)(**arg_set[step + i]) for i in range(job_num)]
        )

    total_times = 0
    for dirc in tqdm(os.listdir(_args.target), desc="calc total time"):
        dpath = os.path.join(_args.target, dirc)
        for mem in os.listdir(dpath):
            fpath = os.path.join(dpath, mem)

            if mem[-3:] == "wav":
                dt = load_wav(fpath)
                total_times += len(dt) / FS
                break
            else:
                continue

    print("Data Amount (before Time-Shift-Modification)")
    print(round(total_times, 2), "[s]")
    print(round(total_times / 3600, 2), "[h]")


if __name__ == "__main__":
    process()
