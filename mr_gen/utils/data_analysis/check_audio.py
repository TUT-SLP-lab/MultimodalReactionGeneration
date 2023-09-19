import os
import argparse
from dfcon import Directory
from dfcon.path_filter import Filter, FileFilter
import wave
from tqdm import tqdm

CH = 1
WIDTH = 2  # byte
FR = 16000


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)


def get_args():
    parser = argparse.ArgumentParser()
    set_args(parser)

    args = parser.parse_args()

    return args


class AudioFilter(Filter):
    def __call__(self, target: str) -> dict:
        try:
            with wave.open(target, "rb") as wr:
                ch = wr.getnchannels()
                width = wr.getsampwidth()
                fr = wr.getframerate()

                status = {"channel": ch, "bit-size": width * 8, "sampling-rate": fr}

                if ch != CH:
                    return status
                if WIDTH != width:
                    return status
                if fr % FR != 0:
                    return status

                return {}
        except wave.Error:
            return {"format error"}


def collect_wav_path(target: str, **_kwargs):
    ff = FileFilter().include_extention("wav")
    dirc = Directory(target).build_structure(ff)
    wav_list = dirc.get_file_path(serialize=True)

    af = AudioFilter()
    corrects = []
    incorrects = []

    for wav in tqdm(wav_list):
        res = af(wav)
        if res == {}:
            corrects.append(wav)
        else:
            incorrects.append((wav, res))

    return {"correct": corrects, "incorrect": incorrects}


if __name__ == "__main__":
    _args = get_args()
    result = collect_wav_path(_args.target)

    correct_path = os.path.join(_args.output, "corrects.log")
    incorrect_path = os.path.join(_args.output, "incorrects.log")

    with open(correct_path, "w", encoding="utf-8") as f:
        for correct_wav in result["correct"]:
            f.write(correct_wav + "\n")
    with open(incorrect_path, "w", encoding="utf-8") as f:
        for incorrect_wav, stat in result["incorrect"]:
            f.write(incorrect_wav + " " + str(stat) + "\n")
