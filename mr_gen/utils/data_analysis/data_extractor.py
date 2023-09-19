import os
import re
import shutil
import argparse
from typing import List
import subprocess
from tqdm import tqdm
from dfcon.path_filter import FileFilter

from mr_gen.utils.data_analysis import check_audio


class AudioFilter(check_audio.AudioFilter):
    def __init__(self) -> None:
        super().__init__()
        self.ex_filter = FileFilter().include_extention("wav")

    def __call__(self, target: str) -> bool:
        if not self.ex_filter(target):
            return True

        res = super().__call__(target)

        return res == {}


def set_args(parser: argparse.ArgumentParser):
    check_audio.set_args(parser)
    parser.add_argument("--overwrite", action="store_true", default=False)


def get_args():
    parser = argparse.ArgumentParser()
    set_args(parser)

    args = parser.parse_args()

    return args


def remove_duplication(target: List[str]) -> List[str]:
    _target = [target[0]]
    for t in target[1:]:
        # the same name directory is stored side by side.
        if _target[-1] != t:
            _target.append(t)
        else:
            continue
    return _target


def extraction():
    _args = get_args()

    if os.path.exists(_args.output) and not _args.overwrite:
        raise OSError(f"'{_args.output}' is already exists.")
    elif os.path.exists(_args.output) and _args.overwrite:
        shutil.rmtree(_args.output)
    elif not os.path.exists(_args.output):
        os.mkdir(_args.output)

    subprocess.run(["cp", "-r", _args.target, _args.output], check=True)
    shutil.rmtree(_args.output + "/イレギュラー")

    dirs = check_audio.collect_wav_path(_args.output)
    incorrects = dirs["incorrect"]
    incorrect_dirs = ["/".join(re.split(r"[\\|/]", ic[0])[:-1]) for ic in incorrects]

    incorrect_dirs = remove_duplication(incorrect_dirs)

    for d in tqdm(incorrect_dirs, desc="remove"):
        shutil.rmtree(d)

    correct_dirs: List[str] = os.listdir(_args.output)

    for d in tqdm(correct_dirs, desc="rename"):
        new_dir_name = d.split("_")[0]

        d = _args.output + "/" + d
        new_dir_name = _args.output + "/" + new_dir_name

        os.rename(d, new_dir_name)


if __name__ == "__main__":
    extraction()
