import os
from dfcon import Directory
from dfcon.path_filter import Filter, FileFilter
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    proc_args = get_args()

    m4a_filter = FileFilter().include_extention("m4a")
    mp3_filter = FileFilter().include_extention("mp3")
    dirc = Directory(proc_args.target).build_structure()

    # .m4a -> .mp3
    for filename in dirc.get_file_path(m4a_filter, serialize=True):
        filename = filename.rsplit(".", maxsplit=1)[0]
        os.system(f"ffmpeg -i {filename}.m4a -ab 256k {filename}.mp3")

    # .mp3 -> wav
    for filename in dirc.get_file_path(mp3_filter, serialize=True):
        print(filename)
        filename = filename.rsplit(".", maxsplit=1)[0]
        os.system(f"sox {filename}.mp3 {filename}.wav channels 1 rate 16k")
