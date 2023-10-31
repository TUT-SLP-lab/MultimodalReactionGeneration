import os
import re
import argparse
import numpy as np
from tqdm import tqdm
from mr_gen.utils import open_video, mp4_collector, parallel_luncher
from mr_gen.utils.video_analysis import HeadPoseEstimation


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--procnum", type=int, default=1)
    parser.add_argument("--estimation-fps", type=float, default=25.0)
    parser.add_argument("--overwrite", action="store_true", default=False)


def get_args():
    parser = argparse.ArgumentParser()
    set_args(parser)

    args = parser.parse_args()

    return args


def _video_div(trg_video_path: str, use_tq: bool = False) -> None:
    video_dir_path = os.path.dirname(trg_video_path)
    host_video_path = os.path.join(video_dir_path, "host.mp4")
    comp_video_path = os.path.join(video_dir_path, "comp.mp4")

    with open_video(trg_video_path, mode="r") as vr:
        width = vr.get_cap_width()
        half = width // 2
        fps = vr.get_fps()

        vw_comp = open_video(comp_video_path, mode="w", fps=fps)
        vw_host = open_video(host_video_path, mode="w", fps=fps)

        iterator = tqdm(vr, leave=False) if use_tq else vr

        for frame in iterator:
            comp_frame, *_, host_frame = np.split(frame, [half, -half], axis=1)
            vw_comp.write(comp_frame)
            vw_host.write(host_frame)

        vw_comp.close()
        vw_host.close()


def video_analysis(
    target: str,
    output: str,
    procnum: int = 1,
    est_fps: float = 12.5,
    overwrite: bool = False,
    **_
):
    if not os.path.isdir(output):
        os.mkdir(output)

    trg_list = mp4_collector(target_path=target, name="movie")
    trg_list = [(path, i % procnum == 0) for i, path in enumerate(trg_list)]
    parallel_luncher(
        job=_video_div, arg_list=trg_list, pnum=procnum, unpack=True, pn="Div Video"
    )

    # apply facemesh for video
    video_path_list = mp4_collector(target_path=target, name=["host", "comp"])
    io_path_list = []
    for video_path in video_path_list:
        output_path = re.sub(target, output, video_path)
        output_path = output_path.rsplit(".", maxsplit=1)[0] + ".head"
        io_path_list.append((video_path, output_path))

        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

    estimater = HeadPoseEstimation(estimate_fps=est_fps, redo=overwrite)
    output_path_list = estimater(io_path_list, pnum=procnum, visualize="sample")

    return output_path_list


if __name__ == "__main__":
    _args = get_args()
    video_analysis(**vars(_args))
