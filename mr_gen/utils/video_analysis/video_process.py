import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from mr_gen.utils import open_video, mp4_collector, parallel_luncher
from mr_gen.utils.video_analysis import HeadPoseEstimation


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pnum", type=int, default=1)
    parser.add_argument("--est-fps", type=float, default=25.0)
    parser.add_argument("--redo", action="store_true", default=False)


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

        if use_tq:
            vr = tqdm(vr, leave=False, position=1)

        for frame in vr:
            comp_frame, *_, host_frame = np.split(frame, [half, -half], axis=1)
            vw_comp.write(comp_frame)
            vw_host.write(host_frame)

        vw_comp.close()
        vw_host.close()


def video_analysis(
    target: str, output: str, pnum=1, est_fps=12.5, redo=False, tq_pos=0, **_
):
    if not os.path.isdir(output):
        os.mkdir(output)

    trg_list = mp4_collector(target_path=target, name="movie")
    trg_list = [(path, i % pnum == 0) for i, path in enumerate(trg_list)]
    parallel_luncher(
        _video_div, trg_list, pnum, unpack=True, desc="Div video", position=tq_pos
    )

    # apply facemesh for video
    video_path_list = mp4_collector(target_path=target, name=["host", "comp"])
    io_path_list = []
    for video_path in tqdm(video_path_list, desc="Build ...", position=tq_pos):
        out_base_name = os.path.basename(video_path).rsplit(".", maxsplit=1)[0]
        output_path = os.path.join(output, out_base_name, out_base_name + ".head")
        io_path_list.append((video_path, output_path))

        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        wave_path = video_path.rsplit(".", maxsplit=1)[0] + ".wav"
        out_wave_path = wave_path.replace(target, output)
        if not os.path.isfile(out_wave_path) or redo:
            shutil.copy2(wave_path, out_wave_path)

    estimater = HeadPoseEstimation(estimate_fps=est_fps, redo=redo)
    output_path_list = estimater(
        io_path_list, pnum=pnum, visualize="sample", position=tq_pos, desc="Est Lmark"
    )

    return output_path_list


if __name__ == "__main__":
    _args = get_args()
    video_analysis(**vars(_args))
