import os
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from mr_gen.utils import open_video
from mr_gen.utils.preprocess.motion import MotionPreprocessor
from mr_gen.utils.preprocess.audio import AudioPreprocessor
from mr_gen.utils.tools.adapter import FaceAdapter
from mr_gen.utils.video_analysis import HeadPoseEstimation
from mr_gen.utils.visualize.calc_dataset_statistics import calc_face_statistics


CENTROID_STD = torch.tensor([0.0, 0.0, 0.0])
ANGLE_STD = torch.tensor([0.0, 0.0, 0.0])
VISUALIZE_PATH = "./data/visualize"


def gen_head_motion(
    movie_path: str,
    audio_path: str,
    output_path: str,
    data_conf: DictConfig,
    audio_conf: DictConfig,
) -> None:
    """Generate head motion from movie and audio.

    Args:
        movie_path (str): movie path
        audio_path (str): audio path
        output_path (str): output path
    """
    base_name = os.path.split(os.path.dirname(movie_path))[1]
    file_base_name = os.path.splitext(os.path.basename(movie_path))[0]
    head_out_path = os.path.join(
        VISUALIZE_PATH, base_name, file_base_name, file_base_name + ".head"
    )  # ex. ./data/visualize/data004/comp/comp.head
    head_dir = os.path.dirname(head_out_path)

    # frame length
    with open_video(movie_path, "r") as f:
        frame_length = len(f)

    # analyze head pose
    estimater = HeadPoseEstimation()
    estimater([(movie_path, head_out_path)])

    # preprocess head motion
    motion_preprocessor = MotionPreprocessor(data_conf)
    head_seq = motion_preprocessor(head_dir, 0, frame_length, data_conf.context_stride)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig = None):
    gen_head_motion(
        cfg.movie_path, cfg.audio_path, cfg.output_path, cfg.data, cfg.audio
    )


if __name__ == "__main__":
    main()
