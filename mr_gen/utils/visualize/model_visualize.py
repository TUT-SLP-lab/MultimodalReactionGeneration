import os
import pickle
from typing import Tuple
import hydra
import torch
import pytorch_lightning as pl
import soundfile as sf
from omegaconf import DictConfig
from tqdm import tqdm

from mr_gen.utils import open_video
from mr_gen.utils.video import patch_audio
from mr_gen.utils.io import ZERO_PADDING
from mr_gen.model.model_loader import load_model
from mr_gen.utils.visualize import head_pose_plotter
from mr_gen.utils.tools.adapter import FaceAdapter
from mr_gen.utils.video_analysis import HeadPoseEstimation
from mr_gen.utils.preprocess import AudioPreprocessor, MotionPreprocessor

VISUALIZE_PATH = "./data/visualize"


def get_input(
    heads: torch.Tensor,
    fbank: torch.Tensor,
    start: int,
    end: int,
    stride: int,
    trg_start: int,
    trg_end: int,
    trg_stride: int,
    data_conf: DictConfig,
    audio_conf: DictConfig,
) -> Tuple[torch.Tensor, ...]:
    """Get input for LSTM.

    Args:
        heads (torch.Tensor): head motion list
        fbank (torch.Tensor): fbank
        start (int): start frame index
        end (int): end frame index
        data_conf (DictConfig): data config
        audio_conf (DictConfig): audio config

    Returns:
        torch.Tensor: initial input
    """
    # audio params
    shift = audio_conf.shift
    sample_rate = audio_conf.sample_rate

    # movie params
    fps = data_conf.fps

    transformer = lambda x: int(sample_rate / shift / fps * x)

    # calc frame idx
    start_frame_idx = transformer(start)
    end_frame_idx = transformer(end)

    input_fbank = fbank[start_frame_idx:end_frame_idx]
    input_head = heads[start:end:stride]
    output_head = heads[trg_start:trg_end:trg_stride]

    return (input_fbank, input_head, output_head)


def load_face(head_dir) -> FaceAdapter:
    base_name = os.path.split(head_dir)[1]
    idx_str = str(1).zfill(ZERO_PADDING)
    file_name = base_name + "_" + idx_str + ".head"
    target_path = os.path.join(head_dir, file_name)

    with open(target_path, "rb") as f:
        head: FaceAdapter = pickle.load(f)[1]  # (idx, head)

    return head


def cat_audio(
    video_path: str,
    out_path: str,
    audio_path: str,
    start: int,
    stop: int,
    fps: float,
    stride: int,
) -> None:
    # extract audio & data info
    waveforme, sample_rate = sf.read(audio_path)
    transformer = lambda x: int(sample_rate * x / fps)
    start_idx = transformer(start)
    stop_idx = transformer(stop + stride)

    # write audio
    waveforme = waveforme[start_idx:stop_idx]
    wave_out_path = out_path.rsplit(".")[0] + ".wav"
    sf.write(wave_out_path, waveforme, sample_rate)

    # patch audio
    patch_audio(out_path, video_path, wave_out_path)


def gen_head_motion(
    model: str,
    model_path: str,
    model_config: DictConfig,
    movie_path: str,
    audio_path: str,
    output_path: str,
    data_conf: DictConfig,
    audio_conf: DictConfig,
    plot_answer: bool = False,
) -> None:
    """Generate head motion from movie and audio.

    Args:
        movie_path (str): movie path
        audio_path (str): audio path
        output_path (str): output path
    """
    base_name = os.path.split(os.path.dirname(movie_path))[1]
    file_base_name = os.path.splitext(os.path.basename(movie_path))[0]
    target_base_name = "comp" if file_base_name == "host" else "host"
    head_out_path = os.path.join(
        VISUALIZE_PATH, base_name, target_base_name, target_base_name + ".head"
    )  # ex. ./data/visualize/data004/comp/comp.head
    head_dir = os.path.dirname(head_out_path)

    context_stride = data_conf.context_stride
    context_length = data_conf.context_size * context_stride
    context_start = data_conf.context_start
    context_end = context_start + context_length
    target_position = data_conf.target_position
    target_length = data_conf.target_size * data_conf.target_stride

    # frame length
    with open_video(movie_path, "r") as f:
        frame_length = len(f)

    # analyze head pose
    estimater = HeadPoseEstimation()
    estimater([(movie_path, head_out_path)])

    # preprocess head motion
    motion_preprocessor = MotionPreprocessor(data_conf)
    head_seq = motion_preprocessor(head_dir, 0, frame_length, 1)

    # preprocess audio
    audio_preprocessor = AudioPreprocessor(audio_conf)
    fbank = audio_preprocessor(audio_path, 0, -1)

    # load face
    head = load_face(head_dir)
    face = head.face

    # calc statistics
    angle_transformer = lambda x: x * head.angle_std + head.angle_mean
    centroid_transformer = lambda x: x * head.centroid_std + head.centroid_mean

    # load model @ eval mode
    model: pl.LightningModule = load_model(model, model_path, model_config).eval()

    # open video reader / writer
    video_reader = open_video(movie_path, "r")
    video_writer = open_video(output_path, "w", fps=data_conf.fps / context_stride)

    input_head = None
    start = 1e9
    stop = 0

    for i, frame in enumerate(tqdm(video_reader, desc="generate head motion")):
        if i % data_conf.sample_stride != 0:
            continue
        minimun_start = abs(context_start) + data_conf.delta_order * context_stride + 1
        if i < minimun_start:
            continue
        # end condition
        if i + target_position + target_length + 1 > len(head_seq):
            break

        start = min(start, i)
        stop = max(stop, i)

        # get input
        input_fbank, _input_head, output_head = get_input(
            head_seq,
            fbank,
            i + context_start,
            i + context_end,
            context_stride,
            i + target_position,
            i + target_position + 1,
            1,  # always 1
            data_conf,
            audio_conf,
        )

        with torch.no_grad():
            # predict head motion
            input_head = _input_head.unsqueeze(0) if input_head is None else input_head
            input_fbank = input_fbank.unsqueeze(0)
            output_head = output_head.unsqueeze(0)
            # print(input_fbank.shape, input_head.shape, output_head.shape)
            pred_head: torch.Tensor = model(input_fbank, input_head)

        # collect head motion
        to_numpy = lambda x: x.detach().numpy()

        centroid = centroid_transformer(to_numpy(pred_head[0, 0, :3]))
        angle = angle_transformer(to_numpy(pred_head[0, 0, 3:6]))
        info = {"face": face, "centroid": centroid, "angle": angle}

        # plot head motion
        frame = head_pose_plotter(frame, info, (50, 50, 255))

        if plot_answer:
            ans_centroid = centroid_transformer(to_numpy(output_head[0, 0, :3]))
            ans_angle = angle_transformer(to_numpy(output_head[0, 0, 3:6]))
            ans_info = {"face": face, "centroid": ans_centroid, "angle": ans_angle}

            # plot answer
            frame = head_pose_plotter(frame, ans_info, (255, 50, 50))

        # write frame
        video_writer.write(frame)

        # collect new input
        new_record = torch.cat([input_head, pred_head], dim=1)[:, 1:, :]
        new_record[0, -1, 6:12] = new_record[0, -1, 0:6] - new_record[0, -2, 0:6]
        new_record[0, -1, 12:18] = new_record[0, -1, 6:12] - new_record[0, -2, 6:12]
        input_head = new_record.contiguous()

    # close video reader / writer
    video_reader.close()
    video_writer.close()

    # sub audio path genarate
    sub_audio_dir = os.path.dirname(audio_path)
    audio_name = os.path.basename(audio_path)
    sub_audio_name = "host.wav" if audio_name == "comp.wav" else "comp.wav"
    sub_audio_path = os.path.join(sub_audio_dir, sub_audio_name)

    # patch audio
    cat_audio(
        output_path,
        output_path.rsplit(".")[0] + "_patched.mp4",
        sub_audio_path,
        start,
        stop,
        data_conf.fps,
        data_conf.sample_stride,
    )


@hydra.main(version_base=None, config_path="./")
def main(cfg: DictConfig = None):
    gen_head_motion(
        cfg.model_type,
        cfg.model_path,
        cfg,
        cfg.movie_path,
        cfg.audio_path,
        cfg.output_path,
        cfg.data,
        cfg.audio,
    )


if __name__ == "__main__":
    main()
