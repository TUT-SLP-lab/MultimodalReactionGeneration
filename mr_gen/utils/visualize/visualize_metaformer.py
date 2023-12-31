import os
import pickle
from typing import Union
import time
import hydra
import numpy as np
from numpy import ndarray
from PIL import Image
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import soundfile as sf
from omegaconf import DictConfig
from tqdm import tqdm

from mr_gen.utils import open_video
from mr_gen.utils.video import patch_audio
from mr_gen.model.model_loader import load_model
from mr_gen.utils.visualize import head_pose_plotter
from mr_gen.utils.tools.adapter import FaceAdapter
from mr_gen.databuild import DataBuilderNX
from mr_gen.utils.visualize.dataloader.dataloader import collate_fn, HeadMotionDatasetNX
from mr_gen.model.lstmformer import Metaformer
from mr_gen.model.lstm_with_sampling import LSTMwithSample

VISUALIZE_PATH = "./data/visualize"
HEAD_PATH = "./data/sample.head"
VIDEO_BASE_PATH = "./data/visualize_move_video"

FACE_W = 300
HEAD_OFFSET = -10


def setup_dataloader(
    data: DictConfig, motion: DictConfig, audio: DictConfig
) -> DataLoader:
    # build dataset
    data.data_dir = "./data/visualize_move"
    dataset_path = DataBuilderNX(data, None).data_site
    dataset = HeadMotionDatasetNX(dataset_path, motion, audio)

    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        shuffle=False,
    )


def load_face() -> ndarray:
    with open(HEAD_PATH, "rb") as f:
        head: FaceAdapter = pickle.load(f)[1]  # (idx, head)

    return head.face


def cat_audio(
    video_path: str,
    out_path: str,
    audio_path: str,
    start: int,  # video frame index
    stop: int,  # video frame index
    fps: float,  # video fps
    stride: int,  # video stride
) -> None:
    # extract audio & data info
    waveforme, sample_rate = sf.read(audio_path)
    transformer = lambda x: int(sample_rate * x / fps)
    start_idx = transformer(start)
    stop_idx = transformer(stop + stride)

    # write audio
    waveforme = waveforme[start_idx:stop_idx]
    wave_out_path = out_path.rsplit(".", maxsplit=1)[0] + ".wav"
    sf.write(wave_out_path, waveforme, sample_rate)

    # patch audio
    patch_audio(out_path, video_path, wave_out_path)


def generation_step(batch, model: Union[LSTMwithSample, Metaformer], model_name: str):
    fbank_partner = batch[0]
    motion_partner = batch[1]
    motion_self = batch[2]
    leading_fbank_partner = batch[3]
    leading_motion_partner = batch[4]
    leading_motion_self = batch[5]
    _target = batch[6]

    inputs = (
        fbank_partner[0],
        motion_partner[0],
        motion_self[0],
        leading_fbank_partner[0],
        leading_motion_partner[0],
        leading_motion_self[0],
        _target[0],
    )

    st_info = motion_self[1]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    a_mean = torch.tensor(st_info["angle_mean"]).unsqueeze(0).unsqueeze(0).to(device)
    a_std = torch.tensor(st_info["angle_std"]).unsqueeze(0).unsqueeze(0).to(device)
    c_mean = torch.tensor(st_info["centroid_mean"]).unsqueeze(0).unsqueeze(0).to(device)
    c_std = torch.tensor(st_info["centroid_std"]).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        start_time = time.time()
        prediction, target = model.prediction(
            list(inputs), use_scheduled_sampling=False, full_generation=True
        )
        end_time = time.time()

    speed_path = os.path.join(VISUALIZE_PATH, model_name, "speed.log")
    with open(speed_path, "a", encoding="utf-8") as f:
        f.write(f"{end_time - start_time}\n")

    pred_angle = prediction[:, :, :3] * a_std + a_mean
    trgt_angle = target[:, :, :3] * a_std + a_mean
    pred_centd = prediction[:, :, 3:6] * c_std + c_mean
    trgt_centd = target[:, :, 3:6] * c_std + c_mean

    res_dict = {
        "pred": {
            "centroid": pred_centd.squeeze(0),
            "angle": pred_angle.squeeze(0),
            "path": motion_self[2],
            "sss": motion_self[3],
        },
        "target": {
            "centroid": trgt_centd.squeeze(0),
            "angle": trgt_angle.squeeze(0),
            "path": motion_self[2],
            "sss": _target[3],
        },
    }

    return res_dict


def record_statics(statics, centroid, angle, face, black_board, clr=(50, 255, 50)):
    black_board = np.zeros_like(black_board)
    _centd = centroid.copy()
    _centd[0] = 0.5  # center
    info = {"face": face, "centroid": _centd, "angle": angle}
    head_pose_plotter(black_board, info, clr)
    qrt = black_board.shape[1] // 3
    black_board = black_board[qrt:-qrt, qrt:-qrt]
    statics.append(black_board)
    return statics


def gen_head_motion(
    model: str,
    model_path: str,
    model_config: DictConfig,
    data_conf: DictConfig,
    audio_conf: DictConfig,
    motion_conf: DictConfig,
    plot_answer: bool = False,
) -> None:
    """Generate head motion from movie and audio.

    Args:
        movie_path (str): movie path
        audio_path (str): audio path
        output_path (str): output path
    """
    # load face
    face = load_face()

    model_name = model

    # load model @ eval mode
    model: pl.LightningModule = load_model(model, model_path, model_config).eval()
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")

    # head motion generation
    for batch in tqdm(setup_dataloader(data_conf, motion_conf, audio_conf)):
        res = generation_step(batch, model, model_name)

        target_path = res["target"]["path"][1]
        target_seg_name = os.path.basename(res["target"]["path"][0])
        target_seg_name = target_seg_name.rsplit(".", maxsplit=1)[0]

        target_name: str = os.path.basename(target_path)
        target_is_host = "host" in target_name
        target_data_name = os.path.split(os.path.dirname(target_path))[-1]
        movie_path = os.path.join(VIDEO_BASE_PATH, target_data_name)
        if target_is_host:
            movie_path = os.path.join(movie_path, "comp.mp4")
        else:
            movie_path = os.path.join(movie_path, "host.mp4")
        audio_path = os.path.join(VIDEO_BASE_PATH, target_data_name, "pair.wav")

        visualize_path = os.path.join(VISUALIZE_PATH, model_name)
        output_path = os.path.join(visualize_path, target_data_name)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        output_path_dir = os.path.join(output_path, target_seg_name)
        if not os.path.isdir(output_path_dir):
            os.makedirs(output_path_dir)
        output_path = os.path.join(
            output_path_dir, target_seg_name.rsplit(".", maxsplit=1)[0] + ".mp4"
        )

        # open video reader / writer
        video_reader = open_video(movie_path, "r")
        video_writer = open_video(output_path, "w", fps=data_conf.pred_fps)

        results = [
            res["pred"]["centroid"],
            res["pred"]["angle"],
            res["target"]["centroid"],
            res["target"]["angle"],
        ]

        start = res["pred"]["sss"][0]
        stop = res["target"]["sss"][1]
        stride = res["pred"]["sss"][2]

        statics = []
        t_statics = []

        for i, (centd, angle, t_centd, t_angle) in enumerate(zip(*results)):
            # tensor to ndarray
            centd: ndarray = centd.cpu().numpy()
            angle: ndarray = angle.cpu().numpy()
            t_centd: ndarray = t_centd.cpu().numpy()
            t_angle: ndarray = t_angle.cpu().numpy()

            info = {"face": face, "centroid": centd, "angle": angle}

            target_frame_idx = start + (i + 1) * stride
            frame = video_reader[target_frame_idx]

            black_board = np.zeros_like(frame)

            if plot_answer:
                t_info = {"face": face, "centroid": t_centd, "angle": t_angle}
                # plot answer
                gray = (50, 50, 50)
                pink_gray = (50, 50, 100)
                black_board = head_pose_plotter(black_board, t_info, gray, pink_gray)

            # plot head motion
            black_board = head_pose_plotter(black_board, info, (50, 255, 50))

            # cat frame and black_board
            frame = np.concatenate([frame, black_board], axis=1)

            # write frame
            video_writer.write(frame)

            if (i + 1) % 3 == 0:
                statics = record_statics(
                    statics, centd, angle, face, black_board, (50, 255, 50)
                )
                t_statics = record_statics(
                    t_statics, t_centd, t_angle, face, black_board, (170, 170, 170)
                )

        # close video reader / writer
        video_reader.close()
        video_writer.close()

        patch_mp4_path = output_path.rsplit(".", maxsplit=1)[0] + "_patched.mp4"

        # patch audio
        cat_audio(
            output_path,
            patch_mp4_path,
            audio_path,
            start,
            stop,
            data_conf.fps,
            2,
        )

        for i in range(0, len(statics), 8):
            f_s = np.concatenate(statics[i : i + 8], axis=1)
            Image.fromarray(f_s).save(
                os.path.join(output_path_dir, f"static_{i//8}.png")
            )
        for i in range(0, len(t_statics), 8):
            f_s = np.concatenate(t_statics[i : i + 8], axis=1)
            Image.fromarray(f_s).save(
                os.path.join(output_path_dir, f"t_static_{i//8}.png")
            )


@hydra.main(version_base=None, config_path="./")
def main(cfg: DictConfig = None):
    gen_head_motion(
        cfg.model_type,
        cfg.model_path,
        cfg,
        cfg.data,
        cfg.audio,
        cfg.motion,
        True,
    )


if __name__ == "__main__":
    main()
