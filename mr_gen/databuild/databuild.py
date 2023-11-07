import math
import os
import json
import wave
import shutil
import pickle
import dataclasses
from logging import Logger
from datetime import datetime
from typing import Any, Dict, Optional
from omegaconf import DictConfig
from tqdm import tqdm
import torchaudio._backend.soundfile_backend as torchaudio_sf
from dfcon import Directory, FileFilter

from mr_gen.utils import DummyLogger
from mr_gen.utils.tools.adapter import FaceAdapter
from mr_gen.utils.io import ZERO_PADDING
from mr_gen.utils.parallel import parallel_luncher


CACHE_PATH = "./data/temp"
DATAINFO_FILE = "datainfo.pkl"
DATASET_FILE = "dataset.json"


@dataclasses.dataclass
class DataBuildData(object):
    data_dir: str
    fps: float
    context_start: int
    sample_stride: int
    context_size: int
    context_stride: int
    target_type: str
    target_position: int
    target_size: int
    target_stride: int
    delta_order: int
    no_cache_build: bool
    clear_cache: bool

    sample_rate: int
    nfft: int
    shift: int

    use_centroid: bool
    use_angle: bool

    def __post_init__(self):
        self.context_length = self.context_size * self.context_stride
        self.context_end = self.context_start + self.context_length
        self.target_length = self.target_size * self.target_stride
        self.target_end = self.target_position + self.target_length

        self.check_args()

    def check_args(self):
        if self.target_type not in ["direct", "context"]:
            raise ValueError("target_type must be 'direct' or 'context'")
        if self.target_type == "direct" and self.target_size != 1:
            raise ValueError("target_size must be 1 when target_type is 'direct'")
        if self.target_size < 1:
            raise ValueError("target_size must be positive.")
        if self.context_size < 1:
            raise ValueError("context_size must be positive.")
        if self.context_start >= 0:
            raise ValueError("context_start must be negative.")
        if self.context_stride < 1:
            raise ValueError("context_stride must be positive.")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive.")

        self.fft_freq = self.sample_rate / self.shift
        self.sample_fps = self.fps / self.sample_stride
        if self.fft_freq / self.sample_fps % 1 != 0:
            raise ValueError(
                "stft frequency (sampleingrate /shift) must be multiple of fps."
            )


class DataBuilder(DataBuildData):
    def __init__(self, cfg: DictConfig, logger: Optional[Logger] = None):
        content: Dict[str, Any] = dict(cfg)
        super().__init__(**content)

        self.cfg = cfg
        self.logger = logger if logger is not None else DummyLogger()

        # generate data site path
        ymd = datetime.now().strftime("%Y%m%d%H%M%S")
        self.base_dir_name = os.path.split(self.data_dir)[-1]
        self.data_site = CACHE_PATH + "/" + self.base_dir_name + "_" + ymd

        # clear cache
        if self.clear_cache:
            self.logger.info("Clear dataset cache.")
            shutil.rmtree(CACHE_PATH)
            os.mkdir(CACHE_PATH)
            with open(CACHE_PATH + "/.gitkeep", "w", encoding="utf-8") as f:
                f.write("")

        # check cache, and make data site or set cache path
        if not self.judge_rebuild_data():
            self.data_site = self.check_cache()
            self.logger.info("Already builded data.")
            return
        self.logger.info("Not found cache data (or re-build).")

        self.data_file = self.data_site + "/" + DATASET_FILE
        self.fft_fps_freq_rate = int(self.sample_rate / self.shift / self.fps)

        # build directory structure
        ffilter = FileFilter().include_extention(["wav"])
        ffilter = ffilter.contained(["host", "comp"])
        direc = Directory(self.data_dir).build_structure(ffilter)

        # collect wav file path
        self.wav_list = direc.get_file_path(serialize=True)

        self.logger.info("Start build data.")
        self.build()
        self.logger.info("Finish build data.")

    def group_key_gen(self, path: str) -> str:
        return os.path.basename(path).rsplit(".")[0]

    def check_cache(self) -> Optional[str]:
        for dirs in os.listdir(CACHE_PATH):
            if dirs.rsplit("_", maxsplit=1)[0] != self.base_dir_name:
                continue
            if DATAINFO_FILE in os.listdir(CACHE_PATH + "/" + dirs):
                with open(CACHE_PATH + "/" + dirs + "/" + DATAINFO_FILE, "rb") as f:
                    prev_self: DataBuilder = pickle.load(f)
                if prev_self == self:
                    return CACHE_PATH + "/" + dirs
        return None

    def judge_rebuild_data(self) -> bool:
        prev_dir = self.check_cache()
        self.builded = False
        if prev_dir and not self.no_cache_build:
            return False  # mean already builded data
        else:
            if prev_dir and self.no_cache_build:
                self.logger.info(f"Clear previous cache : {prev_dir}")
                shutil.rmtree(prev_dir)

            os.mkdir(self.data_site)
            with open(self.data_site + "/" + DATAINFO_FILE, "wb") as f:
                pickle.dump(self, f)

            return True  # mean not builded data or rebuild

    def is_head_none(self, head_dir: str, start: int, end: int, stride: int) -> bool:
        idx = 0
        for idx in range(start, end, stride):
            file_idx = str(idx).zfill(ZERO_PADDING)
            file_name = os.path.split(head_dir)[1] + "_" + file_idx + ".head"
            with open(head_dir + "/" + file_name, "rb") as f:
                face: FaceAdapter = pickle.load(f)[1]  # (idx, head)
            if face is None:
                return True

        return False

    def coordinate_audio_start(self, audio_start: int, audio_end: int) -> int:
        duration_sample = audio_end - audio_start
        edge = (duration_sample - self.nfft) % self.shift
        if edge == 0:
            return audio_start

        lack = self.shift - edge
        return audio_start - lack

    def build(self):
        arg_list = []
        for i, wav_file in enumerate(self.wav_list):
            base_path, wav_name = os.path.split(wav_file)
            base_name = wav_name.rsplit(".", maxsplit=1)[0]
            head_dir = os.path.join(base_path, base_name)
            parallel = os.cpu_count()

            arg_list.append((head_dir, wav_file, i % parallel == 0))

        parallel_luncher(
            self.make_segment,
            arg_list,
            parallel,
            unpack=True,
            desc="Bld data.",
            position=0,
        )

    def make_segment(self, head_dir: str, wav_file: str, use_tq: bool = False):
        head_list = os.listdir(head_dir)
        head_len = len(head_list)

        # compute audio start offset
        audio_offset = (self.shift * self.delta_order) + (self.nfft - self.shift)
        head_offset = math.ceil(audio_offset * self.fps / self.sample_rate) + 1

        _, sample_rate = torchaudio_sf.load(wav_file, 0, 1)
        if sample_rate != self.sample_rate:
            raise ValueError("sample rate of wav file is not match.")

        # collect audio length
        with wave.open(wav_file, "rb") as f:
            audio_samples = f.getnframes()

        # for display progress
        path, dir_name = os.path.split(head_dir)
        path, base_name = os.path.split(path)
        target_name = os.path.join(base_name, dir_name)
        iterator = range(0, head_len, self.sample_stride)

        if use_tq:
            iterator = tqdm(iterator, position=1, desc=target_name, leave=False)

        for i in iterator:
            # start condition
            minimun_start = (
                abs(self.context_start)
                + self.delta_order * self.context_stride
                + head_offset
            )
            if i < minimun_start:
                continue
            # end condition
            if i + self.target_position + self.target_size > head_len:
                break

            jdic = {
                "head_dir": head_dir,
                "wav_file": wav_file,
                "fps": self.fps,
                "sample_fps": self.sample_fps,
                "idx": i,
            }

            # check head file of context
            cntx_start = i + self.context_start - self.delta_order * self.context_stride
            cntx_end = i + self.context_end
            if self.is_head_none(head_dir, cntx_start, cntx_end, self.context_stride):
                continue
            jdic["context"] = {
                "start": cntx_start,
                "end": cntx_end,
                "stride": self.context_stride,
            }

            # check head file of target
            _target_position = i + self.target_position
            trgt_start = _target_position - self.delta_order * self.target_stride
            trgt_end = i + self.target_end
            if self.is_head_none(head_dir, trgt_start, trgt_end, self.target_stride):
                continue
            jdic["target"] = {
                "start": trgt_start,
                "end": trgt_end,
                "stride": self.target_stride,
            }

            # calculate range of audio
            fft_length = int(self.context_size * self.fft_freq / self.sample_fps)
            sample_length = fft_length * self.shift + audio_offset
            audio_end = int(cntx_end * self.sample_rate / jdic["fps"])
            audio_start = audio_end - sample_length
            if audio_start < 0 or audio_end >= audio_samples:
                continue

            jdic["audio"] = {"start": audio_start, "end": audio_end}

            jdic["delta_order"] = self.delta_order

            ext_name = "_".join(os.path.split(target_name)) + str(i) + ".json"
            output_path = self.data_file.rsplit(".", maxsplit=1)[0] + "_" + ext_name

            # dump json
            with open(output_path, "w", encoding="utf-8") as f:
                jdic_str = json.dumps(jdic)
                f.write(jdic_str + "\n")
