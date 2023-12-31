import math
import os
import json
import shutil
import pickle
import dataclasses
from logging import Logger
from datetime import datetime
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
from dfcon import Directory, DircFilter
from toolpack.parallel import parallel_luncher

import numpy as np
import torch

from mr_gen.utils import DummyLogger
from mr_gen.utils.io import ZERO_PADDING
from mr_gen.databuild.utterance_analysis.speech_segmentation import get_uttrance_section


CACHE_PATH = "./data/temp"
DATAINFO_FILE = "datainfo.pkl"
DATASET_FILE = "dataset.json"


@dataclasses.dataclass
class DataBuildDataNX(object):
    data_dir: str
    fps: float
    pred_fps: float
    pred_shift: int
    max_len: int
    min_len: int
    shift_len: int
    leading_len: int

    sample_rate: int
    nfft: int
    shift: int

    threshold: float
    minimum_utterance_length: float
    pause_with_voice: float
    pause_without_voice: float
    mergin: float

    use_partner_motion: bool
    use_partner_audio: bool
    use_self_motion: bool
    use_self_audio: bool

    target_shift: int

    use_centroid: bool
    use_angle: bool
    delta_order: int

    def __post_init__(self):
        self.check_args()

        if self.pred_fps is None:
            self.pred_fps = self.fps / self.pred_shift
        if self.pred_shift is None:
            self.pred_shift = self.fps / self.pred_fps
        self.fft_rate = self.sample_rate / self.shift
        self.target_shift_real = self.target_shift * self.pred_shift

        # coordinate leading length
        self.leading_len -= self.leading_len % self.pred_shift

    def check_args(self):
        if self.pred_fps is None and self.pred_shift is None:
            raise ValueError("Specify either pred_fps or pred_shift.")
        if self.pred_fps is not None:
            if (self.fps / self.pred_fps) % 1 != 0:
                raise ValueError("pred_fps must be a multiple of fps.")
            if (self.sample_rate / self.shift) / self.pred_fps % 1 != 0:
                raise ValueError("pred_fps must be a multiple of sample_rate/shift.")
        if self.pred_shift is not None:
            if (self.sample_rate / self.shift) / self.pred_shift % 1 != 0:
                raise ValueError("pred_shift must be a multiple of sample_rate/shift.")
        if self.max_len < self.min_len:
            raise ValueError("max_len must be greater than min_len.")


class DataBuilderNX(DataBuildDataNX):
    def __init__(self, cfg: DictConfig, logger: Optional[Logger] = None):
        content: Dict[str, Any] = dict(cfg)
        self.no_cache_build = content.pop("no_cache_build", False)
        self.clear_cache = content.pop("clear_cache", False)

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

        self.data_file = os.path.join(self.data_site, DATASET_FILE)
        self.fft_fps_freq_rate = int(self.sample_rate / self.shift / self.fps)

        # build directory structure
        direc = Directory(self.data_dir)
        # get directory list
        dfilter = DircFilter().contained_literal("data").only_terminal()
        self.data_dircs = direc.get_instances(dfilter)
        assert len(self.data_dircs) != 0, "Not found data."

        self.logger.info("Start build data.")
        self.build()
        self.logger.info("Finish build data.")

    def check_cache(self) -> Optional[str]:
        for dirs in os.listdir(CACHE_PATH):
            if dirs.rsplit("_", maxsplit=1)[0] != self.base_dir_name:
                continue
            if DATAINFO_FILE in os.listdir(CACHE_PATH + "/" + dirs):
                with open(CACHE_PATH + "/" + dirs + "/" + DATAINFO_FILE, "rb") as f:
                    prev_self: DataBuilderNX = pickle.load(f)
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

    def build(self):
        arg_list = []
        parallel = os.cpu_count()

        for datas in self.data_dircs:
            datas_files = list(datas.get_file_path())
            wav_file = {"host": "", "comp": ""}
            motion_npz = {"host": [], "comp": []}
            for files in datas_files:
                _file_path = os.path.basename(files)
                if _file_path.startswith("host") and _file_path.endswith(".npz"):
                    motion_npz["host"].append(files)
                elif _file_path.startswith("comp") and _file_path.endswith(".npz"):
                    motion_npz["comp"].append(files)
                elif _file_path.startswith("host") and _file_path.endswith(".wav"):
                    wav_file["host"] = files
                elif _file_path.startswith("comp") and _file_path.endswith(".wav"):
                    wav_file["comp"] = files
                else:
                    pass
            motion_npz["host"].sort()
            motion_npz["comp"].sort()
            arg_list.append((motion_npz, wav_file))

        parallel_luncher(
            self.make_segment_nx,
            arg_list,
            parallel,
            unpack=True,
            # tqdm args
            desc="Bld data.",
            position=0,
        )

    def make_segment_nx(
        self,
        motion_npz: Dict[str, List[str]],
        wav_file: Dict[str, str],
    ):
        host_wave = wav_file["host"]
        comp_wave = wav_file["comp"]
        host_motions = motion_npz["host"]
        comp_motions = motion_npz["comp"]
        motion_ignore = self.collect_motion_ignore(host_motions, comp_motions)
        turn_comp, turn_host = get_uttrance_section(
            host_wave,
            comp_wave,
            self.sample_rate,
            self.nfft,
            self.shift,
            self.threshold,
            self.minimum_utterance_length,
            self.pause_with_voice,
            self.pause_without_voice,
            self.mergin,
        )
        if len(turn_comp) == 0:
            self.logger.info(f"Cannot find utterance section : {comp_wave}")
        if len(turn_host) == 0:
            self.logger.info(f"Cannot find utterance section : {host_wave}")

        host_set = {
            "turn_partner": turn_comp,
            "npz_file_self": host_motions,
            "npz_file_partner": comp_motions,
            "ignores": motion_ignore,
            "wav_self": host_wave,
            "wav_partner": comp_wave,
        }
        comp_set = {
            "turn_partner": turn_host,
            "npz_file_self": comp_motions,
            "npz_file_partner": host_motions,
            "ignores": motion_ignore,
            "wav_self": comp_wave,
            "wav_partner": host_wave,
        }
        for data_set in [host_set, comp_set]:
            if len(data_set["turn_partner"]) == 0:
                continue

            wav_self: str = data_set["wav_self"]
            data_name = os.path.split(os.path.dirname(wav_self))[1]
            audio_name = os.path.basename(wav_self).rsplit(".", maxsplit=1)[0]
            out_name, out_ext = DATASET_FILE.rsplit(".", maxsplit=1)
            output_file = out_name + "_" + data_name + "_" + audio_name + "." + out_ext
            output_file = os.path.join(self.data_site, output_file)

            self.output_segment(
                output_file,
                **data_set,
            )

    def output_segment(
        self,
        output_path: str,
        turn_partner: torch.Tensor,
        npz_file_self: List[str],
        npz_file_partner: List[str],
        ignores: torch.Tensor,
        wav_self: str,
        wav_partner: str,
    ):
        self_npz = [np.load(m) for m in npz_file_self]
        sectons_self = [m["section"] for m in self_npz]
        partner_npz = [np.load(m) for m in npz_file_partner]
        sectons_partner = [m["section"] for m in partner_npz]

        for motion, audio in self.process_motion(turn_partner, ignores):
            start, end, s_lead, e_lead = motion
            s_audio, e_audio, sl_audio, el_audio = audio

            data_path_self = ""
            data_path_partner = ""
            data_offset_self = 0
            data_offset_partner = 0

            target_start = start + self.target_shift_real
            target_end = end + self.target_shift_real
            if ignores[target_start:target_end].sum() > 0:
                continue

            for i, sec in enumerate(sectons_self):
                if sec[0] <= start and end <= sec[1]:
                    data_path_self = npz_file_self[i]
                    data_offset_self = int(sec[0])
                    break
            for i, sec in enumerate(sectons_partner):
                if sec[0] <= start and end <= sec[1]:
                    data_path_partner = npz_file_partner[i]
                    data_offset_partner = int(sec[0])
                    break
            section_info = f"section: start={start}, end={end}"
            ignore_info = f"exist ignore: {ignores[start:end].sum()>0}"
            msg = f"Cannot find motion data : {wav_self}\n{section_info}\n{ignore_info}"
            assert data_path_self != "" and data_path_partner != "", msg

            partner_motion = {
                "path": data_path_partner,
                "seq": {"start": start, "end": end, "stride": self.pred_shift},
                "lead": {"start": s_lead, "end": e_lead, "stride": self.pred_shift},
                "offset": data_offset_partner,
                "delta_order": self.delta_order,
            }
            partner_audio = {
                "path": wav_partner,
                "seq": {"start": s_audio, "end": e_audio, "stride": 1},
                "lead": {"start": sl_audio, "end": el_audio, "stride": 1},
                "delta_order": self.delta_order,
            }
            self_motion = {
                "path": data_path_self,
                "seq": {"start": start, "end": target_end, "stride": self.pred_shift},
                "lead": {"start": s_lead, "end": e_lead, "stride": self.pred_shift},
                "offset": data_offset_self,
                "delta_order": self.delta_order,
            }
            self_audio = {
                "path": wav_self,
                "seq": {"start": s_audio, "end": e_audio, "stride": 1},
                "lead": {"start": sl_audio, "end": el_audio, "stride": 1},
                "delta_order": self.delta_order,
            }
            target = {
                "shift_real_seq": self.target_shift_real,
                "shift_input_seq": self.target_shift,
                "delta_order": self.delta_order,
            }

            segment = {
                "partner_motion": partner_motion if self.use_partner_motion else None,
                "partner_audio": partner_audio if self.use_partner_audio else None,
                "self_motion": self_motion if self.use_self_motion else None,
                "self_audio": self_audio if self.use_self_audio else None,
                "target": target,
            }
            json_segment = json.dumps(segment, ensure_ascii=False)

            idx = str(start).zfill(ZERO_PADDING)
            output_name, output_ext = output_path.rsplit(".", maxsplit=1)
            _output_path = output_name + "_" + idx + "." + output_ext

            with open(_output_path, "w", encoding="utf-8") as f:
                f.write(json_segment + "\n")

    def collect_motion_ignore(
        self, host_motion: List[str], comp_motion: List[str]
    ) -> torch.Tensor:
        motion_ignore_host = np.array([], dtype=np.int32)
        motion_ignore_comp = np.array([], dtype=np.int32)
        host_motion = [np.load(hm) for hm in host_motion]
        comp_motion = [np.load(cm) for cm in comp_motion]

        for motion in host_motion:
            if len(motion_ignore_host) < motion["section"][-1]:
                start = motion["section"][0]
                end = motion["section"][1]
                interval = np.ones(start - len(motion_ignore_host), dtype=np.int32)
                section = np.zeros(end - start, dtype=np.int32)
                motion_ignore_host = np.concatenate(
                    [motion_ignore_host, interval, section], axis=-1
                )
        for motion in comp_motion:
            if len(motion_ignore_comp) < motion["section"][-1]:
                start = motion["section"][0]
                end = motion["section"][1]
                interval = np.ones(start - len(motion_ignore_comp), dtype=np.int32)
                section = np.zeros(end - start, dtype=np.int32)
                motion_ignore_comp = np.concatenate(
                    [motion_ignore_comp, interval, section], axis=-1
                )

        motion_ignore = np.zeros(
            max(len(motion_ignore_host), len(motion_ignore_comp)), dtype=np.int32
        )

        max_len = max(len(motion_ignore_host), len(motion_ignore_comp))
        min_len = min(len(motion_ignore_host), len(motion_ignore_comp))
        ignore_tail = max_len - min_len
        if ignore_tail > 0:
            motion_ignore[-ignore_tail:] = 1

        for i, _ in enumerate(motion_ignore):
            if i < len(motion_ignore_host) and motion_ignore_host[i] == 1:
                motion_ignore[i] = 1
                continue
            if i < len(motion_ignore_comp) and motion_ignore_comp[i] == 1:
                motion_ignore[i] = 1
                continue

        return torch.tensor(motion_ignore)

    def process_motion(
        self,
        turn_section: torch.Tensor,
        motion_ignore: torch.Tensor,
    ):
        # convert turn_section unit from second to fps
        turn_section = turn_section * self.fps
        turn_section = turn_section.int()
        delta_mergin = self.delta_order * self.pred_shift

        audio_offset = (self.nfft - self.shift) + (self.shift * self.delta_order)
        motion_offset = math.ceil(audio_offset * self.fps / self.sample_rate)

        for turn in turn_section:
            start = max(turn[0], self.leading_len + delta_mergin + motion_offset)
            end = min(turn[1], len(motion_ignore))
            if end - start < self.min_len:
                continue

            start = int(start)
            end = int(end)

            for i in range(start, end, self.shift_len):
                _start = i
                _end = min(end, i + self.max_len)
                _s_lead = _start - self.leading_len
                _e_lead = _start

                if _end - _start < self.min_len:
                    continue

                # coordinate end
                rest = (_end - _start) % self.pred_shift
                _end -= rest

                audio_start = int(_start * self.sample_rate / self.fps) - audio_offset
                audio_end = int(_end * self.sample_rate / self.fps)
                audio_s_lead = int(_s_lead * self.sample_rate / self.fps) - audio_offset
                audio_e_lead = int(_e_lead * self.sample_rate / self.fps)

                _start = _start - delta_mergin
                _s_lead = _s_lead - delta_mergin

                if torch.sum(motion_ignore[_start : _end + self.target_shift_real]) > 0:
                    continue
                if torch.sum(motion_ignore[_s_lead:_e_lead]) > 0:
                    continue

                motion = (_start, _end, _s_lead, _e_lead)
                audio = (audio_start, audio_end, audio_s_lead, audio_e_lead)

                yield motion, audio
