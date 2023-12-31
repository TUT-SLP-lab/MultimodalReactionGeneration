import os
from typing import Tuple
import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torchaudio._backend.soundfile_backend import load

# # audio parameters
# SAMPLE_RATE = 16000

# # fft parameters
# N_FFT = 400
# N_SHIFT = 160

# # voiced section parameters
# THRESHOLD = -4

# # utterance section parameters
# MINIMUM_UTTERANCE_SECTION_LENGTH = 1.0  # sec
# PAUSE_WITH_VOICE = 1.0  # sec
# PAUSE_WITHOUT_VOICE = 2.0  # sec

# # turn section parameters
# UTT_SECTION_MERGIN = 1.0  # sec
# HEAD_MOVEMENT_WARMUP = 1.0  # sec


def compute_log_power(wavef: torch.Tensor, n_fft=400, n_shift=160) -> torch.Tensor:
    num_frames = (len(wavef) - n_fft) // n_shift + 1
    log_power = torch.zeros(num_frames)
    for frame_no in range(num_frames):
        start_index = frame_no * n_shift
        log_power[frame_no] = torch.log(
            torch.sum(wavef[start_index : start_index + n_fft] ** 2)
        )
    return log_power


def collect_voiced_section(log_power: torch.Tensor, threshold: float) -> torch.Tensor:
    voiced = log_power > threshold
    voiced = voiced.int()
    voiced = torch.cat([torch.tensor([0]), voiced, torch.tensor([0])])
    voiced = voiced[1:] - voiced[:-1]
    sections = voiced.nonzero().reshape([-1, 2])

    return sections


def detect_utterance_section(
    voiced_sections_first: torch.Tensor,
    voiced_sections_second: torch.Tensor,
    first_index: int,
    second_index: int,
    fft_rate: float,
    pause_with_voice: float,
    pause_without_voice: float,
    min_length: float,
) -> Tuple[int, int, int, int]:
    first_progress = 0
    second_progress = 0

    first_length = len(voiced_sections_first)
    second_length = len(voiced_sections_second)

    first = lambda idx: voiced_sections_first[first_index + idx]
    second = lambda idx: voiced_sections_second[second_index + idx]

    pause_with_voice = int(fft_rate * pause_with_voice)
    pause_without_voice = int(fft_rate * pause_without_voice)

    while (
        first_progress + first_index < first_length
        and second_progress + second_index < second_length
    ):
        if first_progress + first_index + 1 >= first_length:
            break
        pause_length = first(first_progress + 1)[0] - first(first_progress)[1]
        # update second head for effective index
        while second(second_progress)[0] < first(first_progress)[1]:
            if second_progress + second_index + 1 < second_length:
                second_progress += 1
            break
        # judge second section in pause
        in_pause = second(second_progress)[0] < first(first_progress + 1)[0]
        if in_pause and (pause_with_voice <= pause_length < pause_without_voice):
            _start, _end, _fi, _si = detect_utterance_section(
                voiced_sections_second,
                voiced_sections_first,
                second_index + second_progress,
                first_index + first_progress + 1,
                fft_rate,
                pause_with_voice,
                pause_without_voice,
                min_length,
            )
            if _end - _start < int(fft_rate * min_length):
                in_pause = False
        else:
            in_pause = False

        if pause_length >= pause_with_voice and in_pause:
            break
        elif pause_length >= pause_without_voice:
            break
        else:
            first_progress += 1
            continue

    new_first_index = first_index + first_progress + 1
    new_second_index = second_index + second_progress

    start = int(first(0)[0])
    end = int(first(first_progress)[1])

    return (start, end, new_first_index, new_second_index)


def collect_utterance_section(
    voiced_sections_comp: torch.Tensor,
    voiced_sections_host: torch.Tensor,
    fft_rate: float,
    min_length: float,
    pause_with_voice: float,
    pause_without_voice: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # initialize utterance sections
    utterance_sections_comp = torch.zeros([0, 2])
    utterance_sections_host = torch.zeros([0, 2])

    # terminal condition
    comp_length = len(voiced_sections_comp)
    host_length = len(voiced_sections_host)

    # loop index
    comp_index = 0
    host_index = 0

    while comp_index < comp_length and host_index < host_length:
        comp_head = voiced_sections_comp[comp_index]
        host_head = voiced_sections_host[host_index]

        # comp head is earlier than host head
        if comp_head[0] < host_head[0]:
            voiced_sections_first = voiced_sections_comp
            voiced_sections_second = voiced_sections_host
            first_idx = comp_index
            second_idx = host_index
        # host head is earlier than comp head
        else:
            voiced_sections_first = voiced_sections_host
            voiced_sections_second = voiced_sections_comp
            first_idx = host_index
            second_idx = comp_index

        start, end, first_idx, second_idx = detect_utterance_section(
            voiced_sections_first,
            voiced_sections_second,
            first_idx,
            second_idx,
            fft_rate,
            pause_with_voice,
            pause_without_voice,
            min_length,
        )

        # first section is longer than min_length -> judge utterance section
        if end - start >= int(fft_rate * min_length):
            # comp head is earlier than host head
            if comp_head[0] < host_head[0]:
                # update utterance sections
                utterance_sections_comp = torch.cat(
                    [
                        utterance_sections_comp,
                        torch.tensor([start, end]).view([1, 2]),
                    ]
                )
                # update index
                comp_index = first_idx
                host_index = second_idx
            # host head is earlier than comp head
            else:
                # update utterance sections
                utterance_sections_host = torch.cat(
                    [
                        utterance_sections_host,
                        torch.tensor([start, end]).view([1, 2]),
                    ]
                )
                # update index
                comp_index = second_idx
                host_index = first_idx

        # first section is shorter than min_length -> merge sections mode
        else:
            if comp_head[0] < host_head[0]:
                # update index
                comp_index = first_idx
                # host_index += second_progress # Don't update host_index !
            else:
                # update index
                host_index = first_idx
                # comp_index += second_progress # Don't update comp_index !

    return (utterance_sections_comp, utterance_sections_host)


def plot_utterance_section(
    output_dir: str,
    waveform_comp: torch.Tensor,
    waveform_host: torch.Tensor,
    lp_comp: torch.Tensor,
    lp_host: torch.Tensor,
    ut_sec_comp: torch.Tensor,
    ut_sec_host: torch.Tensor,
    sampling_rate: float,
    window_size=400,
    stride=160,
    time_range=(0, 15),
):
    # calc times
    wav_start = time_range[0] * sampling_rate
    wav_end = time_range[1] * sampling_rate
    # calc log power times
    lp_start = (time_range[0] * sampling_rate) // stride
    lp_end = (time_range[1] * sampling_rate) // stride

    # coordiante log power sequence
    coordinater = math.ceil(window_size / stride)
    lp_comp = torch.cat([torch.zeros(coordinater), lp_comp])
    lp_host = torch.cat([torch.zeros(coordinater), lp_host])

    # highlight utterance section: convert to sec
    ut_sec_comp = ut_sec_comp / sampling_rate * stride
    ut_sec_host = ut_sec_host / sampling_rate * stride

    # grouping
    waveform = [waveform_comp[wav_start:wav_end], waveform_host[wav_start:wav_end]]
    lp = [lp_comp[lp_start:lp_end], lp_host[lp_start:lp_end]]
    ut_sec = [ut_sec_comp, ut_sec_host]
    color_set1 = ["paleturquoise", "navajowhite"]
    color_set2 = ["blue", "red"]

    # time axis
    times = np.arange(wav_start, wav_end) / sampling_rate
    lp_times = np.arange(wav_start, wav_end, stride) / sampling_rate

    # prepare figure
    fig = plt.figure(figsize=(20, 12))
    axs = fig.subplots(2, 1)

    for i in range(2):
        axs[i].set_title("comp utterance section")
        # plot waveform
        axs[i].set_ylabel("amplitude")
        axs[i].set_xlabel("times")
        axs[i].set_xlim(time_range[0], time_range[1])
        axs[i].set_ylim(-0.8, 0.8)
        axs[i].plot(times, waveform[i], label="waveform", color="gray", linewidth=3)

        # plot log power
        ax_lp = axs[i].twinx()
        ax_lp.set_ylabel("power")
        ax_lp.set_xlim(time_range[0], time_range[1])
        ax_lp.set_ylim(-8, 4)
        ax_lp.plot(lp_times, lp[i], label="log power", color=color_set1[i])

        # highlight utterance section
        for sec in ut_sec[i]:
            start = sec[0]
            end = sec[1]

            if end < time_range[0] or time_range[1] < start:
                continue
            start = max(start, time_range[0])
            end = min(end, time_range[1])
            axs[i].axvspan(start, end, color=color_set2[i], alpha=0.3)

    # save figure
    start = str(time_range[0]).zfill(3)
    end = str(time_range[1]).zfill(3)
    output_path = os.path.join(output_dir, f"utterance_{start}_{end}.png")
    fig.savefig(output_path)

    # reset fig
    plt.close(fig)
    plt.clf()


def utterance_to_turn_section(
    utterance_sections: torch.Tensor,
    mergin: float,
    samplerate: int,
    stride: int,
    length: float,
):
    # initialize turn sections
    turn_sections = torch.zeros([0, 2])

    # convert index -> sec
    utterance_sections = utterance_sections / samplerate * stride

    for utt_sec in utterance_sections:
        # expand utterance section
        start = max(utt_sec[0] - mergin, 0)
        end = min(utt_sec[1] + mergin, length)
        # update turn sections
        turn_sections = torch.cat(
            [turn_sections, torch.tensor([start, end]).view([1, 2])]
        )

    return turn_sections


def get_uttrance_section(
    host_path: str,
    comp_path: str,
    sampling_rate: int,
    window_size: int = 400,
    stride: int = 160,
    threshold: float = -4,
    minimum_utterance_length: float = 1.0,
    pause_with_voice: float = 1.0,
    pause_without_voice: float = 2.0,
    mergin: float = 1.0,
    # for experiment or debug
    exp_plot: bool = False,
    exp_plot_dir: str = "data/temp/utterance_section",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """for getting utterance section

    Args:
        host_path (str): host wav path
        comp_path (str): comp wav path
        sampling_rate (int): sampling rate
        window_size (int, optional): fft window size. Defaults to 400.
        stride (int, optional): fft shift size. Defaults to 160.
        threshold (float, optional): voice log power threshold. Defaults to -4.
        minimum_utterance_length (float, optional): minimum utterance section length. Defaults to 1.0.
        pause_with_voice (float, optional): maximum pause length with voice. Defaults to 1.0.
        pause_without_voice (float, optional): maximum pause length without voice. Defaults to 2.0.
        mergin (float, optional): utterance section mergin. Defaults to 1.0.
        exp_plot (bool, optional): plot utterance section by matplotlib. Defaults to False.
        exp_plot_dir (str, optional): output dir for plot result. Defaults to "data/temp/utterance_section".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: utterance section of host and comp. Size is [num_section, 2]. The unit is sec.
    """
    # load waveforms
    waveform_comp, sample_rate_comp = load(comp_path)
    waveform_host, sample_rate_host = load(host_path)
    # check sample rate
    assert sample_rate_comp == sample_rate_host
    assert sample_rate_comp == sampling_rate
    # check length
    assert len(waveform_comp) == len(waveform_host)

    # reshape waveform to 1D
    waveform_comp = waveform_comp[0]
    waveform_host = waveform_host[0]

    # compute log power
    log_power_comp = compute_log_power(waveform_comp, n_fft=window_size, n_shift=stride)
    log_power_host = compute_log_power(waveform_host, n_fft=window_size, n_shift=stride)

    # collect voiced section by log power THRESHOLD
    voiced_sections_comp = collect_voiced_section(log_power_comp, threshold)
    voiced_sections_host = collect_voiced_section(log_power_host, threshold)

    # collect utterance section by voiced section
    utterance_section_comp, utterance_section_host = collect_utterance_section(
        voiced_sections_comp,
        voiced_sections_host,
        sampling_rate / stride,
        minimum_utterance_length,
        pause_with_voice,
        pause_without_voice,
    )

    time_length = 15
    audio_length = len(waveform_comp) / sampling_rate

    # plot utterance section
    if exp_plot:
        # make output dir
        data_dir = os.path.dirname(host_path)
        data_lot = os.path.split(data_dir)[-1]
        output_dir = os.path.join(exp_plot_dir, data_lot)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in tqdm(range(math.floor(audio_length / time_length))):
            start = i * time_length
            end = (i + 1) * time_length

            plot_utterance_section(
                output_dir,
                waveform_comp,
                waveform_host,
                log_power_comp,
                log_power_host,
                utterance_section_comp,
                utterance_section_host,
                sampling_rate,
                time_range=(start, end),
            )

    # turn section extraction
    turn_section_comp = utterance_to_turn_section(
        utterance_section_comp,
        mergin,
        sampling_rate,
        stride,
        audio_length,
    )
    turn_section_host = utterance_to_turn_section(
        utterance_section_host,
        mergin,
        sampling_rate,
        stride,
        audio_length,
    )

    return turn_section_comp, turn_section_host
