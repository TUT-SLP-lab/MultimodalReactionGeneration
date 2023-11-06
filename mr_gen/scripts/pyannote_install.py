# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained voice activity detection pipeline

import numpy as np
import pyworld as pw
import torch
from torchaudio._backend.soundfile_backend import load
import torchaudio.transforms as T
from matplotlib import pyplot as plt

# import librosa
# import torchaudio
N_FFT = 400
HOP_LENGTH = 160


def plot_spectrogram(specgram, lp, sr, wf, pitch, window_size=400, stride=160):
    times = np.arange(0, len(wf)) / sr

    _fig = plt.figure(figsize=(20, 12))
    _ax1, _ax2, _ax3 = _fig.subplots(3, 1)
    _ax1.set_title("log mel spectrogram")
    _ax1.set_ylabel("freq_bin")
    _ax1.imshow(
        specgram, origin="lower", extent=[0, len(wf) / sr, 0, 25], aspect="auto"
    )

    _ax2.set_title("log power")
    _ax2.set_ylabel("power")
    _ax2.set_xlabel("times")
    _ax2.set_xlim(0, len(wf) / sr)
    _ax2.plot(times[window_size - 1 :: stride], lp)

    _ax3.set_title("waveform")
    _ax3.set_ylabel("amplitude")
    _ax3.set_xlabel("times")
    _ax3.set_xlim(0, len(wf) / sr)
    _ax3.plot(times, wf, label="waveform", color="gray", linewidth=3)

    _ax4 = _ax3.twinx()
    _ax4.set_ylabel("frequency [Hz]")
    f0_time = np.arange(0, len(wf) + 1, 80) / sr
    _ax4.plot(f0_time, pitch, label="f0", color="green", linewidth=3)

    _fig.savefig("mr_gen/scripts/comp_mel.png")


def compute_log_power(wavef: torch.Tensor, n_fft=400, n_shift=160) -> torch.Tensor:
    num_frames = (len(wavef) - n_fft) // n_shift + 1
    log_power = torch.zeros(num_frames)
    for frame_no in range(num_frames):
        start_index = frame_no * n_shift
        log_power[frame_no] = torch.log(
            torch.sum(wavef[start_index : start_index + n_fft] ** 2)
        )
    return log_power


# from mr_gen.utils.data_analysis.data_alignment import load_wav

# import numpy as np
# waveforme, sample_rate = sf.read("mr_gen/scripts/comp.wav")
# waveforme = waveforme[48000:96000].copy()
waveforme, sample_rate = load("mr_gen/scripts/comp.wav")
print(waveforme.shape)
waveforme = waveforme[0, 48000:96000].clone()
print(waveforme.shape)


to_mel = T.MelSpectrogram(
    sample_rate=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=25, center=False
)
log_mel_specgram = torch.log(torch.clamp(to_mel(waveforme), 1e-5) * 1)
log_pw = compute_log_power(waveforme, n_fft=N_FFT, n_shift=HOP_LENGTH)

print(log_mel_specgram.shape)
print(log_pw.shape)
print(log_pw.unsqueeze(0).shape)
print(torch.cat([log_mel_specgram, log_pw.unsqueeze(0)], dim=0).shape)

# plot_spectrogram(
#     log_mel_specgram.numpy(), log_pw.numpy(), sample_rate, waveforme.numpy()
# )

# res = torch_audio_vad.load_wav("mr_gen/scripts/comp.wav")
# print(res.shape)
# print(res[:3])
# res = T.Vad(sample_rate=16000)(waveforme[48000:96000])
# print(res.shape)

# US_AUTH_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
# input(US_AUTH_TOKEN)

# vad_pipeline = Pipeline.from_pretrained(
#     "pyannote/segmentation",
#     use_auth_token=US_AUTH_TOKEN,
# )
# vad_pipeline = Pipeline.from_pretrained(
#     "pyannote/voice-activity-detection",
#     use_auth_token=US_AUTH_TOKEN,
# )
# initial_params = {
#     "onset": 0.4,
#     "offset": 0.2,
#     "min_duration_on": 0.0,
#     "min_duration_off": 0.0,
# }
# vad_pipeline.instantiate(initial_params)
# vad_pipeline.default_parameters()

# waveforme = load_wav("mr_gen/scripts/comp.wav")

waveforme_np: np.ndarray = waveforme.numpy()
waveforme_np = waveforme_np.astype(np.float64)

# calc f0
hop_length = 80 / sample_rate * 1000  # 5ms
_f0, _time = pw.dio(waveforme_np, sample_rate, frame_period=hop_length)  # type: ignore
# _f0, _time = pw.dio(waveforme_np, sample_rate, allowed_range=0.1)  # type: ignore
f0 = pw.stonemask(waveforme_np, _f0, _time, sample_rate)  # type: ignore
print(pw.dio.__code__.co_varnames[: pw.dio.__code__.co_argcount])  # type: ignore
print(pw.stonemask.__code__.co_varnames[: pw.stonemask.__code__.co_argcount])  # type: ignore
print(f0.shape)
print(_f0.shape)
print(_time.shape)
print(_time[:3])
# _f0, _time = pw.dio(noise_amp, samplerate2)
# noise_f0 = pw.stonemask(noise_amp, _f0, _time, samplerate2)
# noise_mean_f0 = np.mean(noise_f0[noise_f0 > 30])
vad = np.zeros_like(waveforme_np, dtype=np.float32)
for i in range(len(f0) - 1):
    if f0[i] > 0:
        vad[i * 80 : (i + 1) * 80] = 1.0

# output = vad_pipeline("mr_gen/scripts/comp.wav")

plot_spectrogram(
    log_mel_specgram.numpy(),
    log_pw.numpy(),
    sample_rate,
    waveforme.numpy(),
    f0,
    N_FFT,
    HOP_LENGTH,
)

# print(waveforme_np.shape)
# print(f0.shape)
# print(output.shape)
# times = np.arange(0, len(waveforme_np)) / sample_rate

# # plt.figure(figsize=(20, 4))
# # plt.plot(times, waveforme_np, label="waveform", color="gray", linewidth=3)

# fig = plt.figure(figsize=(20, 4))
# ax1 = fig.add_subplot(111)
# times = np.arange(0, len(waveforme_np)) / sample_rate
# fs = 1.0
# ln1 = ax1.plot(times, waveforme_np, label="waveform", color="gray", linewidth=3)

# ax2 = ax1.twinx()
# ln2 = ax2.plot(times[::80], f0[:-1], label="f0", color="green", linewidth=3)

# h1, l1 = ax1.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax1.legend(h1 + h2, l1 + l2, loc="lower right")

# ax1.set_xlabel("times")
# ax1.set_ylabel("Amplitude")
# ax1.grid(True)
# ax2.set_ylabel("frequency [Hz]")


# # va = np.zeros_like(times, dtype=np.float32)
# # for speech in output.get_timeline().support():
# #     # active speech between speech.start and speech.end
# #     # print(speech.start, speech.end)
# #     start = int(speech.start * sample_rate)
# #     stop = int(speech.end * sample_rate)
# #     va[start:stop] = 1.0
# # print(len(output.get_timeline().support()))
# # print(output.get_timeline().support()[:3])
# # print(sum(va))
# vad = vad * 0.6 - 0.3
# ax1.plot(times, vad, label="vad", color="orange", linewidth=3)

# fig.savefig("mr_gen/scripts/comp_pitch.png")
