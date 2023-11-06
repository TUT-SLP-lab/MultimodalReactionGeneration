import time
import pickle
import soundfile as sf
import numpy as np
import torch
import torchaudio
import torchaudio._backend.soundfile_backend as torchaudio_sf
import torchaudio.transforms as T
import wave
from dfcon import Directory, FileFilter

PATH = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features"


def task():
    ffilter = FileFilter().include_extention("head")
    d = Directory(PATH).build_structure(ffilter)
    path_list = d.get_file_path(serialize=True)

    start = time.time()
    for path in path_list:
        with open(path, "rb") as f:
            pickle.load(f)
    end = time.time()

    print("file num: ", len(path_list))
    print("elapsed time: ", end - start, " [sec]")
    print("elapsed time: ", (end - start) / len(path_list) * 1000, " [ms]")


def task2():
    path = "/home/MultimodalReactionGeneration/data/test/face.head"
    # path = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features/data004/comp/comp_16614.head"
    start = time.time()
    for _ in range(1000):
        with open(path, "rb") as f:
            pickle.load(f)
    end = time.time()

    print("iteration: ", 1000)
    print("elapsed time: ", end - start, " [sec]")
    print("elapsed time: ", (end - start), " [ms]")


def binary2float(frames, length, sampwidth):
    if sampwidth == 1:
        data = np.frombuffer(frames, dtype=np.uint8)
        data = data - 128
    elif sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16)
    elif sampwidth == 3:
        a8 = np.frombuffer(frames, dtype=np.uint8)
        tmp = np.empty([length, 4], dtype=np.uint8)
        tmp[:, :sampwidth] = a8.reshape(-1, sampwidth)
        tmp[:, sampwidth:] = (tmp[:, sampwidth - 1 : sampwidth] >> 7) * 255
        data = tmp.view("int32")[:, 0]
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32)
    else:
        raise Exception("sampwidth {} is not supported.".format(sampwidth))
    data = data.astype(float) / (2 ** (8 * sampwidth - 1))  # Normalize (int to float)
    return data


def read_wave(file_name, start=0, end=0):
    file = wave.open(file_name, "rb")  # open file
    sampwidth = file.getsampwidth()
    nframes = file.getnframes()
    file.setpos(start)
    if end == 0:
        length = nframes - start
    else:
        length = end - start + 1
    frames = file.readframes(length)
    file.close()  # close file
    return binary2float(frames, length, sampwidth)


N_FFT = 400
HOP_LENGTH = 160


def compute_log_power(wavef: torch.Tensor, n_fft=400, n_shift=160) -> torch.Tensor:
    num_frames = (len(wavef) - n_fft) // n_shift + 1
    log_power = torch.zeros(num_frames)
    for frame_no in range(num_frames):
        start_index = frame_no * n_shift
        log_power[frame_no] = torch.log(
            torch.sum(wavef[start_index : start_index + n_fft] ** 2)
        )
    return log_power


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def task3():
    path = "/home/MultimodalReactionGeneration/data/multimodal_dialogue_features/data004/comp.wav"
    start = time.time()
    for _ in range(1000):
        # with wave.open(path, "rb") as f:
        wavf, sr = torchaudio_sf.load(path, frame_offset=4800000, num_frames=48000)
        to_mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=25
        )
        log_mel_specgram = torch.log(torch.clamp(to_mel(wavf[0]), 1e-5) * 1)
        log_pw = compute_log_power(wavf[0], n_fft=N_FFT, n_shift=HOP_LENGTH)
        # torch.tensor(read_wave(path, 4800000, 4800000 + 48000))
    end = time.time()

    print("iteration: ", 1000)
    print("elapsed time: ", end - start, " [sec]")
    print("elapsed time: ", (end - start), " [ms]")


if __name__ == "__main__":
    task2()
