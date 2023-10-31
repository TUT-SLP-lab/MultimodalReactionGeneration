# import wave
from matplotlib import pyplot as plt
from mr_gen.utils.feature_extractor import FeatureExtractor
from mr_gen.utils.data_analysis.data_alignment import load_wav

SAMPLE = "data/multimodal_dialogue_formd/data004/comp.wav"
FS = 16000


def process():
    waveform = load_wav(SAMPLE)
    extractor = FeatureExtractor(
        sample_frequency=16000,
        frame_length=1024,
        frame_shift=256,
        point=True,
        high_frequency=8000,
        dither=1e-7,
    )

    ret1 = extractor.ComputeMFCC(waveform[int(1.2 * FS) : int(2.2 * FS)])
    ret2, _ = extractor.ComputeFBANK(waveform[int(1.2 * FS) : int(2.2 * FS)])
    ret3, _ = extractor.ComputeSPEC(waveform[int(1.2 * FS) : int(2.2 * FS)])

    plt.pcolor(ret1.T)
    plt.savefig("ret1A.png")

    plt.pcolor(ret2.T)
    plt.savefig("ret1B.png")

    plt.pcolor(ret3.T)
    plt.savefig("ret1C.png")


if __name__ == "__main__":
    process()
