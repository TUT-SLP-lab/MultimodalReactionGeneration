import torch
import torchaudio.transforms as T
import torchaudio._backend.soundfile_backend as torchaudio_sf

DEFO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioPreprocessor:
    def __init__(self, args):
        self.args = args
        self.nfft = args.nfft
        self.shift = args.shift
        self.nmels = args.nmels
        self.sample_rate = args.sample_rate
        self.delta_order = args.delta_order

        self.fbank = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.nfft,
            hop_length=self.shift,
            n_mels=self.nmels,
            center=False,
        )
        self.log = lambda x: torch.log(torch.clamp(x, 1e-6) * 1)

    def __call__(
        self, wavepath: str, start: int, end: int, device: torch.device = DEFO_DEVICE
    ) -> torch.Tensor:
        waveforme, sample_rate = torchaudio_sf.load(wavepath, start, end - start)
        if sample_rate != self.sample_rate:
            raise ValueError("sample_rate must be same as --sample-rate")

        waveforme = waveforme.to(device)
        fbank = self.fbank(waveforme)
        fbank = self.log(fbank + 1e-6)
        power = self.compute_log_power(waveforme, device)
        fbank = torch.cat([fbank, power.unsqueeze(0)], dim=0).T

        fbank_with_delta = self.compute_delta(fbank)
        return fbank_with_delta

    def compute_log_power(
        self, waveform: torch.Tensor, device: torch.device = DEFO_DEVICE
    ) -> torch.Tensor:
        num_frames = (len(waveform) - self.nfft) // self.shift + 1
        log_power = torch.zeros(num_frames, device=device)
        for frame_no in range(num_frames):
            start_index = frame_no * self.shift
            log_power[frame_no] = torch.log(
                torch.sum(waveform[start_index : start_index + self.nfft] ** 2)
            )
        return log_power

    def compute_delta(self, fbank: torch.Tensor) -> torch.Tensor:
        if self.delta_order == 0:
            return fbank

        delta1 = fbank[1:] - fbank[:-1]
        if self.delta_order == 1:
            return torch.cat([fbank[1:], delta1], dim=1)

        delta2 = delta1[1:] - delta1[:-1]
        if self.delta_order == 2:
            return torch.cat([fbank[2:], delta1[1:], delta2], dim=1)

        raise ValueError("delta_order must be 0, 1 or 2")
