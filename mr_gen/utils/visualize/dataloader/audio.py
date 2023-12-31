import torch
import torchaudio.transforms as T
import torchaudio._backend.soundfile_backend as torchaudio_sf


class AudioPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nfft = cfg.nfft
        self.shift = cfg.shift
        self.nmels = cfg.nmels
        self.sample_rate = cfg.sample_rate
        self.delta_order = cfg.delta_order

        self.fbank = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.nfft,
            hop_length=self.shift,
            n_mels=self.nmels,
            center=False,
        )
        self.log = lambda x: torch.log(torch.clamp(x, 1e-6) * 1)

    def __call__(self, wavepath: str, start: int, end: int) -> torch.Tensor:
        length = end if end == -1 else end - start
        waveforme, sample_rate = torchaudio_sf.load(wavepath, start, length)
        if sample_rate != self.sample_rate:
            raise ValueError("sample_rate must be same as --sample-rate")

        fbank = self.fbank(waveforme[0])  # pylint: disable=not-callable
        fbank = self.log(torch.clamp(fbank, 1e-10))
        power = self.compute_log_power(waveforme[0])
        fbank = torch.cat([fbank, power.unsqueeze(0)], dim=0).T.to(torch.float32)

        fbank_with_delta = self.compute_delta(fbank)
        msg = f"start: {start}, end: {end}, stride: {1}"
        assert len(fbank_with_delta) != 0, msg

        fbank_with_delta = fbank_with_delta.unsqueeze(0)

        return ((fbank_with_delta, [fbank_with_delta.shape[1]]), wavepath)

    def compute_log_power(self, waveform: torch.Tensor) -> torch.Tensor:
        num_frames = (len(waveform) - self.nfft) // self.shift + 1
        log_power = torch.zeros(num_frames)
        for frame_no in range(num_frames):
            start_index = frame_no * self.shift
            stop_index = start_index + self.nfft

            power = torch.sum(torch.pow(waveform[start_index:stop_index], 2))
            power = torch.clamp(power, 1e-10)

            log_power[frame_no] = torch.log(power)

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
