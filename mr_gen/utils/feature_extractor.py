import numpy as np

from typing import Tuple


class FeatureExtractor:
    """Feature Extraction (FBANK, MFCC)
    sample_frequency: Sampling frequency of input waveform [Hz]
    frame_length: frame-size [ms]
    frame_shift: Analysis interval (frame shift) [ms]
    num_mel_bins: Number of mel filter banks (= number of dimensions of FBANK features)
    num_ceps: Number of dimensions of MFCC features (including the 0th dimension)
    lifter_coef: Parameters of the liftering process
    low_frequency: Cutoff frequency for low frequency band rejection [Hz]
    high_frequency: Cutoff frequency for high frequency band rejection [Hz]
    dither: Dithering process parameters (noise strength)
    """

    def __init__(
        self,
        sample_frequency: float = 16000.0,
        frame_length: int = 25,
        frame_shift: int = 10,
        point: bool = False,
        num_mel_bins: int = 23,
        num_ceps: int = 13,
        lifter_coef: float = 22.0,
        low_frequency: float = 20.0,
        high_frequency: float = 8000.0,
        dither: float = 1.0,
    ):
        """
        Args:
            sample_frequency (float, optional):Sampling frequency of input waveform [Hz]. Defaults to 16000.
            frame_length (int, optional): frame-size [ms]. Defaults to 25.
            frame_shift (int, optional): Analysis interval (frame shift) [ms]. Defaults to 10.
            num_mel_bins (int, optional): Number of mel filter banks (= number of dimensions of FBANK features). Defaults to 23.
            num_ceps (int, optional): Number of dimensions of MFCC features (including the 0th dimension). Defaults to 13.
            lifter_coef (float, optional): Parameters of the liftering process. Defaults to 22.
            low_frequency (float, optional): Cutoff frequency for low frequency band rejection. Defaults to 20.
            high_frequency (float, optional): Cutoff frequency for high frequency band rejection. Defaults to 8000.
            dither (float, optional): Dithering process parameters (noise strength). Defaults to 1.0.
        """
        self.sample_freq = sample_frequency
        if point:
            self.frame_size = frame_length
            self.frame_shift = frame_shift
        else:
            # Convert window width from milliseconds to samples
            self.frame_size = int(sample_frequency * frame_length * 0.001)
            # Convert frame shift from milliseconds to samples
            self.frame_shift = int(sample_frequency * frame_shift * 0.001)

        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.lifter_coef = lifter_coef
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        # dithering factor
        self.dither_coef = dither

        # Number of points of FFT = Power of 2 over window width
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        # Creating a MelFilter Bank
        self.mel_filter_bank = self.MakeMelFilterBank()

        # Create a basis matrix for the Discrete Cosine Transform (DCT)
        self.dct_matrix = self.MakeDCTMatrix()

        # Create a lifter
        self.lifter = self.MakeLifter()

    def Herz2Mel(self, herz: float) -> float:
        """Convert frequency from hertz to mel"""
        return 1127.0 * np.log(1.0 + herz / 700)

    def MakeMelFilterBank(self) -> np.ndarray:
        """Creating a MelFilter Bank"""
        # メル軸での最大周波数
        mel_high_freq = self.Herz2Mel(self.high_frequency)
        # メル軸での最小周波数
        mel_low_freq = self.Herz2Mel(self.low_frequency)
        # 最小から最大周波数まで，
        # メル軸上での等間隔な周波数を得る
        mel_points = np.linspace(mel_low_freq, mel_high_freq, self.num_mel_bins + 2)

        # パワースペクトルの次元数 = FFTサイズ/2+1
        # ※Kaldiの実装ではナイキスト周波数成分(最後の+1)は
        # 捨てているが，本実装では捨てずに用いている
        dim_spectrum = int(self.fft_size / 2) + 1

        # メルフィルタバンク(フィルタの数 x スペクトルの次元数)
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            # 三角フィルタの左端，中央，右端のメル周波数
            left_mel = mel_points[m]
            center_mel = mel_points[m + 1]
            right_mel = mel_points[m + 2]
            # パワースペクトルの各ビンに対応する重みを計算する
            for n in range(dim_spectrum):
                # 各ビンに対応するヘルツ軸周波数を計算
                freq = 1.0 * n * self.sample_freq / 2 / dim_spectrum
                # メル周波数に変換
                mel = self.Herz2Mel(freq)
                # そのビンが三角フィルタの範囲に入っていれば，重みを計算
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    mel_filter_bank[m][n] = weight

        return mel_filter_bank

    def ExtractWindow(self, waveform, start_index) -> Tuple[np.ndarray, float]:
        """
        One frame of waveform data is extracted and preprocessed.
        The logarithmic power value is also calculated.
        """
        # waveformから，1フレーム分の波形を抽出する
        window = waveform[start_index : start_index + self.frame_size].copy()

        # ディザリングを行う
        # (-dither_coef～dither_coefの一様乱数を加える)
        if self.dither_coef > 0:
            window = (
                window
                + np.random.rand(self.frame_size) * (2 * self.dither_coef)
                - self.dither_coef
            )

        # 直流成分をカットする
        window = window - np.mean(window)

        # 以降の処理を行う前に，パワーを求める
        power = np.sum(window**2)
        # 対数計算時に-infが出力されないよう，フロアリング処理を行う
        if power < 1e-10:
            power = 1e-10
        # 対数をとる
        log_power = np.log(power)

        # プリエンファシス(高域強調)
        # window[i] = 1.0 * window[i] - 0.97 * window[i-1]
        window = np.convolve(window, np.array([1.0, -0.97]), mode="same")
        # numpyの畳み込みでは0番目の要素が処理されない
        # (window[i-1]が存在しないので)ため，
        # window[0-1]をwindow[0]で代用して処理する
        window[0] -= 0.97 * window[0]

        # hamming窓をかける
        # hamming[i] = 0.54 - 0.46 * np.cos(2*np.pi*i / (self.frame_size - 1))
        window *= np.hamming(self.frame_size)

        return (window, log_power)

    def ComputeFBANK(self, waveform) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mel filter bank features (FBANK)
        Output 1: fbank_features: Mel filter bank features
        Output 2: log_power: log power value (used for MFCC extraction)
        """
        # 波形データの総サンプル数
        num_samples = np.size(waveform)
        # 特徴量の総フレーム数を計算する
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        # メルフィルタバンク特徴
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        # 対数パワー(MFCC特徴を求める際に使用する)
        log_power = np.zeros(num_frames)

        # 1フレームずつ特徴量を計算する
        for frame in range(num_frames):
            # 分析の開始位置は，フレーム番号(0始まり)*フレームシフト
            start_index = frame * self.frame_shift
            # 1フレーム分の波形を抽出し，前処理を実施する．
            # また対数パワーの値も得る
            window, log_pow = self.ExtractWindow(waveform, start_index)

            # 高速フーリエ変換(FFT)を実行
            spectrum = np.fft.fft(window, n=self.fft_size)
            # FFT結果の右半分(負の周波数成分)を取り除く
            # ※Kaldiの実装ではナイキスト周波数成分(最後の+1)は捨てているが，
            # 本実装では捨てずに用いている
            spectrum = spectrum[: int(self.fft_size / 2) + 1]

            # パワースペクトルを計算する
            spectrum = np.abs(spectrum) ** 2

            # メルフィルタバンクを畳み込む
            fbank = np.dot(spectrum, self.mel_filter_bank.T)

            # 対数計算時に-infが出力されないよう，フロアリング処理を行う
            fbank[fbank < 0.1] = 0.1

            # 対数をとってfbank_featuresに加える
            fbank_features[frame] = np.log(fbank)

            # 対数パワーの値をlog_powerに加える
            log_power[frame] = log_pow

        return (fbank_features, log_power)

    def ComputeSPEC(self, waveform) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT spectrum"""
        # 波形データの総サンプル数
        num_samples = np.size(waveform)
        # 特徴量の総フレーム数を計算する
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        # メルフィルタバンク特徴
        spec_features = np.zeros((num_frames, int(self.fft_size / 2) + 1))
        # 対数パワー(MFCC特徴を求める際に使用する)
        log_power = np.zeros(num_frames)

        # 1フレームずつ特徴量を計算する
        for frame in range(num_frames):
            # 分析の開始位置は，フレーム番号(0始まり)*フレームシフト
            start_index = frame * self.frame_shift
            # 1フレーム分の波形を抽出し，前処理を実施する．
            # また対数パワーの値も得る
            window, log_pow = self.ExtractWindow(waveform, start_index)

            # 高速フーリエ変換(FFT)を実行
            spectrum = np.fft.fft(window, n=self.fft_size)
            # FFT結果の右半分(負の周波数成分)を取り除く
            # ※Kaldiの実装ではナイキスト周波数成分(最後の+1)は捨てているが，
            # 本実装では捨てずに用いている
            spectrum = spectrum[: int(self.fft_size / 2) + 1]

            spec_features[frame] = spectrum

            # 対数パワーの値をlog_powerに加える
            log_power[frame] = log_pow

        return (spec_features, log_pow)

    def MakeDCTMatrix(self) -> np.ndarray:
        """Create a basis matrix for the Discrete Cosine Transform (DCT)"""
        N = self.num_mel_bins
        # DCT基底行列 (基底数(=MFCCの次元数) x FBANKの次元数)
        dct_matrix = np.zeros((self.num_ceps, self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2 / N) * np.cos(
                    ((2.0 * np.arange(N) + 1) * k * np.pi) / (2 * N)
                )

        return dct_matrix

    def MakeLifter(self) -> np.ndarray:
        """Compute Lifter"""
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        lifter = 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)
        return lifter

    def ComputeMFCC(self, waveform) -> np.ndarray:
        """Compute MFCC"""
        # FBANKおよび対数パワーを計算する
        fbank, log_power = self.ComputeFBANK(waveform)

        # DCTの基底行列との内積により，DCTを実施する
        mfcc = np.dot(fbank, self.dct_matrix.T)

        # リフタリングを行う
        mfcc *= self.lifter

        # MFCCの0次元目を，前処理をする前の波形の対数パワーに置き換える
        mfcc[:, 0] = log_power

        return mfcc
