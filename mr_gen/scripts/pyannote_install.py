# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained voice activity detection pipeline

import os
from pyannote.audio import Pipeline
import pyworld as pw

from mr_gen.utils.data_analysis.data_alignment import load_wav

# import numpy as np

US_AUTH_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

vad_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=US_AUTH_TOKEN,
)

waveforme = load_wav("mr_gen/scripts/comp.wav")
# calc f0
sample_rate = 16000
_f0, _time = pw.dio(waveforme, sample_rate)  # type: ignore
f0 = pw.stonemask(waveforme, _f0, _time, sample_rate)  # type: ignore
# _f0, _time = pw.dio(noise_amp, samplerate2)
# noise_f0 = pw.stonemask(noise_amp, _f0, _time, samplerate2)
# noise_mean_f0 = np.mean(noise_f0[noise_f0 > 30])

# output = pipeline("audio.wav")

# for speech in output.get_timeline().support():
#     # active speech between speech.start and speech.end
#     ...
