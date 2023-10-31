import torch
from torchaudio.functional import vad
from matplotlib import pyplot as plt

from mr_gen.utils.data_analysis.data_alignment import load_wav

waveforme = torch.tensor(load_wav("mr_gen/scripts/comp.wav"))[160000:]
# res1 = vad(
#     waveforme, sample_rate=16000, trigger_level=10, trigger_time=0.25, search_time=0.5
# )
input(1)
res2 = vad(waveforme, sample_rate=16000)
input(2)
plt.plot(waveforme.numpy(), color="blue")
# plt.plot(res1, color="red")
input(3)
plt.plot(res2.numpy(), color="green")
input(4)
plt.savefig("vad.png")
