from Gon.Spectrogram import CQT as CQTGon
from time import time
import numpy as np
import torch

fs = 44100
window = 'hann'

omega = 440
t = np.arange(20*fs) / fs
y = np.zeros(t.shape)
y[int(0.5*fs): int(1.5*fs)] = np.sin(2 * np.pi * omega * t[int(0.5*fs): int(1.5*fs)])
y_tensor = torch.tensor(y, device='cuda:0').float()

dirac = np.zeros(t.shape)
dirac[int(0.5*fs)] = 1
dirac_tensor = torch.tensor(dirac, device='cuda:0').float()

factor = 3
hop_length_1 = int(44100 * 0.01)
hop_length_2 = 1

print("Stride factor:", hop_length_1 / hop_length_2)

cqt_layer_fast = CQTGon(sr=44100, hop_length=hop_length_1, pad_mode='constant', n_bins=84*factor, bins_per_octave=12*factor,
                       window=window).to('cuda:0')
cqt_layer_slow = CQTGon(sr=44100, hop_length=hop_length_2, pad_mode='constant', n_bins=84*factor, bins_per_octave=12*factor,
                       window=window).to('cuda:0')


if __name__ == '__main__':
    start = time()
    tensor_gon = cqt_layer_fast(y_tensor)
    end = time()
    print("Time fast:", round(end - start, 3), "seconds.")

    start = time()
    tensor = cqt_layer_slow(y_tensor)
    end = time()
    print("Time slow:", round(end - start, 3), "seconds.")
