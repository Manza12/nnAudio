from Gon.Spectrogram import CQT as CQTGon
from nnAudio.Spectrogram import CQT
import numpy as np
import torch
import matplotlib.pyplot as plt

fs = 44100
window = 'hann'

omega = 440
t = np.arange(2*fs) / fs
y = np.zeros(t.shape)
y[int(0.5*fs): int(1.5*fs)] = np.sin(2 * np.pi * omega * t[int(0.5*fs): int(1.5*fs)])
y_tensor = torch.tensor(y, device='cuda:0').float()

dirac = np.zeros(t.shape)
dirac[int(0.5*fs)] = 1
dirac_tensor = torch.tensor(dirac, device='cuda:0').float()

factor = 3
cqt_layer_gon = CQTGon(sr=44100, hop_length=int(44100 * 0.01), pad_mode='constant', n_bins=84*factor, bins_per_octave=12*factor,
                       window=window).to('cuda:0')
cqt_layer = CQT(sr=44100, hop_length=int(44100 * 0.01), pad_mode='constant', n_bins=84*factor, bins_per_octave=12*factor, window=window).to('cuda:0')


if __name__ == '__main__':
    tensor_gon = cqt_layer_gon(dirac_tensor)
    tensor = cqt_layer_gon(y_tensor)

    spectrogram_gon = 20 * np.log10(tensor_gon.cpu().numpy()[0, :, :] + 1e-8)
    spectrogram = 20 * np.log10(tensor.cpu().numpy()[0, :, :] + 1e-8)

    plt.figure()
    plt.imshow(spectrogram_gon, aspect='auto', vmin=-80, vmax=0, origin='lower', cmap='hot')

    plt.figure()
    plt.plot(spectrogram_gon[10, :])

    plt.figure()
    plt.plot(spectrogram[:, 100])

    plt.figure()
    plt.imshow(spectrogram, aspect='auto', vmin=-80, vmax=0, origin='lower', cmap='hot')

    plt.show()

    print(tensor_gon.max())
    print(tensor.max())
