from Gon.Spectrogram import CQT as CQTGon
import numpy as np
import torch
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from numpy import pi
import time

fs = 44100
time_resolution = 1e-3
window = ('gaussian', 60)

duration = 2
omega = 110
t = np.arange(duration*fs) / fs
y = np.zeros(t.shape)
duration_note = 1
shift_note = 0.5010
y[int(shift_note*fs): int((shift_note + duration_note)*fs)] = np.sin(2 * np.pi * omega * t[0: int(duration_note*fs)])
y_tensor = torch.tensor(y, device='cuda:0').float()

dirac = np.zeros(t.shape)
dirac[int(0.5*fs)] = 1
dirac_tensor = torch.tensor(dirac, device='cuda:0').float()

factor = 1
# cqt_layer_magnitude = CQTGon(sr=44100, hop_length=int(44100 * 0.01), pad_mode='constant', n_bins=84*factor, bins_per_octave=12*factor,
#                              window=window).to('cuda:0')
cqt_layer_phase = CQTGon(sr=44100, hop_length=int(44100 * time_resolution), pad_mode='constant', n_bins=84*factor, bins_per_octave=12*factor,
                         output_format='Complex', window=window).to('cuda:0')


def colorize(z, black=80, lum_max=1., saturation=1.):
    r = 20 * np.log10(np.abs(z) + 1e-8)
    r[r < -black] = -black
    arg = np.angle(z)

    h = (arg + pi) / (2 * pi) + 0.5

    l = lum_max * (1 + r / black)

    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c


if __name__ == '__main__':
    start = time.time()
    tensor_phase = cqt_layer_phase(y_tensor)
    end = time.time()
    print("Time to compute:", round(end-start, 3), "seconds")

    spectrogram_phase = tensor_phase.cpu().numpy()[0, :, :]

    spectrogram_complex = spectrogram_phase[:, :, 0] + 1j * spectrogram_phase[:, :, 1]
    spectrogram_magnitude = 20 * np.log10(np.abs(spectrogram_complex) + 1e-8)

    time_vector = duration * np.arange(spectrogram_phase.shape[1]) / spectrogram_phase.shape[1]

    freqs = cqt_layer_phase.freqs
    freqs[:] = omega

    phase_shift = np.exp(- 2 * np.pi * 1j * np.expand_dims(freqs, 1) * np.expand_dims(time_vector, 0))

    spectrogram_complex *= phase_shift

    spectrogram_complex = np.transpose(spectrogram_complex)

    plt.figure()
    plt.imshow(spectrogram_magnitude, aspect='auto', vmin=-80, vmax=0, origin='lower', cmap='hot')

    plt.figure()
    img = colorize(spectrogram_complex, black=80, lum_max=0.5)
    plt.imshow(img, aspect='auto', origin='lower')
    plt.show()

