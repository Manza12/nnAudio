import numpy as np
import matplotlib.pyplot as plt

interpolation_type = 'both'

N = 2**5
n = np.arange(N)

nu = 0.1
x = np.sin(2 * np.pi * nu * n)

# Fig 1
plt.figure()
plt.plot(n, x)

X = np.fft.fft(x)
f = n / N

# Fig 2
plt.figure()
plt.plot(f, np.abs(X))

R = 4
N_R = N * R

f_R = np.arange(N_R, dtype=np.double) / N_R
if interpolation_type == 'numeric':
    kernel_time = np.ones(N) / N
    kernel_freq = np.fft.fft(kernel_time, N_R)
    kernel_freq = np.roll(kernel_freq, N_R // 2)
elif interpolation_type == 'analytic':
    kernel_freq = np.sinc((f_R - 0.5) * N)
elif interpolation_type == 'both':
    kernel_time = np.ones(N) / N
    kernel_freq_num = np.fft.fft(kernel_time, N_R)
    kernel_freq_num = np.roll(kernel_freq_num, N_R // 2)

    kernel_freq_ana = np.sinc(f_R * N - N / 2)
    kernel_freq = kernel_freq_ana
else:
    raise Exception()

# Fig 3
plt.figure()
if interpolation_type == 'numeric':
    plt.plot(f_R, np.abs(kernel_freq))
elif interpolation_type == 'analytic':
    plt.plot(f_R, np.abs(kernel_freq))
elif interpolation_type == 'both':
    plt.plot(f_R, np.abs(kernel_freq_ana))
    plt.plot(f_R, np.abs(kernel_freq_num))

X_up = np.zeros(N_R, dtype=complex)
X_up[0::R] = X

X_conv = np.convolve(X_up, kernel_freq, 'same')
X_conv = np.roll(X_conv, -1)

# Fig 4
plt.figure()
plt.plot(f_R, np.abs(X_conv))


# Fig 5
plt.figure()
plt.scatter(f_R, np.abs(X_conv))

plt.scatter(f, np.abs(X), marker='x')

# Fig 6
plt.figure()
plt.plot(f_R, np.abs(X_conv))
plt.plot(f_R, np.abs(np.fft.fft(x, N_R)))


if __name__ == '__main__':
    plt.show()
