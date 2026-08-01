import pickle
import numpy as np
import matplotlib.pyplot as plt

# with open("fitted_curve.pkl", "rb") as f:
#     spline_func = pickle.load(f)

# print(spline_func)

# x = np.arange(0, 4000, 50)
# y = spline_func(x)

spike_freqs = np.load("spike_freqs.npy")
noise_freqs = np.load("noise_freqs.npy")

plt.plot(noise_freqs, spike_freqs)


noise_freq = 2000
synapse_ratio = 0.015
synapse_ratio_2 = 0.02
neuron_count = 1000
def line(x: float) -> float:
    return (x - noise_freq) / (synapse_ratio * neuron_count)

def line_2(x: float) -> float:
    return (x - noise_freq) / (synapse_ratio_2 * neuron_count)

plt.plot(noise_freqs, spike_freqs)
plt.plot(noise_freqs, line(noise_freqs), label="p=0.015")
plt.plot(noise_freqs, line_2(noise_freqs), label="p=0.02")
plt.vlines(noise_freq, 0, 40, colors="red", linestyles="dashed", label="p=0")
plt.xlim([1500, 3000])
plt.ylim([0, 40])
plt.xlabel("Noise Frequency (Hz)")
plt.ylabel("Spike Frequency (Hz)")
plt.title("Spike Frequency vs Noise Frequency")
plt.hlines(5, 0, 4000, colors="gray", linestyles="dashed")
plt.hlines(8.5, 0, 4000, colors="gray", linestyles="dashed")
plt.hlines(15, 0, 4000, colors="gray", linestyles="dashed")
plt.legend()
plt.show()