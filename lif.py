import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


neuron_count: int = 1000
dt: float = 0.01  # units: ms
time_duration: float = 1000.0  # units: ms
time_steps: int = int(time_duration / dt)

V_rest: float = -70.0  # Rest potential, unit: mV
V_reset: float = -60.0  # Reset potential, unit: mV
threshold: float = -50.0  # Threshold potential, unit: mV
t_ref: float = 2.0  # Refractory period, unit: ms
refractory_length = int(t_ref / dt)
C: float = 500.0  # Capacitance, unit: pF
# Since the cells responsible for ampa is mostly pyramidal cells
g_L: float = 25.0  # Leakage conductance, unit: nS
g_ampa: float = 2.1  # AMPA conductance, unit: nS

rng = np.random.default_rng()


tau_ampa = 2  # unit: ms
"""
    Ampa current
    Should follow the following equations:
    I_ampa = g_ampa * (V - V_E) * s_ampa
    where V_E = 0
    s is the gate here, it is between 0 and 1.
    and ds/dt = -s/tau_ampa + delta(t - t_spike)
    so basically it spikes when there is a spike and then decays exponentially to 0
    tau_ampa is about 2ms.
    """
V_E = 0

# noise_freq = 2200 # unit: Hz


V = np.full((neuron_count, time_steps), V_rest, dtype=np.float32)  # unit: mV

not_refractory = np.ones((neuron_count, time_steps), dtype=np.float32)
spikes = np.zeros((neuron_count, time_steps), dtype=np.float32)

I_ext = np.zeros(
    (neuron_count, time_steps), dtype=np.float32
)  # external current, unit: pA
I_rec = np.zeros(
    (neuron_count, time_steps), dtype=np.float32
)  # recurrent current, unit: pA
# W = np.zeros((neuron_count, neuron_count)) # synaptic weight. no units

S_ampa_ext = np.zeros((neuron_count, time_steps), dtype=np.float32)
S_ampa_rec = np.zeros((neuron_count, time_steps), dtype=np.float32)

synapse_ratio = 0.02

noise_freq = 2100
p_noise = noise_freq / 1000 * dt

spikes_external = rng.binomial(1, p_noise, (neuron_count, time_steps))

W = rng.binomial(1, synapse_ratio, (neuron_count, neuron_count)).astype(np.float32)
# W = np.zeros((neuron_count, neuron_count), dtype=np.float32)
# W = scipy.sparse.csr_matrix(rng.random((neuron_count, neuron_count)) < synapse_ratio, dtype=np.float32)


for t in trange(1, time_steps):
    # print(f"Current time step: {t}")

    V[:, t] = V[:, t - 1] + dt * (
        -(V[:, t - 1] - V_rest) * g_L / C
        + (I_ext[:, t - 1] + I_rec[:, t - 1]) * not_refractory[:, t - 1] / C
    )
    spikes[V[:, t] > threshold, t] = 1

    V[spikes[:, t] == 1, t - 1] = 0  # add a spike

    V[spikes[:, t] == 1, t] = V_reset  # reset the neuron

    # refractory_steps = np.clip((t+refractory_length), a_min=None, a_max=time_steps)
    refractory_steps = np.minimum(t + refractory_length, time_steps)
    not_refractory[spikes[:, t] == 1, t:refractory_steps] = 0

    S_ampa_ext[:, t] = (
        S_ampa_ext[:, t - 1]
        - S_ampa_ext[:, t - 1] / tau_ampa * dt
        + spikes_external[:, t - 1]
    )
    I_ext[:, t] = g_ampa * -(V[:, t] - V_E) * S_ampa_ext[:, t]

    S_ampa_rec[:, t] = (
        S_ampa_rec[:, t - 1] - S_ampa_rec[:, t - 1] / tau_ampa * dt + spikes[:, t - 1]
    )
    I_rec[:, t] = g_ampa * -(V[:, t] - V_E) * (W @ S_ampa_rec[:, t])

fig, axs = plt.subplots(3, 1, figsize=(6, 10))

axs[0].set_title("V of neurons 0-4")
axs[0].set_xlabel("Time (ms)")
axs[0].set_ylabel("Voltage (mV)")
axs[1].set_title("I_ext of neurons 0-4")
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("Current (pA)")
axs[2].set_title("Total spikes")
axs[2].set_xlabel("Time (ms)")
axs[2].set_ylabel("neuron")

fig.suptitle(
    f"LIF Model with noise frequency {noise_freq} Hz and synapse ratio {synapse_ratio}"
)

for idx in range(5):
    t = np.arange(time_steps)
    # plt.plot(t, I_ext[idx, :], label=f"I_ext of Neuron {idx}")
    axs[0].plot(t, V[idx, :], label=f"V of Neuron {idx}")
    axs[1].plot(t, I_ext[idx, :], label=f"I_ext of Neuron {idx}")


axs[2].scatter(np.where(spikes == 1)[1], np.where(spikes == 1)[0], s=0.01)

axs[0].legend()
axs[1].legend()
plt.savefig(f"lif_{noise_freq}Hz_synapse{synapse_ratio}.png", dpi=300)
plt.show()

# spike_freq = np.sum(spikes, axis=1).mean(axis=0) / time_duration * 1000 # units: Hz

# print(f"Spike frequency: {spike_freq} Hz")

# window_size = int(50 / dt)
# shift_size = int(1 / dt)
# x = np.arange(0, time_steps - window_size, shift_size)
# spike_freqs = np.zeros_like(x, dtype=np.float64)

# for i, t in enumerate(x):
#     spike_freqs[i] = np.sum(spikes[:, t:t+window_size], axis=1).mean(axis=0) / 50.0 * 1000 # units: Hz

# plt.plot(x, spike_freqs, label="Spike frequency")


# It should follow this relationship:
# y = f(x)
# and
# x = z + p * N * y

# where x is the input frequency,
# y is the output frequency, z is the noise frequency,
# N is the number of neurons and p is the connection ratio.

# so we can solve:
# y = (x - z) / (p * N)
# this is a line with slope 1/(p*N) and x-intercept z

# def line(x: float) -> float:
#     return (x - noise_freq) / (synapse_ratio * neuron_count)

# with open("fitted_curve.pkl", "rb") as f:
#     fitted_curve: Callable[[float], float] = pickle.load(f)

# solution = optimize.root_scalar(lambda x: line(x) - fitted_curve(x), bracket=[0, 4000])

# f_in = solution.root
# f_out = line(f_in)
# plt.plot(np.full_like(spike_freqs, f_out), label="expected frequency")
