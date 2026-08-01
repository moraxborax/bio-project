from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import optimize


neuron_count: int = 1000
dt: float = 0.01 # units: ms
time_duration: float = 1000.0 # units: ms
time_steps: int = int(time_duration / dt)

V_rest: float = -70.0 # Rest potential, unit: mV
V_reset: float = -65.0 # Reset potential, unit: mV
threshold: float = -50.0 # Threshold potential, unit: mV
t_ref: float = 2.0 # Refractory period, unit: ms
refractory_length = int(t_ref / dt) 
C: float = 500.0 # Capacitance, unit: pF
# Since the cells responsible for ampa is mostly pyramidal cells
g_L: float = 25.0 # Leakage conductance, unit: nS
g_ampa: float = 2.1 # AMPA conductance, unit: nS

rng = np.random.default_rng()



tau_ampa = 2 # unit: ms
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


V = np.full((neuron_count, time_steps), V_rest, dtype=np.float64) # unit: mV

not_refractory = np.ones((neuron_count, time_steps), dtype=np.float64) 
spikes = np.zeros((neuron_count, time_steps), dtype=np.float64)

I_ext = np.zeros((neuron_count, time_steps), dtype=np.float64) # external current, unit: pA
I_rec = np.zeros((neuron_count, time_steps), dtype=np.float64) # recurrent current, unit: pA
# W = np.zeros((neuron_count, neuron_count)) # synaptic weight. no units

S_ampa_ext = np.zeros((neuron_count, time_steps), dtype=np.float64)
S_ampa_rec = np.zeros((neuron_count, time_steps), dtype=np.float64)

synapse_ratio = 0.015

noise_freq = 2000
p_noise = noise_freq / 1000 * dt

spikes_external = rng.binomial(1, p_noise, (neuron_count, time_steps)).astype(np.float64)

W = rng.binomial(1, synapse_ratio, (neuron_count, neuron_count)).astype(np.float64)

# W = scipy.sparse.random(neuron_count, neuron_count, density=synapse_ratio, format="csr", rng=rng)

for t in range(1, time_steps):
    # print(f"Current time step: {t}")
    if t % 100 == 0:
        print(f"Current time step: {t}")

    V[:, t] = V[:, t-1] + dt * (
            -(V[:, t-1] - V_rest) * g_L / C + (I_ext[:, t-1] + I_rec[:, t-1]) * not_refractory[:, t-1] / C
        )
    spikes[V[:, t] > threshold, t] = 1

    V[spikes[:, t] == 1, t-1] = 0 # add a spike

    V[spikes[:, t] == 1, t] = V_reset # reset the neuron

    # refractory_steps = np.clip((t+refractory_length), a_min=None, a_max=time_steps)
    refractory_steps = np.minimum(t+refractory_length, time_steps)
    not_refractory[spikes[:, t] == 1, t:refractory_steps] = 0


    S_ampa_ext[:, t] = S_ampa_ext[:, t-1] -S_ampa_ext[:, t-1] / tau_ampa * dt + spikes_external[:, t-1]
    I_ext[:, t] = g_ampa * -(V[:, t] - V_E) * S_ampa_ext[:, t]

    S_ampa_rec[:, t] = S_ampa_rec[:, t-1] -S_ampa_rec[:, t-1] / tau_ampa * dt + spikes[:, t-1]
    I_rec[:, t] = g_ampa * -(V[:, t] - V_E) * (W @ S_ampa_rec[:, t])


# for idx in range(neuron_count):
#     t = np.arange(time_steps)
#     # plt.plot(t, I_ext[idx, :], label=f"I_ext of Neuron {idx}")
#     plt.plot(t, V[idx, :], label=f"V of Neuron {idx}")

# plt.legend()

# plt.show()

# spike_freq = np.sum(spikes, axis=1).mean(axis=0) / time_duration * 1000 # units: Hz
    
# print(f"Spike frequency: {spike_freq} Hz")

window_size = int(50 / dt)
shift_size = int(1 / dt)
x = np.arange(0, time_steps - window_size, shift_size)
spike_freqs = np.zeros_like(x, dtype=np.float64)

for i, t in enumerate(x):
    spike_freqs[i] = np.sum(spikes[:, t:t+window_size], axis=1).mean(axis=0) / 50.0 * 1000 # units: Hz

plt.plot(x, spike_freqs, label="Spike frequency")


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
plt.legend()
plt.show()