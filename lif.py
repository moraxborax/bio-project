import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pickle

neuron_count: int = 100
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

spike_freqs = []

V = np.full((neuron_count, time_steps), V_rest) # unit: mV

not_refractory = np.ones((neuron_count, time_steps)) 
spikes = np.zeros((neuron_count, time_steps))

I_ext = np.zeros((neuron_count, time_steps)) # external current, unit: pA
I_rec_out = np.zeros((neuron_count, time_steps)) # recurrent current, unit: pA
W = np.zeros((neuron_count, neuron_count)) # synaptic weight. no units

S_ampa_ext = np.zeros((neuron_count, time_steps))
S_ampa_rec = np.zeros((neuron_count, time_steps))

synapse_ratio = 0.3

for noise_freq in range(0, 4000, 50):
    p_noise = noise_freq / 1000 * dt

    spikes_external = rng.binomial(1, p_noise, (neuron_count, time_steps))

    V.fill(V_rest)
    not_refractory.fill(1)
    spikes.fill(0)
    I_ext.fill(0)
    I_rec_out.fill(0)
    W.fill(0)

    S_ampa_ext.fill(0)
    S_ampa_rec.fill(0)


    for t in range(1, time_steps):
        V[:, t] = V[:, t-1] + dt * (
                -(V[:, t-1] - V_rest) * g_L / C + (I_ext[:, t-1] + W @ I_rec_out[:, t-1]) * not_refractory[:, t-1] / C
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
        I_ext[:, t] = g_ampa * -(V[:, t] - V_E) * S_ampa_ext[:, t]


    # for idx in range(neuron_count):
    #     t = np.arange(time_steps)
    #     # plt.plot(t, I_ext[idx, :], label=f"I_ext of Neuron {idx}")
    #     plt.plot(t, V[idx, :], label=f"V of Neuron {idx}")

    # plt.legend()

    # plt.show()

    spike_freq = np.sum(spikes, axis=1).mean(axis=0) / time_duration * 1000 # units: Hz
    # print(f"Spike frequency: {spike_freq} Hz")
    spike_freqs.append(spike_freq)

plt.plot(range(0, 4000, 50), spike_freqs)
plt.xlabel("Noise Frequency (Hz)")
plt.ylabel("Spike Frequency (Hz)")
plt.title("Spike Frequency vs Noise Frequency")
plt.show()

spline_func = make_interp_spline(range(0, 4000, 50), spike_freqs, k=3)

with open("spline_func.pkl", "wb") as f:
    pickle.dump(spline_func, f)

