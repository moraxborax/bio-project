import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

neuron_count: int = 200
dt: float = 0.01 # units: ms
time_duration: float = 1000.0 # units: ms
time_steps: int = int(time_duration / dt)

V_rest: float = -70.0 # Rest potential, unit: mV
V_reset: float = -60.0 # Reset potential, unit: mV
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

V = np.full((neuron_count, time_steps), V_rest, dtype=np.float32) # unit: mV

not_refractory = np.ones((neuron_count, time_steps), dtype=np.float32) 
spikes = np.zeros((neuron_count, time_steps), dtype=np.float32)

I_ext = np.zeros((neuron_count, time_steps), dtype=np.float32) # external current, unit: pA
I_rec_out = np.zeros((neuron_count, time_steps), dtype=np.float32) # recurrent current, unit: pA
# W = np.zeros((neuron_count, neuron_count)) # synaptic weight. no units

S_ampa_ext = np.zeros((neuron_count, time_steps), dtype=np.float32)
S_ampa_rec = np.zeros((neuron_count, time_steps), dtype=np.float32)

synapse_ratio = 0

spikes_external = np.zeros((neuron_count, time_steps), dtype=np.float32)

noise_freqs = range(0, 4000, 100)
spike_freqs = np.zeros_like(noise_freqs, dtype=np.float32)
for idx, noise_freq in enumerate(noise_freqs):

    print(f"Current noise frequency: {noise_freq} Hz")
    p_noise = noise_freq / 1000 * dt

    spikes_external[:] = rng.binomial(1, p_noise, (neuron_count, time_steps))

    V.fill(V_rest)
    not_refractory.fill(1)
    spikes.fill(0)
    I_ext.fill(0)
    

    S_ampa_ext.fill(0)
    # S_ampa_rec.fill(0)


    for t in trange(1, time_steps):
        
        V[:, t] = V[:, t-1] + dt * (
                -(V[:, t-1] - V_rest) * g_L / C + I_ext[:, t-1] * not_refractory[:, t-1] / C
            )
        spikes[V[:, t] > threshold, t] = 1

        V[spikes[:, t] == 1, t-1] = 0 # add a spike

        V[spikes[:, t] == 1, t] = V_reset # reset the neuron

        # refractory_steps = np.clip((t+refractory_length), a_min=None, a_max=time_steps)
        refractory_steps = np.minimum(t+refractory_length, time_steps)
        not_refractory[spikes[:, t] == 1, t:refractory_steps] = 0

        

        S_ampa_ext[:, t] = S_ampa_ext[:, t-1] -S_ampa_ext[:, t-1] / tau_ampa * dt + spikes_external[:, t-1]
        I_ext[:, t] = g_ampa * -(V[:, t] - V_E) * S_ampa_ext[:, t]

        

    spike_freq = np.sum(spikes, axis=1).mean(axis=0) / time_duration * 1000 # units: Hz
    # print(f"Spike frequency: {spike_freq} Hz")
    spike_freqs[idx] = spike_freq

plt.plot(noise_freqs, spike_freqs)
plt.xlabel("Noise Frequency (Hz)")
plt.ylabel("Spike Frequency (Hz)")
plt.title("Spike Frequency vs Noise Frequency")
plt.savefig("spike_freq_vs_noise_freq.png", dpi=300)
plt.show()

np.save("spike_freqs.npy", spike_freqs)
np.save("noise_freqs.npy", noise_freqs)


