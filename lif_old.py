"""
Leaky integrate and fire model

The model is a simple model of a neuron that can be used to simulate the behavior of a neuron.

C dV/dt = -g_L (V - V_rest) + I_ext

where C is the capacitance, g_L is the leak conductance, V_rest is the rest potential, and I_ext is the external current.

equivalently:

dV/dt = -(V-V_rest)/tau + I_ext/C
where tau = RC = C/g_L
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class LIF:
    def __init__(
        self,
        C: float,
        g_L: float,
        V_rest: float = -70,
        V_reset: float = -65,
        threshold: float = -50,
        t_ref: float = 2,
        dt: float = 0.01,
    ):
        """
        Initialize the LIF model

        Args:
            C: capacitance: unit: pF
            g_L: leak conductance, unit: nS
            V_rest: rest potential, unit: mV
            V_reset: reset potential, unit: mV
            threshold: threshold potential, unit: mV
            t_ref: refractory period, unit: ms
            dt: time step, unit: ms
        """
        self.C = C
        self.g_L = g_L
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.threshold = threshold
        self.dt = dt
        self.ref_cycles = int(t_ref / dt)
        self.activations = []
        self.V = self.V_rest
        self.curr_ref_cycle = 0
        self.refractory = False

    def reset(self):
        """
        Reset the LIF model
        """
        self.V = self.V_rest
        self.curr_ref_cycle = 0
        self.refractory = False
        self.activations = []

    


    def step(self, I_ext: float):
        """
        Step the LIF model
        Args:
            I_ext: external current, unit: pA
        Returns:
            V: voltage, unit: mV
            activations: list of booleans, True if the neuron fired, False otherwise
        """
        V_prev = self.V
        # Received current
        I_ttl = 0 if self.refractory else I_ext
        self.V = V_prev + self.dt * (
            -(V_prev - self.V_rest) * self.g_L / self.C + I_ttl / self.C
        )
        if self.V > self.threshold:
            self.V = self.V_reset
            self.refractory = True
            self.activations.append(True)

        else:
            self.activations.append(False)

        if self.refractory:
            self.curr_ref_cycle += 1
            if self.curr_ref_cycle >= self.ref_cycles:
                self.refractory = False
                self.curr_ref_cycle = 0


dt = 0.01
V_rest = -70

lif = LIF(C=200, g_L=25, V_rest=-70, V_reset=-60, threshold=-50, dt=dt)

t = np.arange(0, 200, dt)

I_ext = -1000
I_array = np.zeros_like(t)
start = 20
end = 50
start_idx = int(start / dt)
end_idx = int(end / dt)
I_array[start_idx:end_idx] = I_ext
Vs = []
for t_i in range(len(t)):
    lif.step(I_ext=I_array[t_i])
    if lif.activations[-1]:
        Vs.append(0)
    else:
        Vs.append(lif.V)
plt.plot(t, I_array / 1000 + V_rest - 10, label = "I_ext / 1000 + V_rest - 10")
# plt.plot(t, np.array(lif.activations) + V_rest )
plt.plot(t, np.array(Vs), label = "V")

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("LIF Model")

plt.legend()
plt.savefig("lif_old.png", dpi=300)
plt.show()

    

