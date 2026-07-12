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
    def __init__(self, C: float, g_L: float, V_rest: float=-70, V_reset: float=-65, threshold: float=-50, dt: float=0.01):
        """
        Initialize the LIF model

        Args:
            C: capacitance: unit: pF
            g_L: leak conductance, unit: nS
            V_rest: rest potential, unit: mV
            I_ext: external current, unit: pA
            threshold: threshold potential, unit: mV
            dt: time step, unit: ms
        """
        self.C = C
        self.g_L = g_L
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.threshold = threshold
        self.dt = dt
        self.activations = []
        self.V = self.V_rest

    def reset(self):
        """
        Reset the LIF model
        """
        self.V = self.V_rest
        
        self.activations = []
        
    
    def step(self, I_ext: float):
        """
        Step the LIF model
        """
        V_prev = self.V
        self.V = V_prev + self.dt *(-(V_prev-self.V_rest) * self.g_L/self.C + I_ext/self.C)
        if self.V > self.threshold:
            
            self.V = self.V_reset
            self.activations.append(True)
        else:
            self.activations.append(False)




