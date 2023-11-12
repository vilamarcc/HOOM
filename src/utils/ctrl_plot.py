import matplotlib.pyplot as plt
import control as ctrl
import numpy as np


def plot_step_response(plant, T = np.linspace(0, 100, 100000)):
    """
    Plots the response to a unit step input given a plant.
    """
    time, response = ctrl.step_response(plant, T)
    plt.figure()
    plt.plot(time, response)
    plt.title('Step Response')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.grid(True)
    plt.show()