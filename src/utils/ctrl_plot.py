from tkinter import NO
from turtle import color
from matplotlib import style
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

def simple_latex_plot(x, y, x_label, y_label, title, filename = None, size = 12, grid = True):
    """
    Plots a simple graph using tex fonts and saves it as .eps file.
    """

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['font.size'] = size

    plt.figure()
    plt.plot(x, y, color='black', style='-', linewidth=1.5)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if grid:
        plt.grid(True)
    
    plt.tight_layout()

    if filename:
        if not filename.endswith('.eps'):
            filename += '.eps'
        plt.savefig(filename, format='eps')

    plt.show()