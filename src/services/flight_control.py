import sympy as sp
import control as ctrl
import numpy as np
import math

def pade_TF(lag, order=1):
    """
    Returns a Padé approximation given a lag and order.
    Used for sensor modelling.
    """
    base_TF = ctrl.tf([1], [lag, 1])
    pade_TF = ctrl.pade(base_TF, order)
    return pade_TF

def forlag_TF(wb=10.0):
    """
    Returns a first order lag transfer function.
    Used for actuator modelling.
    If cutoff frequency is not specified, default to 10 rad/s.
    """
    s = sp.symbols('s')
    forlag_TF = 1 / (s / wb + 1)
    return forlag_TF

def pid(kp=1.0, ki=1.0, kd=1.0, tau=1.0):
    """
    Returns a PID transfer function.
    Used for controller modelling.
    If tau is not specified, default to 1 s.
    """
    s = sp.symbols('s')
    pid_TF = kp * (1 + 1 / (tau * s) + tau * kd * s / (1 + tau * s / ki))
    return pid_TF

def getKdl(G):
    """
    Returns the direct link constant so that 1 degree of pilot stick
    deflection produces a change of 1 degree in pitch angle.
    """
    K_dl = 1 / G['K']
    return K_dl

def get_closed_loop_sys(plant, sensor, controller, actuator): # se tiene que mirar el output de esta función para ver si es correcto
    """
    Returns the closed loop transfer function of a system given its plant, controller, sensor and actuator.
    """
    G_closed_loop = (plant * pid * controller * actuator) / (1 + plant * sensor * controller * actuator)

    return G_closed_loop

class TimeVaryingStateSpace(ctrl.StateSpace):
    """
    This class represents a time-varying state-space model.

    It extends the StateSpace class from the Python Control Systems Library.

    Attributes:
    A_func (function): A function that returns the system matrix A at a given time.
    B_func (function): A function that returns the input matrix B at a given time.
    dt (float): The time step for the simulation.

    Methods:
    update(t): Updates the system matrices A and B at the given time t.
    step(U, T, X0): Simulates the system's response over time.
    """

    def __init__(self, A, B, C, D, dt):
        """
        Initializes the TimeVaryingStateSpace class with the given system matrices and time step.

        Parameters:
        A (function): A function that returns the system matrix A at a given time.
        B (function): A function that returns the input matrix B at a given time.
        C (numpy.ndarray): The output matrix.
        D (numpy.ndarray): The direct transmission matrix.
        dt (float): The time step for the simulation.
        """
        super().__init__(A(0), B(0), C, D, dt)
        self.A_func = A
        self.B_func = B
        self.dt = dt

    def update(self, t):
        """
        Updates the system matrices A and B at the given time t.

        Parameters:
        t (float): The current time.
        """
        self.A = self.A_func(t)
        self.B = self.B_func(t)

    def step(self, U=0, T=None, X0=0):
        """
        Simulates the system's response over time.

        Parameters:
        U (numpy.ndarray or float, optional): The input to the system. Defaults to 0.
        T (float, optional): The final time for the simulation. If not provided, the simulation runs indefinitely.
        X0 (numpy.ndarray or float, optional): The initial state of the system. Defaults to 0.

        Returns:
        T (numpy.ndarray): The time points for the simulation.
        X (numpy.ndarray): The state of the system at each time point.
        """
        T = np.arange(0, T, self.dt)
        X = np.zeros((len(T), self.A.shape[0]))
        X[0] = X0
        for i in range(1, len(T)):
            self.update(T[i])
            X[i] = X[i-1] + self.dt * (self.A @ X[i-1] + self.B @ U)
        return T, X