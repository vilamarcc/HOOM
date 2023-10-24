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

def forlag_TF(wb=10):
    """
    Returns a first order lag transfer function.
    Used for actuator modelling.
    If cutoff frequency is not specified, default to 10 rad/s.
    """
    s = sp.symbols('s')
    forlag_TF = 1 / (s / wb + 1)
    return forlag_TF

def pid(kp=1, ki=1, kd=1, tau=1):
    """
    Returns a PID transfer function.
    Used for controller modelling.
    If tau is not specified, default to 1 s.
    """
    s = sp.symbols('s')
    pid_TF = kp * (1 + 1 / (tau * s) + tau * kd * s / (1 + tau * s / ki))
    return pid_TF

def get_closed_loop_sys(plant, pid, sensor, actuator): # se tiene que mirar el output de esta función para ver si es correcto
    """
    Returns the closed loop transfer function of a system given its plant, PID controller, sensor, and actuator.
    """
    return ctrl.feedback(ctrl.series(ctrl.series(ctrl.series(plant, pid), sensor), actuator))