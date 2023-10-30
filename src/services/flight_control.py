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