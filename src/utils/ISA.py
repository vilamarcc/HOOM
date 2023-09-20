import math
import numpy as np

# ---- Atmosphere ----
        
def  ISA(h):
    """
    Calculates the temperature, pressure, and density of the atmosphere at a given altitude 'h'.

    Args:
        h (float): The altitude in meters.

    Returns:
        Tuple: A tuple containing three floats. The first float is the temperature 'T' in Kelvin. The second float is the pressure 'P' in Pascals. The third float is the density 'rho' in kg/m^3.
    """

    # ISA model. h: meters.
    R = 287.04   # [m^2/(K*s^2)]
    P_0 = 101325  # [Pa]
    T_0 = 288.15  # [K]
    g_0 = 9.80665 # [m/s^2]
    
    if h < 11000:
        T = T_0 - 6.5*h/1000
        P = P_0*(1 - 0.0065*h/T_0)**5.2561
    else:
        T = T_0 - 6.5*11000/1000
        P_11 = P_0*(1 - 0.0065*11000/T_0)**5.2561
        P = P_11*math.exp(-g_0/(R*T)*(h-11000))

    rho = P/(R*T)

    return T, P, rho      