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

## ---- Unit conversion ----
        
def ft_to_m(ft_value):
    # Converts from ft to m 
    m_value = ft_value/3.2808
    return m_value

def m_to_ft(m_value):
    # Converts from ft to m 
    ft_value = m_value*3.2808
    return ft_value
        
def kt_to_ms(kt_value):
    # Converts from knots to m/s
    ms_value = kt_value*0.5144
    return ms_value

def rad_to_deg(rad_value):
    # Converts from radians to degrees
    deg_value = rad_value * 180/np.pi
    return deg_value

def deg_to_rad(deg_value):
    # Converts from radians to degrees
    rad_value = deg_value * np.pi/180
    return

def lb_to_kg(lb_value):
    # Converts from lbs to kgs
    kg_value = lb_value*0.453592
    return kg_value

def slgft2_tokgm2(slgft2_value):
    # Converts from slugs*ft^2 to kg*m^2
    kgm2_value = slgft2_value*1.35581795
    return kgm2_value

def lbft2_to_pa(lbft2_value):
    # Converts from lb/ft^2 to Pa
    pa_value = lbft2_value*47.880172
    return pa_value

def kmh_to_ms(kmh_value):
    # Converts from km/h to m/s
    ms_value = kmh_value/3.6
    return ms_value

