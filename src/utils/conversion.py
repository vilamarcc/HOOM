import numpy as np
from sympy import symbols, Matrix, inverse_laplace_transform
from sympy.abc import s, t

## ---- Unit conversion ----

def ft_to_m(ft_value):
    """ Converts from ft to m 
    """
    m_value = ft_value/3.2808
    return m_value

def m_to_ft(m_value):
    """ Converts from ft to m 
    """
    ft_value = m_value*3.2808
    return ft_value
        
def kt_to_ms(kt_value):
    """ Converts from knots to m/s
    """
    ms_value = kt_value*0.5144
    return ms_value

def rad_to_deg(rad_value):
    """ Converts from radians to degrees
    """
    deg_value = rad_value*180.0/np.pi
    return deg_value

def deg_to_rad(deg_value):
    """ Converts from radians to degrees
    """
    rad_value = deg_value*np.pi/180.0
    return rad_value

def lb_to_kg(lb_value):
    """ Converts from lbs to kgs
    """
    kg_value = lb_value*0.453592
    return kg_value

def slgft2_tokgm2(slgft2_value):
    """ Converts from slugs*ft^2 to kg*m^2
    """
    kgm2_value = slgft2_value*1.35581795
    return kgm2_value

def lbft2_to_pa(lbft2_value):
    """ Converts from lb/ft^2 to Pa
    """
    pa_value = lbft2_value*47.880172
    return pa_value

def kmh_to_ms(kmh_value):
    """ Converts from km/h to m/s
    """
    ms_value = kmh_value/3.6
    return ms_value


## ---- Time conversion ----

def matrix_inverse_laplace_transform(M):
    """
    This function performs the inverse Laplace transform on a matrix.

    Parameters:
    M (sympy.Matrix): A sympy Matrix object in the Laplace domain.

    Returns:
    sympy.Matrix: The input matrix transformed to the time domain using the inverse Laplace transform.
    """
    return Matrix(M.shape[0], M.shape[1], lambda i, j: inverse_laplace_transform(M[i, j], s, t))


def to_time_domain(M):
    """
    This function converts a matrix from the Laplace domain to the time domain.

    Parameters:
    M (sympy.Matrix): A sympy Matrix object in the Laplace domain.

    Returns:
    sympy.Matrix: The input matrix converted to the time domain.
    """
    M_time = matrix_inverse_laplace_transform(M)
    return M_time