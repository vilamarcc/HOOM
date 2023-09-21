import utils.ISA
import sympy as sp
import control as ctrl
import numpy as np

def get_scales(plane):
    """
    Calculates and returns characteristic values for a given plane.

    Parameters:
        plane (object): A object class containing information about the plane.
            It should have the following attributes:
                - 'FC': A dictionary containing flight conditions.
                    It should have the following keys:
                        - 'hs': The altitude of the plane.
                        - 'us': The speed of the plane.
                - 'c': The chord length of the plane.
                - 'Sw': The wing area of the plane.
                - 'b': The wingspan of the plane.

    Returns:
        dict: A dictionary containing characteristic values.
            It has the following keys:
                - 'lon': A dictionary containing longitudinal characteristic values.
                    It has the following keys:
                        - 'length': The half chord length of the plane.
                        - 'area': The wing area of the plane.
                        - 'speed': The speed of the plane.
                        - 'mass': The mass of the plane.
                        - 'time': The time of flight.
                        - 'press': The air pressure.
                        - 'force': The force on the wing.
                        - 'moment': The moment on the wing.
                        - 'inertia': The moment of inertia.
                - 'lat': A dictionary containing lateral-directional characteristic values.
                    It has the following keys:
                        - 'length': The half wingspan of the plane.
                        - 'area': The wing area of the plane.
                        - 'speed': The speed of the plane.
                        - 'mass': The mass of the plane.
                        - 'time': The time of flight.
                        - 'press': The air pressure.
                        - 'force': The force on the wing.
                        - 'moment': The moment on the wing.
                        - 'inertia': The moment of inertia.
    """
    _, _, rho = utils.ISA(plane.FC['hs'])  # Assuming Utils.ISA returns three values

    scales = {}

    # Longitudinal
    scales['lon'] = {
        'length': plane.c / 2,
        'area': plane.Sw,
        'speed': plane.FC['us'],
        'mass': plane.c * plane.Sw * rho / 2,
        'time': plane.c / (2 * plane.FC['us']),
        'press': 0.5 * rho * plane.FC['us']**2,
        'force': 0.5 * rho * plane.FC['us']**2 * plane.Sw,
        'moment': 0.5 * rho * plane.FC['us']**2 * plane.Sw * plane.c,
        'inertia': rho * plane.Sw * (plane.c / 2)**3
    }

    # Lateral-directional
    scales['lat'] = {
        'length': plane.b / 2,
        'area': plane.Sw,
        'speed': plane.FC['us'],
        'mass': plane.b * plane.Sw * rho / 2,
        'time': plane.b / (2 * plane.FC['us']),
        'press': 0.5 * rho * plane.FC['us']**2,
        'force': 0.5 * rho * plane.FC['us']**2 * plane.Sw,
        'moment': 0.5 * rho * plane.FC['us']**2 * plane.Sw * plane.b,
        'inertia': rho * plane.Sw * (plane.b / 2)**3
    }

    return scales

def syms2tf(G):
    """
    Generates a transfer function from a given symbolic expression.
    
    Parameters:
        G (symbolic expression): The symbolic expression representing the transfer function.
        
    Returns:
        TF_from_sym (TransferFunction): The transfer function generated from the symbolic expression.
    """
    symNum, symDen = sp.fraction(G)  # Get num and den of Symbolic TF
    num = sp.poly(symNum)            # Convert Symbolic num to polynomial
    den = sp.poly(symDen)            # Convert Symbolic den to polynomial
    TF_from_sym = ctrl.TransferFunction(num, den)

    return TF_from_sym

def tf2syms(G):
    """
    Returns a symbolic expression from a transfer function.
    
    Args:
        G (TransferFunction): The transfer function.
    
    Returns:
        SymbolicExpression: The symbolic expression obtained from the transfer function.
    """
    s = sp.symbols('s')
    num, den = ctrl.tfdata(G, 'v')
    sym_from_TF = sp.poly(num, s) / sp.poly(den, s)

    return sym_from_TF

def factorize_G(G):
    """
    Returns a factorized version of a transfer function and the associated characteristic times, natural frequencies, and damping factors.
    
    Args:
        G (Transfer Function): The transfer function to be factorized.
        
    Returns:
        dict: A dictionary containing the factorized transfer function, the associated characteristic times, natural frequencies, and damping factors.
            - 'Gfact_sym' (SymPy expression): The factorized transfer function as a SymPy expression.
            - 'Gfact' (TransferFunction): The factorized transfer function as a TransferFunction object.
            - 'K' (float): The static gain of the factorized transfer function.
            - 'zeros' (dict): A dictionary containing information about the zeros of the factorized transfer function.
                - 'tau' (list): The characteristic times of the zeros.
                - 'wn' (list): The natural frequencies of the zeros.
                - 'xi' (list): The damping factors of the zeros.
            - 'poles' (dict): A dictionary containing information about the poles of the factorized transfer function.
                - 'tau' (list): The characteristic times of the poles.
                - 'wn' (list): The natural frequencies of the poles.
                - 'xi' (list): The damping factors of the poles.
    """
    
    s = sp.symbols('s')
    
    # -- Poles (can be real or complex) --
    poles = ctrl.pole(G)
    count = 0
    tau_p = []
    wn_p = []
    xi_p = []
    null_p = 0
    
    while count < len(poles):
        if np.isreal(poles[count]):
            if poles[count] != 0:
                tau_p.append(-1/poles[count])
            else:
                null_p += 1
            count += 1
        else:
            lambda_p = poles[count]
            n_p = np.real(lambda_p)
            w_p = abs(np.imag(lambda_p))
            wn_p.append(np.sqrt(n_p**2 + w_p**2))
            xi_p.append(-n_p/np.sqrt(n_p**2 + w_p**2))
            count += 2
    
    Denfact = 1
    for tau in tau_p:
        Denfact *= (tau*s + 1)
    for wn, xi in zip(wn_p, xi_p):
        Denfact *= ((s/wn)**2 + 2*xi*(s/wn) + 1)
    Denfact *= s**null_p
    
    # -- Zeros (can be real or complex) --
    zeros = ctrl.zero(G)
    count = 0
    tau_z = []
    wn_z = []
    xi_z = []
    null_z = 0
    
    while count < len(zeros):
        if np.isreal(zeros[count]):
            if zeros[count] != 0:
                tau_z.append(-1/zeros[count])
            else:
                null_z += 1
            count += 1
        else:
            lambda_z = zeros[count]
            n_z = np.real(lambda_z)
            w_z = abs(np.imag(lambda_z))
            wn_z.append(np.sqrt(n_z**2 + w_z**2))
            xi_z.append(-n_z/np.sqrt(n_z**2 + w_z**2))
            count += 2

    # -- Static gain -- (removing s associated with null zeros and poles)
    K = sp.limit(ctrl.tf2sym(G)*s**null_p/s**null_z, s, 0)

    Numfact = 1
    for tau in tau_z:
        Numfact *= (tau*s + 1)
    for wn, xi in zip(wn_z, xi_z):
        Numfact *= ((s/wn)**2 + 2*xi*(s/wn) + 1)
    Numfact *= s**null_z

    # -- Transfer function --
    Gfact_sym = K*Numfact/Denfact
    Gfact = ctrl.TransferFunction(Gfact_sym.as_numer_denom()[0], Gfact_sym.as_numer_denom()[1])
    K = float(K)
    
    G_f = {
        'Gfact_sym': Gfact_sym,
        'Gfact': Gfact,
        'K': K,
        'zeros': {
            'tau': tau_z,
            'wn': wn_z,
            'xi': xi_z
        },
        'poles': {
            'tau': tau_p,
            'wn': wn_p,
            'xi': xi_p
        }
    }
    
    return G_f