import sympy as sp
import src.utils.scales as utils

def lon_dynamics(plane):
    """
    Calculates all lognitudinal transfer functions for the given plane.
    Parameters:
        - plane (object): A class object containing the parameters of the plane.

    Returns:
        - G (dict): A dictionary containing the transfer functions for the logitudinal dynamics of the plane. The keys of the dictionary are the names of the transfer functions, and the values are the corresponding transfer function objects.
        - A (sympy matrix): The state matrix of the logitudinal dynamics.
        - B (sympy matrix): The control matrix of the logitudinal dynamics.
    Note:
    - Assumes control by elevator deflection (deltae) and thrust (deltaT).
    - The transfer functions are calculated based on the parameters of the plane.
    """

    # -- Calculate all longitudinal transfer functions --
    # Assumes control by elevator deflection (deltae) and thrust (deltaT).

    # - Symbolic variables -

    s = sp.symbols('s')

    # - State matrix -
    A = sp.zeros(3)
    A[0, 0] = 2 * plane.lon['mu_lon'] * plane.lon['t_lon'] * s - plane.lon['CX']['u']
    A[0, 1] = -plane.lon['CX']['alpha']
    A[0, 2] = -plane.FC['CZs']
    A[1, 0] = -(plane.lon['CZ']['u'] + 2 * plane.FC['CZs'])
    A[1, 1] = (2 * plane.lon['mu_lon'] - plane.lon['CZ']['alphadot']) * plane.lon['t_lon'] * s - plane.lon['CZ']['alpha']
    A[1, 2] = -(2 * plane.lon['mu_lon'] + plane.lon['CZ']['q']) * plane.lon['t_lon'] * s
    A[2, 0] = -plane.lon['Cm']['u']
    A[2, 1] = -plane.lon['Cm']['alphadot'] * plane.lon['t_lon'] * s - plane.lon['Cm']['alpha']
    A[2, 2] = plane.lon['Iyb_nd'] * plane.lon['t_lon']**2 * s**2 - plane.lon['Cm']['q'] * plane.lon['t_lon'] * s

    #Control matrix -
    B = sp.zeros(3, 2)
    B[0, 0] = plane.lon['CX']['deltae']
    B[0, 1] = plane.lon['CX']['T']
    B[1, 0] = plane.lon['CZ']['deltae']
    B[1, 1] = plane.lon['CZ']['T']
    B[2, 0] = plane.lon['Cm']['deltae']
    B[2, 1] = plane.lon['Cm']['T']

    # - Characteristic equation -
    Det = sp.det(A)

    # -- TRANSFER FUNCTIONS --
    names = [["udeltae", "uT"], ["alphadeltae", "alphaT"], ["thetadeltae", "thetaT"], ["qdeltae", "qdeltaT"]]
    x, u = len(names), len(names[0])
    G = {}

    for i in range(u):
        for j in range(x - 1):
            Asub = A.copy()
            Asub[:, j] = B[:, i]
            Num = sp.det(Asub)
            G[f'G{names[j][i]}'] = utils.factorize_G(utils.syms2tf(Num / Det))

        Num = Num * s
        # Pitch rate transfer function (qhat = Dtheta/Dt)
        G[f'G{names[j + 1][i]}'] = utils.factorize_G(utils.syms2tf(Num / Det) * plane.lon['t_lon'])

    return G,A,B

def lat_dynamics(plane):
    """
    Calculates all lateral-directional transfer functions for the given plane.
    Parameters:
        - plane (object): A class object containing the parameters of the plane.

    Returns:
        - G (dict): A dictionary containing the transfer functions for the lateral-directional dynamics of the plane. The keys of the dictionary are the names of the transfer functions, and the values are the corresponding transfer function objects.
        - A (sympy matrix): The state matrix of the lateral-directional dynamics.
        - B (sympy matrix): The control matrix of the lateral-directional dynamics.
    Note:
    - Assumes control by aileron deflection (deltaa) and rudder deflection (deltar).
    - The transfer functions are calculated based on the parameters of the plane.
    """
    # -- Calculate all lateral-directional transfer functions --
    # Assumes control by aileron deflection (deltaa) and rudder deflection (deltar).
    
    # - Symbolic variables -
    s = sp.symbols('s')
    
    # - State matrix -
    A = sp.zeros(3)
    A[0, 0] = 2 * plane.lat['mu_lat'] * plane.lat['t_lat'] * s - plane.lat['CY']['beta']
    A[0, 1] = plane.FC['CZs'] - plane.lat['CY']['p'] * plane.lat['t_lat'] * s
    A[0, 2] = 2 * plane.lat['mu_lat'] - plane.lat['CY']['r']
    A[1, 0] = -plane.lat['Cl']['beta']
    A[1, 1] = plane.lat['Ixb_nd'] * plane.lat['t_lat']**2 * s**2 - plane.lat['Cl']['p'] * plane.lat['t_lat'] * s
    A[1, 2] = -plane.lat['Ixzb_nd'] * plane.lat['t_lat'] * s - plane.lat['Cl']['r']
    A[2, 0] = -plane.lat['Cn']['beta']
    A[2, 1] = -plane.lat['Ixzb_nd'] * plane.lat['t_lat']**2 * s**2 - plane.lat['Cn']['p'] * plane.lat['t_lat'] * s
    A[2, 2] = plane.lat['Izb_nd'] * plane.lat['t_lat'] * s - plane.lat['Cn']['r']
    
    # - Control matrix -
    B = sp.zeros(3, 2)
    B[0, 0] = plane.lat['CY']['deltaa']  # Generally, 0
    B[0, 1] = plane.lat['CY']['deltar']
    B[1, 0] = plane.lat['Cl']['deltaa']
    B[1, 1] = plane.lat['Cl']['deltar']
    B[2, 0] = plane.lat['Cn']['deltaa']
    B[2, 1] = plane.lat['Cn']['deltar']
    
    # - Characteristic equation -
    Det = A.det()
    
    # -- TRANSFER FUNCTIONS --
    names = [["betadeltaa", "betadeltar"], ["phideltaa", "phideltar"], ["rdeltaa", "rdeltar"], ["pdeltaa", "pdeltar"], ["psideltaa", "psideltar"]]
    x, u = len(names), len(names[0])
    G = {}
    
    Num = sp.zeros(x, u)
    
    for i in range(u):
        for j in range(x - 2):
            Asub = A.copy()
            Asub[:, j] = B[:, i]
            Num[j, i] = Asub.det()
            G[f'G{names[j][i]}'] = utils.factorize_G(utils.syms2tf(Num[j, i] / Det))
    
    # p transfer functions
    Num[3, 0] = Num[1, 0] * s * plane.lat['t_lat']
    Num[3, 1] = Num[1, 1] * s * plane.lat['t_lat']

    G[f'G{names[3][0]}'] = utils.factorize_G(utils.syms2tf(Num[3, 0] / Det))
    G[f'G{names[3][1]}'] = utils.factorize_G(utils.syms2tf(Num[3, 1] / Det))
    
    # Psi transfer functions
    Num[4, 0] = Num[2, 0] / s / plane.lat['t_lat']
    Num[4, 1] = Num[2, 1] / s / plane.lat['t_lat']

    G[f'G{names[4][0]}'] = utils.factorize_G(utils.syms2tf(Num[4, 0] / Det))
    G[f'G{names[4][1]}'] = utils.factorize_G(utils.syms2tf(Num[4, 1] / Det))
    
    return G,A,B