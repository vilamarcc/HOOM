import sympy as sp
import utils.scales as utils

def lon_dynamics(model):
    """
    Generate the transfer functions for longitudinal dynamics.

    Parameters:
    model (object): The model object containing the longitudinal parameters.

    Returns:
    dict: A dictionary containing the transfer functions for the longitudinal dynamics.
    """
    # -- Calculate all longitudinal transfer functions --
    # Assumes control by elevator deflection (deltae) and thrust (deltaT).

    # - Symbolic variables -
    s = sp.symbols('s')

    # - State matrix -
    A = sp.zeros(3)
    A[0, 0] = 2 * model.lon['mu_lon'] * model.lon['t_lon'] * s - model.lon['CX']['u']
    A[0, 1] = -model.lon['CX']['alpha']
    A[0, 2] = -model.FC['CZs']
    A[1, 0] = -(model.lon['CZ']['u'] + 2 * model.FC['CZs'])
    A[1, 1] = (2 * model.lon['mu_lon'] - model.lon['CZ']['alphadot']) * model.lon['t_lon'] * s - model.lon['CZ']['alpha']
    A[1, 2] = -(2 * model.lon['mu_lon'] + model.lon['CZ']['q']) * model.lon['t_lon'] * s
    A[2, 0] = -model.lon['Cm']['u']
    A[2, 1] = -model.lon['Cm']['alphadot'] * model.lon['t_lon'] * s - model.lon['Cm']['alpha']
    A[2, 2] = model.lon['Iyb_nd'] * model.lon['t_lon']**2 * s**2 - model.lon['Cm']['q'] * model.lon['t_lon'] * s

    #Control matrix -
    B = sp.zeros(3, 2)
    B[0, 0] = model.lon['CX']['deltae']
    B[0, 1] = model.lon['CX']['T']
    B[1, 0] = model.lon['CZ']['deltae']
    B[1, 1] = model.lon['CZ']['T']
    B[2, 0] = model.lon['Cm']['deltae']
    B[2, 1] = model.lon['Cm']['T']

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
        G[f'G{names[j + 1][i]}'] = utils.factorize_G(utils.syms2tf(Num / Det) * model.lon['t_lon'])

    return G

def lat_dynamics(plane):
    """
    Calculates all lateral-directional transfer functions for the given plane.
    Parameters:
        - plane (object): A class object containing the parameters of the plane.

    Returns:
        - G (dict): A dictionary containing the transfer functions for the lateral-directional dynamics of the plane. The keys of the dictionary are the names of the transfer functions, and the values are the corresponding transfer function objects.
    Note:
    - Assumes control by aileron deflection (deltaa) and rudder deflection (deltar).
    - The transfer functions are calculated based on the parameters of the plane.
    """
    # -- Calculate all lateral-directional transfer functions --
    # Assumes control by aileron deflection (deltaa) and rudder deflection (deltar).
    
    # - Symbolic variables -
    s = sp.symbols('s')
    
    # - State matrix -
    Alat = sp.zeros(3)
    Alat[0, 0] = 2 * plane.lat['mu_lat'] * plane.lat['t_lat'] * s - plane.lat['CY']['beta']
    Alat[0, 1] = plane.FC['CZs'] - plane.lat['CY']['p'] * plane.lat['t_lat'] * s
    Alat[0, 2] = 2 * plane.lat['mu_lat'] - plane.lat['CY']['r']
    Alat[1, 0] = -plane.lat['Cl']['beta']
    Alat[1, 1] = plane.lat['Ixb_nd'] * plane.lat['t_lat']**2 * s**2 - plane.lat['Cl']['p'] * plane.lat['t_lat'] * s
    Alat[1, 2] = -plane.lat['Ixzb_nd'] * plane.lat['t_lat'] * s - plane.lat['Cl']['r']
    Alat[2, 0] = -plane.lat['Cn']['beta']
    Alat[2, 1] = -plane.lat['Ixzb_nd'] * plane.lat['t_lat']**2 * s**2 - plane.lat['Cn']['p'] * plane.lat['t_lat'] * s
    Alat[2, 2] = plane.lat['Izb_nd'] * plane.lat['t_lat'] * s - plane.lat['Cn']['r']
    
    # - Control matrix -
    Blat = sp.zeros(3, 2)
    Blat[0, 0] = plane.lat['CY']['deltaa']  # Generally, 0
    Blat[0, 1] = plane.lat['CY']['deltar']
    Blat[1, 0] = plane.lat['Cl']['deltaa']
    Blat[1, 1] = plane.lat['Cl']['deltar']
    Blat[2, 0] = plane.lat['Cn']['deltaa']
    Blat[2, 1] = plane.lat['Cn']['deltar']
    
    # - Characteristic equation -
    Det = sp.det(Alat, method='berkowitz')
    
    # -- TRANSFER FUNCTIONS --
    names = [["betadeltaa", "betadeltar"], ["phideltaa", "phideltar"], ["rdeltaa", "rdeltar"], ["pdeltaa", "pdeltar"], ["psideltaa", "psideltar"]]
    x, u = len(names), len(names[0])
    G = {}
    
    Num = sp.zeros(x - 2, u)
    
    for i in range(u):
        for j in range(x - 2):
            Asub = Alat.copy()
            Asub[:, j] = Blat[:, i]
            Num[j, i] = sp.det(Asub, method='berkowitz')
            G[f'G{names[j][i]}'] = utils.factorize_G(utils.syms2tf(Num[j, i] / Det))
    
    # p transfer functions
    Num[2, 0] = Num[0, 0] * s * plane.lat['t_lat']
    Num[2, 1] = Num[0, 1] * s * plane.lat['t_lat']
    G[f'G{names[2][0]}'] = utils.factorize_G(utils.syms2tf(Num[2, 0] / Det))
    G[f'G{names[2][1]}'] = utils.factorize_G(utils.syms2tf(Num[2, 1] / Det))
    
    # Psi transfer functions
    Num[3, 0] = Num[1, 0] / s / plane.lat['t_lat']
    Num[3, 1] = Num[1, 1] / s / plane.lat['t_lat']
    G[f'G{names[3][0]}'] = utils.factorize_G(utils.syms2tf(Num[3, 0] / Det))
    G[f'G{names[3][1]}'] = utils.factorize_G(utils.syms2tf(Num[3, 1] / Det))
    
    return G