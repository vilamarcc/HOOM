import sympy as sp
import utils


def longitudinal_dynamics(model):

    s = sp.symbols('s')

    A = sp.zeros(3)

    A = sp.zeros(3)
    A[0, 0] = 2 * model.lon['mu_lon'] * model.lon['t_lon'] * s - model.lon['CX']['u']
    A[0, 1] = -model.lon['CX']['alpha']
    A[0, 2] = -model['FC']['CZs']
    A[1, 0] = -(model.lon['CZ']['u'] + 2 * model['FC']['CZs'])
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

def lateral_dynamics(model):
    return 0
