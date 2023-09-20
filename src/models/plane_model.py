import math
import numpy as np
from utils.ISA import *

class plane_model:

    # ------- UAV class --------
    # Contains geometry, mass, stability derivatives and TFs of a given fixed wing UAV
    # Transforms dimensional quantities to non dimensional and projects
    # stability derivatives on body axes. Calculates TFs from derivatives
    # and dynamic equations.

    def __init__(self, model_name = None, Sw = 0, c = 0, b = 0, m = 0, Ixb = 0, Iyb = 0, Izb = 0, Ixzb = 0, model = None, FC = None, lat = None, lon = None, Vmax = 0, hmax = 0, Vc = 0):

        # -- Name [str] -- 
        self.model_name = model_name

        # -- Geometric [float] --
        self.Sw = Sw
        self.c = c
        self.b = b

        # -- Mass [float] --
        self.m = m
        self.Ixb = Ixb
        self.Iyb = Iyb
        self.Izb = Izb
        self.Ixzb = Ixzb

        # -- Performances [float] -- 
        self.Vmax = Vmax 
        self.Vc = Vc   
        self.hmax = hmax

        # -- Model [float] --
        self.model = model

        # -- Flight condition [struct] -- ESTO ALOMEJOR NO HACE FALTA
        self.FC = FC

        # -- Channels --
        # Each one contains necessary non-dimensional magnitudes,
        # stability derivatives and transfer functions
        self.lat = lat
        self.lon = lon

        """     - Longitudinal channel -
               Body-axes long. stability derivatives
             C_Xu, C_Xalpha, C_Xdeltae
             C_Zu, C_Zalpha, C_Zalphadot, C_Zq, C_Zdeltae
             C_mdeltae
             C_mu, C_malpha, C_malphadot, C_mq, C_mTu,
               Non dimensional time, mass, inertia
             t_lon, mu_lon, Iyb_nd
               Transfer functions
             G
    
             - Lateral-directional channel -
               Body-axes lat.-dir. stability derivatives
             C_Ybeta, C_Yp, C_Yr, C_Ydeltaa, C_Ydeltar
             C_lbeta, C_lp, C_lr, C_ldeltaa, C_ldeltar
             C_nbeta, C_np, C_nr, C_nTb, C_ndeltaa, 
             C_ndeltar
               Non dimensional time, mass, inertia
             t_lat, mu_lat, Ixb_nd, Izb_nd, Ixzb_nd
               Transfer functions
             G """

    def loadplane(self,model = None):

        # - Geometric and mass properties -
        self.model_name    = model.name
        self.Sw            = model.Sw
        self.c             = model.c
        self.b             = model.b
        self.m             = model.m
        self.Ixb           = model.Ixb
        self.Iyb           = model.Iyb
        self.Izb           = model.Izb
        self.Ixzb          = model.Ixzb

        # - Performance properties -
        self.Vmax = model.Vmax
        self.Vc = model.Vc
        self.hmax = model.hmax

        # - Flight derivatives. By default, dimensionless -
        # Store raw inputs

        self.model = {}

        self.model['CL'] = model.CL
        self.model['CD'] = model.CD
        self.model['CT'] = model.CT
        self.model['CY'] = model.CY
        self.model['Cl'] = model.Cl
        self.model['Cm'] = model.Cm
        self.model['Cn'] = model.Cn
    
    def loadDynamics(self):

        # -- State matrix --
        Alon = np.zeros((4,4))
        Alat = np.zeros((4,4))


