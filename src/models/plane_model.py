import math
import numpy as np
from src.utils.ISA import *
from src.models.dynamics import lat_dynamics, lon_dynamics
from src.utils.scales import *

class plane_model:

    # ------- Plane class --------
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

        # -- Flight condition [struct] -- 
        self.FC = FC

        # -- Channels --
        # Each one contains necessary non-dimensional magnitudes,
        # stability derivatives and transfer functions
        self.lat = lat
        self.lon = lon

        #     - Longitudinal channel -
        #       Body-axes long. stability derivatives
        #     C_Xu, C_Xalpha, C_Xdeltae
        #     C_Zu, C_Zalpha, C_Zalphadot, C_Zq, C_Zdeltae
        #     C_mdeltae
        #     C_mu, C_malpha, C_malphadot, C_mq, C_mTu,
        #       Non dimensional time, mass, inertia
        #     t_lon, mu_lon, Iyb_nd
        #       Transfer functions
        #     G
        #     - Lateral-directional channel -
        #       Body-axes lat.-dir. stability derivatives
        #     C_Ybeta, C_Yp, C_Yr, C_Ydeltaa, C_Ydeltar
        #     C_lbeta, C_lp, C_lr, C_ldeltaa, C_ldeltar
        #     C_nbeta, C_np, C_nr, C_nTb, C_ndeltaa, 
        #     C_ndeltar
        #       Non dimensional time, mass, inertia
        #     t_lat, mu_lat, Ixb_nd, Izb_nd, Ixzb_nd
        #       Transfer functions
        #     G """

    def loadplane(self,model = None):
        """
        Load flight condition and derivatives into class.
        Includes non-dimensional characteristics.
        
        Parameters:
            model (optional): The model selfect containing the flight condition and derivatives.
        
        Returns:
            None
        """
        # -- Load flight condition and derivatives into class --
        # Includes non dimensional characteristics

        scales = get_scales(model)

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

        # - Non dimensional properties -
        self.lat = {}
        self.lon = {}

        self.lon['t_lon'] = scales['lon']['time']
        self.lat['t_lat'] = scales['lat']['time']
        self.lon['mu_lon'] = model.m/scales['lon']['mass']
        self.lat['mu_lat'] = model.m/scales['lat']['mass']
        self.lat['Ixb_nd'] = model.Ixb/scales['lat']['inertia']
        self.lat['Izb_nd'] = model.Izb/scales['lat']['inertia']
        self.lon['Iyb_nd'] = model.Iyb/scales['lon']['inertia']
        self.lat['Ixzb_nd'] = model.Ixzb/scales['lat']['inertia']

        # - Flight condition -
        self.FC = model.FC

        self.FC['CXs'] = model.FC['CTs']*np.cos(model.FC['epss']) + model.FC['CLs']*model.FC['alphabs'] - model.FC['CDs']
        self.FC['CZs'] = -model.FC['CTs']*np.sin(model.FC['epss']) - model.FC['CLs'] - model.FC['CDs']*model.FC['alphabs']
        self.FC['Cms'] = 0

        # - Performance properties -
        #self.Vmax = model.Vmax
        #self.Vc = model.Vc
        #self.hmax = model.hmax

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

        # Longitudinal

        self.lon['CX'] = {
            'u': self.model['CT']['u']*np.cos(self.FC['epss']) - self.model['CD']['u'],
            'alpha': self.model['CT']['alpha']*np.cos(self.FC['epss']) + self.FC['CLs'] - self.model['CD']['alpha'],
            'deltae': -self.model['CD']['deltae'],
            'T': self.FC['CTs']*np.cos(self.FC['epss'])
        }
        
        self.lon['CZ'] = {
            'u': -self.model['CT']['u']*np.sin(self.FC['epss']) - self.model['CL']['u'],
            'alpha': -self.model['CT']['alpha']*np.sin(self.FC['epss']) -self.model['CL']['alpha'] - self.FC['CDs'],
            'alphadot': -self.model['CL']['alpha'],
            'q': -self.model['CL']['q'],
            'deltae': -self.model['CL']['deltae'],
            'T': -self.FC['CTs']*np.sin(self.FC['epss'])
        }

        self.lon['Cm'] = self.model['Cm']
        self.lon['Cm']['u'] = self.model['Cm']['u'] + self.model['Cm']['Tu']
        
        # Lateral-directional
        self.lat['CY'] = model.CY
        self.lat['Cl'] = model.Cl
        self.lat['Cn'] = model.Cn
        
        # - Transfer functions -
        # Longitudinal
        self.lon['G'],self.lon['A'],self.lon['B'] = lon_dynamics(self)
        
        # Lateral
        self.lat['G'],self.lat['A'],self.lat['B'] = lat_dynamics(self)

    
    def set_FC(self,newFC):
        """
        Sets a new flight condition of the aircraft.

        Parameters:
        - newFC: dictionary containing the new flight condition values
            - 'hs': flight altitude
            - 'us': cruise speed
            - 'CTs': static thrust coefficient value
            - 'epss': static thrust offset angle
            - 'CLs': static lift coefficient value
            - 'CDs': static drag coefficient value
            - 'Cms': static momentum coefficient value
            - 'CmTs: static thrust engine momentum coefficient value
            - 'alphabs': static angle of attack 

        Returns:
        None
        """
        # - Flight condition -
        self.FC = newFC

        self.FC['CXs'] = newFC['CTs']*np.cos(newFC['epss']) + newFC['CLs']*newFC['alphabs'] - newFC['CDs']
        self.FC['CZs'] = -newFC['CTs']*np.sin(newFC['epss']) - newFC['CLs'] - newFC['CDs']*newFC['alphabs']
        self.FC['Cms'] = 0

        # Longitudinal
        self.lon['CX']['u'] = self.model['CT']['u']*np.cos(self.FC['epss']) - self.model['CD']['u']
        self.lon['CX']['alpha'] = self.model['CT']['alpha']*np.cos(self.FC['epss']) + self.FC['CLs'] - self.model['CD']['alpha']
        self.lon['CX']['delta'] = -self.model['CD']['delta']
        self.lon['CX']['T'] = self.FC['CTs']*np.cos(self.FC['epss'])
        self.lon['CZ']['u'] = -self.model['CT']['u']*np.sin(self.FC['epss']) - self.model['CL']['u']
        self.lon['CZ']['alpha'] = -self.model['CT']['alpha']*np.sin(self.FC['epss']) -self.model['CL']['alpha'] - self.FC['CDs']
        self.lon['CZ']['alphadot'] = -self.model['CL']['alpha']
        self.lon['CZ']['q'] = -self.model['CL']['q']
        self.lon['CZ']['delta'] = -self.model['CL']['delta']
        self.lon['CZ']['T'] = -self.FC['CTs']*np.sin(self.FC['epss'])
        self.lon['Cm'] = self.model['Cm']
        self.lon['Cm']['u'] = self.model['Cm']['u'] + self.model['Cm']['Tu']

                # - Transfer functions -
        # Longitudinal
        self.lon['G'] = lon_dynamics(self)
        
        # Lateral
        self.lat['G'] = lat_dynamics(self)