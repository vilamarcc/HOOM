from utils.conversion import *
import numpy as np

# This file contais some useful examples of plane data used to run the library, more can be added by following the same class structure for future use

 # -- Global Hawk Characteristics & Stability derivatives --

class GlobalHawk:
    def __init__(self):

        # -- Name [str] -- 
        self.name = "Global Hawk"

        # -- Geometric [float] --
        self.Sw = 48.68 # [m^2]
        self.c = 1.48   # [m]
        self.b = 35.42  # [m]

        # -- Mass [float] -- REVISAR DATOS MÁSICOS (Condición MPL, MaxRange, etc)
        self.m = 9375     # kg
        self.Ixb = 168907 # [kg*m^2]
        self.Iyb = 45466  # [kg*m^2]
        self.Izb = 210513 # [kg*m^2]
        self.Ixzb = -1660 # [kg*m^2]

        # -- Performances [float] -- 
        self.Vmax = kmh_to_ms(800.0) # [m/s]
        self.Vc = kmh_to_ms(650.0)   # [m/s]
        self.hmax = m_to_ft(19812) # [ft]

        # Non-dimensional derivatives [dict]
        # Longitudinal
        self.CD = {'u': 0.053,
            'alpha': 0.036,
            'deltae': 0.022
        }
                
        self.CL = {'u': 0.213,
            'alpha': 6.578,
            'alphadot': 0.463,
            'q': 9.031,
            'deltae': 0.161
        }
                
        self.CT = {'u': -0.051,
            'alpha': 0
        }
        
        self.Cm = {'u': 0.063,
            'Tu': 0.044,
            'alpha': -2.701,
            'alphadot': -1.900,
            'q': -13.777,
            'T': 0,
            'deltae': -0.661
        }
        
        # Lateral-directional
        self.CY = {'beta': -0.296,
            'p': 0,
            'r': 0.088,
            'deltaa': 0,
            'deltar': -0.117
        }
                
        self.Cl = {'beta': -0.022,
            'p': -0.567,
            'r': 0.056,
            'deltaa': 0.262,
            'deltar': -0.004
        }
                
        self.Cn = {'beta': 0.007,
            'p': -0.022,
            'Tbeta': 0,
            'r': -0.018,
            'deltaa': -0.001,
            'deltar': 0.020
        }
        
        FC = {
                'hs': ft_to_m(10000),
                'us': kt_to_ms(320),
                'alphabs': deg_to_rad(0),
                'CLs': 0.154,
                'CDs': 0.026,
                'CTs': 0.026 / np.cos(deg_to_rad(0)),  # Assumed equal to CDs (alpha = 0)
                'Cms': 0,  # Assumed 0
                'CmTs': 0,  # Assumed 0
                'epss': deg_to_rad(0)
        }

# -- Learjet 24 Characteristics & Stability derivatives --

class Learjet24:
    
    # Contains geometric and mass properties, FC and stability derivatives
    # of a Learjet 24 airplane at cruise condition.
    # Extracted from Leyes de Control de Vuelo Appendix B.

    def __init__(self):
        # Name [str]
        self.name = 'Learjet 24'

        # Geometric [float]
        self.Sw = ft_to_m(ft_to_m(230))  # [m^2]
        self.c = ft_to_m(7.0)  # [m]
        self.b = ft_to_m(34.0)  # [m]

        # Mass [float]
        self.m = lb_to_kg(13000)  # [kg]
        self.Ixb = slgft2_tokgm2(28000)  # [kg*m^2]
        self.Iyb = slgft2_tokgm2(18800)  # [kg*m^2]
        self.Izb = slgft2_tokgm2(47000)  # [kg*m^2]
        self.Ixzb = slgft2_tokgm2(1300)  # [kg*m^2]

        # Flight condition [dict]
        self.FC = {
            'hs': ft_to_m(40000),
            'us': ft_to_m(677),  # Speed given in ft/s
            'alphabs': deg_to_rad(2.7),
            'CLs': 0.410,
            'CDs': 0.0335,
            'CTs': 0.0335 / np.cos(deg_to_rad(0)),  # CTxs = CDs
            'Cms': 0, # Assumed 0
            'CmTs': 0, # Assumed 0
            'epss': deg_to_rad(0)
        }

        # Non-dimensional derivatives [dict]
        # Longitudinal
        self.CD = {
            'u': 0.104,
            'alpha': 0.30,
            'deltae': 0,
            'ih': 0
        }

        self.CL = {
            'u': 0.40,
            'alpha': 5.84,
            'alphadot': 2.2,
            'q': 4.7,
            'deltae': 0.46,
            'ih': 0.94
        }

        self.CT = {
            'u': -0.07,
            'alpha': 0
        }

        self.Cm = {
            'u': 0.050,
            'Tu': -0.003,
            'alpha': -0.64,
            'alphadot': -6.7,
            'q': -15.5,
            'T': 0,
            'deltae': -1.24,
            'ih': -2.5
        }

        # Lateral-directional
        self.CY = {
            'beta': -0.730,
            'p': 0,
            'r': 0.40,
            'deltaa': 0,
            'deltar': -0.140
        }

        self.Cl = {
            'beta': -0.110,
            'p': -0.450,
            'r': 0.160,
            'deltaa': 0.178,
            'deltar': -0.019
        }

        self.Cn = {
            'beta': 0.127,
            'p': -0.008,
            'Tbeta': 0,
            'r': -0.200,
            'deltaa': -0.020,
            'deltar': 0.074
        }