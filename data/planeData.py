from utils.conversion import *

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

        # Non-dimensional derivatives [struct]
        # Longitudinal
        self.CD = {'u': 0.053,
            'alpha': 0.036,
            'deltae': 0.022}
                
        self.CL = {'u': 0.213,
            'alpha': 6.578,
            'alphadot': 0.463,
            'q': 9.031,
            'deltae': 0.161}
                
        self.CT = {'u': -0.051,
            'alpha': 0}
        
        self.Cm = {'u': 0.063,
            'Tu': 0.044,
            'alpha': -2.701,
            'alphadot': -1.900,
            'q': -13.777,
            'T': 0,
            'deltae': -0.661}
        
        # Lateral-directional
        self.CY = {'beta': -0.296,
            'p': 0,
            'r': 0.088,
            'deltaa': 0,
            'deltar': -0.117}
                
        self.Cl = {'beta': -0.022,
            'p': -0.567,
            'r': 0.056,
            'deltaa': 0.262,
            'deltar': -0.004}
                
        self.Cn = {'beta': 0.007,
            'p': -0.022,
            'Tbeta': 0,
            'r': -0.018,
            'deltaa': -0.001,
            'deltar': 0.020}