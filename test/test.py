from Data import planeData
from Models import UAVmodel as mod
from Utils.ISA import kmh_to_ms

GH = planeData.GlobalHawk()


GH_plane = mod.UAVmodel()

GH_plane.loadplane(GH)

j = 1