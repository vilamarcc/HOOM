import unittest
import math
import numpy as np
from src.models import plane_model as model
from data import plane_data

class Testplane(unittest.TestCase):
    def test_Kdim_values_longitudinal_globalhawk(self):
        # Create a GlobalHawk plane object
        GH = plane_data.GlobalHawk()

        # Create a plane model object and load the plane data
        plane = model.plane_model()
        plane.loadplane(GH)

        # Calculate the scales and Kdim values
        scales = [plane.FC['us'] / (180 / math.pi), 1, 1, 1 / plane.lon['t_lon']]
        varnames_lon = ["udeltae", "alphadeltae", "thetadeltae", "qdeltae"]
        Gnames_lon = ['G' + varname for varname in varnames_lon]

        for i in range(len(scales)):
            Gname = Gnames_lon[i]
            K_dim = plane.lon['G'][Gname]['K'] * scales[i]
            plane.lon['G'][Gname]['Kdim'] = K_dim

        # Define the expected and actual values for Kdim
        expected_values_lon = [5.342, -0.17, -1.53, -1.53] # GlobalHawk static longitudinal gains
        actual_values_lon = [plane.lon['G'][Gnames_lon[0]]['Kdim'], plane.lon['G'][Gnames_lon[1]]['Kdim'], plane.lon['G'][Gnames_lon[2]]['Kdim'], plane.lon['G'][Gnames_lon[3]]['Kdim']]

        # Print the expected and actual values
        print("Plane name: ", plane.model_name)
        print("Static longitudinal gains: ", Gnames_lon)
        print("Expected values: ", expected_values_lon)
        print("Actual values: ", actual_values_lon)

        # Check that the actual values match the expected values
        self.assertTrue((np.abs(np.subtract(actual_values_lon[0],expected_values_lon[0])/expected_values_lon[0]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lon[1],expected_values_lon[1])/expected_values_lon[1]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lon[2],expected_values_lon[2])/expected_values_lon[2]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lon[3],expected_values_lon[3])/expected_values_lon[3]) < 0.02))

    def test_Kdim_values_lateral_globalhawk(self):
        # Create a GlobalHawk plane object
        GH = plane_data.GlobalHawk()

        # Create a plane model object and load the plane data
        plane = model.plane_model()
        plane.loadplane(GH)

        # Calculate the scales and Kdim values
        scales = [1, 1 / plane.lat['t_lat'], 1, 1 / plane.lat['t_lat'], 1, 1 / plane.lat['t_lat'], 1]
        varnames_lat = ["phideltaa", "rdeltaa", "betadeltar", "rdeltar","psideltaa","pdeltaa", "psideltar"]
        Gnames_lat = ['G' + varname for varname in varnames_lat]

        for i in range(len(scales)):
            Gname = Gnames_lat[i]
            K_dim = plane.lat['G'][Gname]['K'] * scales[i]
            plane.lat['G'][Gname]['Kdim'] = K_dim

        # Define the expected and actual values for Kdim
        expected_values_lat = [72697, 4211, 262, 957, 4211, 72697, 957] # GlobalHawk static lateral gains
        actual_values_lat = [plane.lat['G'][Gnames_lat[0]]['Kdim'], plane.lat['G'][Gnames_lat[1]]['Kdim'], plane.lat['G'][Gnames_lat[2]]['Kdim'], plane.lat['G'][Gnames_lat[3]]['Kdim'], plane.lat['G'][Gnames_lat[4]]['Kdim'], plane.lat['G'][Gnames_lat[5]]['Kdim'], plane.lat['G'][Gnames_lat[6]]['Kdim']]

        # Print the expected and actual values
        print("Plane name: ", plane.model_name)
        print("Static lateral gains: ", Gnames_lat)
        print("Expected values: ", expected_values_lat)
        print("Actual values: ", actual_values_lat)

        # Check that the actual values match the expected values
        self.assertTrue((np.abs(np.subtract(actual_values_lat[0],expected_values_lat[0])/expected_values_lat[0]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lat[1],expected_values_lat[1])/expected_values_lat[1]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lat[2],expected_values_lat[2])/expected_values_lat[2]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lat[3],expected_values_lat[3])/expected_values_lat[3]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lat[4],expected_values_lat[4])/expected_values_lat[4]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lat[5],expected_values_lat[5])/expected_values_lat[5]) < 0.02))
        self.assertTrue((np.abs(np.subtract(actual_values_lat[6],expected_values_lat[6])/expected_values_lat[6]) < 0.02))

if __name__ == '__main__':
    unittest.main()
    print("Everything passed")