from math import radians, sin, cos, sqrt, asin, atan2, degrees, pi, acos
import numpy as np

def get_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface using the Haversine formula.

    Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.
    
    Returns:
        float: The distance between the two points in meters.
    """
    earth_radius = 6378  # Earth radius in km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    a = sin(d_lat/2) ** 2 + sin(d_lon/2) ** 2 * cos(lat1_rad) * cos(lat2_rad)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = round((earth_radius * c) * 1000, 3)
    return distance  # Distance in meters

def get_location_at_bearing(lat1, lon1, distance, angle):
    """
    Calculate the latitude and longitude of a location given the initial latitude, longitude, distance, and bearing angle.

    Parameters:
        lat1 (float): The initial latitude in degrees.
        lon1 (float): The initial longitude in degrees.
        distance (float): The distance in meters.
        angle (float): The bearing angle in degrees.

    Returns:
        lat2 (float): The latitude of the new location in degrees.
        lon2 (float): The longitude of the new location in degrees.
    """
    earth_radius = 6378  # Earth radius in km
    distance_km = distance / 1000  # Distance in km
    
    # Convert current lat and lon points to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    
    # Calculate the new latitude using the Haversine formula
    lat2_rad = asin(sin(lat1_rad) * cos(distance_km / earth_radius) + 
                         cos(lat1_rad) * sin(distance_km / earth_radius) * cos(angle))
    lat2 = degrees(lat2_rad)
    
    # Calculate the new longitude using the Haversine formula
    lon2_rad = lon1_rad + atan2(sin(angle) * sin(distance_km / earth_radius) * cos(lat1_rad), 
                                     cos(distance_km / earth_radius) - sin(lat1_rad) * sin(lat2_rad))
    lon2 = degrees(lon2_rad)
    
    return lat2, lon2

def get_bearing_between_coordinates(lat1, lon1, lat2, lon2):
    """
    Calculates the bearing between two sets of coordinates.

    Parameters:
        lat1 (float): The latitude of the first coordinate.
        lon1 (float): The longitude of the first coordinate.
        lat2 (float): The latitude of the second coordinate.
        lon2 (float): The longitude of the second coordinate.

    Returns:
        float: The bearing between the two coordinates in radians.
    """
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    lon1_rad = radians(lon1)
    lon2_rad = radians(lon2)

    y = sin(lon2_rad - lon1_rad) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(lon2_rad - lon1_rad)

    brng = atan2(y, x)
    return round(brng, 3) # Bearing in radians

def get_intersection(lat1, lon1, brng1, lat2, lon2, brng2):
    """
    Calculates the intersection point between two lines given their starting points, bearings, and distances.

    Parameters:
        lat1 (float): The latitude of the first line's starting point in degrees.
        lon1 (float): The longitude of the first line's starting point in degrees.
        brng1 (float): The bearing of the first line in degrees.
        lat2 (float): The latitude of the second line's starting point in degrees.
        lon2 (float): The longitude of the second line's starting point in degrees.
        brng2 (float): The bearing of the second line in degrees.

    Returns:
        tuple: A tuple containing the latitude and longitude of the intersection point in degrees.
    """
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    lambda1 = radians(lon1)
    lambda2 = radians(lon2)
    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1
    delta12 = 2 * asin(
        sqrt(
            sin(delta_phi/2) ** 2 +
            cos(phi1) * cos(phi2) * sin(delta_lambda/2) ** 2
        )
    )
    cos_theta_a = (
        sin(phi2) - sin(phi1) * cos(delta12)
    ) / (
        sin(delta12) * cos(phi1)
    )
    cos_theta_b = (
        sin(phi1) - sin(phi2) * cos(delta12)
    ) / (
        sin(delta12) * cos(phi2)
    )
    theta_a = acos(min(max(cos_theta_a, -1), 1))
    theta_b = acos(min(max(cos_theta_b, -1), 1))
    theta12 = sin(lambda2 - lambda1)
    if sin(lambda2 - lambda1) > 0:
        theta12 = theta_a
        theta21 = 2 * pi - theta_b
    else:
        theta12 = 2 * pi - theta_a
        theta21 = theta_b
    alpha1 = brng1 - theta12
    alpha2 = theta21 - brng2
    if sin(alpha1) == 0 and sin(alpha2) == 0:
        return 0, 0 # Infinite intersections
    if sin(alpha1) * sin(alpha2) < 0:
        return 0, 0 # Ambiguous intersections
    cos_alpha3 = -cos(alpha1) * cos(alpha2) + sin(alpha1) * sin(alpha2) * cos(delta12)
    delta13 = atan2(
        sin(delta12) * sin(alpha1) * sin(alpha2),
        cos(alpha2) + cos(alpha1) * cos_alpha3
    )
    
    phi3 = asin(
        sin(phi1) * cos(delta13) + cos(phi1) * sin(delta13) * cos(brng1)
    )
    delta_lambda13 = atan2(
        sin(brng1) * sin(delta13) * cos(phi1),
        cos(delta13) - sin(phi1) * sin(phi3)
    )
    lambda3 = lambda1 + delta_lambda13
    return degrees(phi3), degrees(lambda3)

def from_xy_to_lat_long(x, y, lat1, lon1):
    """
    Converts Cartesian coordinates (x, y) to latitude and longitude using the haversine formula.

    Parameters:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        lat1 (float): The latitude of the reference point in degrees.
        lon1 (float): The longitude of the reference point in degrees.

    Returns:
        tuple: A tuple containing the latitude and longitude of the converted point in degrees.
    """
    radius = 6378 * 1000
    distance = sqrt(x**2 + y**2)
    bearing = atan2(y, -x) - pi / 2
    latitude1 = lat1 * pi / 180
    longitude1 = lon1 * pi / 180
    latitude2 = asin(sin(latitude1) * cos(distance / radius) + cos(latitude1) * sin(distance / radius) * cos(bearing))
    longitude2 = longitude1 + atan2(sin(bearing) * sin(distance / radius) * cos(latitude1), cos(distance / radius) - sin(latitude1) * sin(latitude2))

    return degrees(latitude2), degrees(longitude2)
