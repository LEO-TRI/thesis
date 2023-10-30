from math import radians
from scipy.spatial import KDTree
from scripts_thesis.utils import to_array, is_array_like

class CoordinateCounter:
    def __init__(self, latitude, longitude):
        # Initialize the class with geographical coordinates
        self.latitude = radians(latitude)
        self.longitude = radians(longitude)
        self.data_points = []
        self.kd_tree = None

    def add_data_point(self, latitude, longitude):

        if isinstance(latitude, float) & isinstance(longitude, float):
            self.data_points.append((radians(latitude), radians(longitude)))
            self.kd_tree = KDTree(self.data_points)
        elif is_array_like(latitude) & is_array_like(longitude):
            latitude, longitude = to_array(latitude), to_array(longitude)
            for lat, lon in zip(latitude, longitude):
                self.data_points.append((radians(lat), radians(lon)))
                self.kd_tree = KDTree(self.data_points)


    def calculate_points_within_distance(self, distance_km):
        # Calculate the number of data points within a given distance (in kilometers) efficiently
        if self.kd_tree is None:
            pass

        # Convert distance to radians
        distance_radians = distance_km / 6371.0

        # Query the KD-Tree for points within the given distance
        count = len(self.kd_tree.query_ball_point([self.latitude, self.longitude], distance_radians))
        return count
