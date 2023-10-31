import numpy as np
from math import radians
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree

from scripts_thesis.utils import to_array

class CoordinateCounterTree:
    def __init__(self, data_coordinates : list, ball_tree: BallTree):
        # Initialize the class with geographical coordinates
        self.data_coordinates = data_coordinates
        self.ball_tree = ball_tree

    
    @classmethod
    def from_data_points(self, latitude: list, longitude: list) -> "CoordinateCounterTree":



        data_coordinates = np.concatenate([to_array(latitude), to_array(longitude)], axis=1)
        data_coordinates = np.radians(data_coordinates)

        #returns a list of tuples (lat, lon)
        #data_coordinates = list(map(tuple, data_coordinates)) 
        
        ball_tree = BallTree(data_coordinates, metric="haversine")

        return CoordinateCounterTree(data_coordinates, ball_tree)


    def calculate_points_within_distance(self, point_coordinates: np.array, distance_km: float=1.0):
        
        # Calculate the number of data points within a given distance (in kilometers) efficiently
        distance_radians = distance_km / 6371.0

        # Query the KD-Tree for points within the given distance

        
        count = self.ball_tree.query_radius(np.radians(point_coordinates), r = distance_radians, count_only=True)

        return count

#