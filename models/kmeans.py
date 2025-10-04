"""

"""

import random

class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_points = set()

    def get_k_means():

        random.seed(33)

        initial_centroid_points = random.sample(sorted(list(user_feature_map.keys())), k)
        centroids = {i:user_feature_map[initial_centroid_users[i]] for i in range(k)}

        for _ in range(10):
            