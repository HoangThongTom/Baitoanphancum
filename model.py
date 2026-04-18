import numpy as np
import math
from distance import compute_distance_matrix, compute_cluster_distance
from cluster_ops import find_closest_clusters, merge_clusters

def __init__(self, n_clusters=2, linkage='complete'):
    self.n_clusters = n_clusters
    self.linkage = linkage

    self.X = None
    self.N = None
    self.clusters = None
    self.distance_matrix = None
    self.labels = None

def initialize(self):
    self.clusters = [[i] for i in range(self.N)]
    self.distance_matrix = compute_distance_matrix(self.X)      # Duyệt lại coi phù hợp với tên biến của distance.py k

def fit(self, X):
    self.X = X
    self.N = len(X)
    self.initialize()
    while len(self.clusters) > self.n_clusters:                 # Duyệt lại coi phù hợp cluster_ops.py k
        i, j = find_closest_clusters(
            self.clusters, self.distance_matrix, self.linkage   
        )
        self.clusters = merge_clusters(self.clusters, i, j)
        self.labels = self.create_labels()
        return self

def create_labels(self):
    labels = np.zeros(self.N)

    for cluster_id, indices in self.clusters.items():  # Duyệt lại coi phù hợp với tên biến của cluster_ops.py k
        for idx in indices:
            labels[idx] = cluster_id
    
    return labels

def get_labels(self):
    return self.labels
    