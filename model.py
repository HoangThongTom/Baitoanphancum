import numpy as np

from distance import compute_distance_matrix
from cluster_ops import find_closets_clusters, merge_clusters, update_distance_matrix

class AgglomerativeClusteringCustom:
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
        self.distance_matrix = compute_distance_matrix(self.X, self.clusters)      # Duyệt lại coi phù hợp với tên biến của distance.py k

    def fit(self, X):
        self.X = X
        self.N = len(X)
        self.initialize()
        while len(self.clusters) > self.n_clusters:                 
            i, j = find_closets_clusters(self.distance_matrix)  
            self.clusters = merge_clusters(self.clusters, i, j)      
            self.distance_matrix = update_distance_matrix(self.clusters, self.X, complete=(self.linkage == 'complete'))                                                      
            self.labels = self.create_labels()
        return self

    def create_labels(self):
        labels = np.zeros(self.N)
        for cluster_id, indices in enumerate(self.clusters):  
            for idx in indices:
                labels[idx] = cluster_id
            
        return labels

    def get_labels(self):
        return self.labels
    

    def fit_predict(self, X):
        self.fit(X)
        return self.get_labels()
