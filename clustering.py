class AgglomerativeClusteringCustom:

    def fit_predict(self, X):
        self.fit(X)
        return self.get_labels()
