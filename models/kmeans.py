"""
A basic K-Means Clustering
"""

import numpy as np

class KMeansBasic:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _init_centers(self, X):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _pairwise_sq_dists(self, A, B):
        # Matrix of squared euclidean distances
        return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        centers = self._init_centers(X)

        for _ in range(self.max_iter):
            
            dist2 = self._pairwise_sq_dists(X, centers)
            labels = np.argmin(dist2, axis=1)

            
            new_centers = centers.copy()
            for j in range(self.n_clusters):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = X[mask].mean(axis=0)
                

            
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dist2 = self._pairwise_sq_dists(X, self.cluster_centers_)
        return np.argmin(dist2, axis=1)

