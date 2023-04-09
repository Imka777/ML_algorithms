import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    indices = np.argpartition(ranks, top, axis=axis)
    indices = np.take(indices, np.arange(top), axis=axis)
    new_ranks = np.take_along_axis(ranks, indices, axis=axis)
    indices_top = np.argsort(new_ranks, axis=axis)
    indices = np.take_along_axis(indices, indices_top, axis=axis)

    if return_ranks:
        return(np.take_along_axis(ranks, indices, axis=axis), indices)
    
    return indices


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X

        return self

    def kneighbors(self, X, return_distance=False):
        ranks = self._metric_func(X, self._X)
        
        return get_best_ranks(ranks, self.n_neighbors, return_ranks=return_distance)

