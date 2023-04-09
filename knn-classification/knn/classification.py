import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)

        return self


    def _predict_precomputed(self, indices, distances):
        ans = np.empty(0)

        if self._weights == 'distance':
            for row, dist in zip(indices, distances):
                ans = np.append(ans, np.argmax(np.bincount(np.array(self._labels[row]), weights=1/(dist + self.EPS))))
        else:
            for row in indices:
                if row.dtype != 'O':
                    ans = np.append(ans, np.argmax(np.bincount(self._labels[row])))
                else:
                    print(row)
        
        return ans

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)

        splited = np.array_split(X, X.shape[0]//self._batch_size)

        if return_distance:
            distances = np.array([])
            indices = np.array([])

            for batch in splited:
                dist, index = self._finder.kneighbors(batch, return_distance=return_distance)
                distances = np.vstack((distances, dist)) if distances.size else dist
                indices = np.vstack((indices, index)) if indices.size else index

            return (distances, indices)
        
        else:
            indices = np.array([])

            for batch in splited:
                index = self._finder.kneighbors(batch, return_distance=return_distance)
                indices = np.vstack((indices, index)) if indices.size else index

            return indices
        
