from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    result = {}
    k_list = sorted(k_list, reverse=True)

    for train_index, test_index in cv.split(X):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X[train_index], X[test_index]
        isfirst = True
        
        for k in k_list:
            model = BatchedKNNClassifier(k, **kwargs)
            model.fit(X_train, y_train)

            if isfirst:
                distances, indices = model.kneighbors(X_test, return_distance=True)
                score = scorer(y_test, model._predict_precomputed(
                    indices=indices, distances=distances))
                isfirst = False
            else:
                score = scorer(y_test, model._predict_precomputed(
                    indices=indices[:, :k], distances=distances[:, :k]))

            if result.get(k) is None:
                result[k] = np.array([score])
            else:
                result[k] = np.append(result[k], score)

    return result
