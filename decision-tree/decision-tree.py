import numpy as np
from collections import deque

class Criterion:
    def get_best_split(self, feature, target):
        """
        Parameters
        ----------
        feature : feature vector, np.ndarray.shape = (n_samples, )
        target  : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        threshold : value to split feature vector, float
        q_value   : impurity improvement, float
        """
        
        indicies = feature.argsort()
        target_srt = target[indicies]
        feature_srt = feature[indicies]
        N = len(target)

        q_best = -np.inf
        i_best = None
        
        h_0 = self.score(target)
        equal_features = feature_srt[:-1] == feature_srt[1:]
        
        for i in range(1, N):
            if equal_features[i - 1]:
                q = 0
            else:
                h_l = self.score(target_srt[:i])
                h_r = self.score(target_srt[i:])
                q = h_0 - i/N * h_l - (N-i)/N * h_r
            
            if q > q_best:
                q_best = q
                i_best = i
        
        thrl = (feature_srt[i_best] + feature_srt[i_best - 1]) / 2
        
        return thrl, q_best
        

    def score(self, target):
        """
        Parameters
        ----------
        target : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        impurity : float
        """

        raise NotImplementedError

    def get_predict_val(self, target):
        """
        Parameters
        ----------
        target : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        prediction :
            - classification: probability distribution in node, np.ndarray.shape = (n_classes, )
            - regression: best constant approximation, float
        """

        raise NotImplementedError


class GiniCriterion(Criterion):
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def get_predict_val(self, classes):
        pred = np.bincount(classes, minlength = self.n_classes) / len(classes)

        return pred
        
    def score(self, classes):
        pred = self.get_predict_val(classes)

        return 1 - (pred ** 2).sum()
        


class EntropyCriterion(Criterion):
    EPS = 1e-6
    
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def get_predict_val(self, classes):
        pred = np.bincount(classes, minlength = self.n_classes) / len(classes)

        return pred

    def score(self, classes):
        pred = self.get_predict_val(classes)

        return (-pred*np.log(pred + self.EPS)).sum()


class MSECriterion(Criterion):
    def get_predict_val(self, target):
        return np.sum(target) / target.shape[0]

    def score(self, target):
        c = self.get_predict_val(target)

        return np.sum((target - c) ** 2) / target.shape[0]


class TreeNode:
    def __init__(self, impurity, predict_val, depth):
        self.impurity = impurity        # node impurity
        self.predict_val = predict_val  # prediction of node
        self.depth = depth              # current node depth
        
        self.feature = None             # feature to split
        self.threshold = None           # threshold to split
        self.improvement = -np.inf      # node impurity improvement after split
        
        self.child_left = None
        self.child_right = None
    
    @property
    def is_terminal(self):
        return self.child_left is None and self.child_right is None
    
    @classmethod
    def get_best_split(cls, X, y, criterion):
        '''
        Finds best split for current node
        
        Parameters
        ----------
        X : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        
        Returns
        -------
        feature   : best feature to split,  int
        threshold : value to split feature, float
        q_value   : impurity improvement,   float
        '''

        q_best = - np.inf
        t_best = None
        f_best = None
        
        for fi in range(X.shape[1]):
            thr, q = criterion.get_best_split(X[:,fi], y)

            if q > q_best:
                q_best = q
                t_best = thr
                f_best = fi
    
        return f_best, t_best, q_best
        
        
    
    def get_best_split_mask(self, X):
        '''
        Parameters
        ----------
        X : samples in node, np.ndarray.shape = (n_samples, n_features)
        
        Returns
        -------
        right_mask : indicates samples in right node after split
            np.ndarray.shape = (n_samples, )
            np.ndarray.dtype = bool
        '''

        return X[:, self.feature] >= self.threshold
    
    def split(self, X, y, criterion, **split_params):
        '''
        Split current node
        
        Parameters
        ----------
        X : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion : criterion to split by, Criterion
        split_params : result of get_best_split method
        
        Returns
        -------
        right_mask : indicates samples in right node after split
            np.ndarray.shape = (n_samples, )
            np.ndarray.dtype = bool
            
        child_left  : TreeNode
        child_right : TreeNode
        '''
        
        self.feature = split_params['feature']
        self.threshold = split_params['threshold']
        self.improvement = split_params['improvement']
        
        mask_right = self.get_best_split_mask(X)
        self.child_left = self.from_criterion(y[~mask_right], criterion, self.depth + 1)
        self.child_right = self.from_criterion(y[mask_right], criterion, self.depth + 1)
        
        return mask_right, self.child_left, self.child_right
        

    @classmethod
    def from_criterion(cls, y, criterion, depth=0):
        return cls(
            impurity=criterion.score(y),
            predict_val=criterion.get_predict_val(y),
            depth=depth,
        )


class DecisionTree:
    def __init__(self, max_depth=None, min_leaf_size=None, min_improvement=None):
        self.criterion = None
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_improvement = min_improvement

    def _build_nodes(self, X, y, criterion, indices, node):
        '''
        Builds tree recursively
        
        Parameters
        ----------
        X : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion : criterion to split by, Criterion
        indices : samples' indices in node,
            np.ndarray.shape = (n_samples, )
            nd.ndarray.dtype = int
        node : current node to split, TreeNode
        '''
        
        if self.max_depth is not None and node.depth >= self.max_depth:
            return 
        
        if self.min_leaf_size is not None and self.min_leaf_size > len(indices):
            return 
        
        if np.unique(y[indices]).shape[0] <= 1:
            return
        
        X_node = X[indices]
        y_node = y[indices]
        
        feature, threshold, improvement = node.get_best_split(X_node, y_node, criterion)
        
        if self.min_improvement is not None and self.min_improvement > improvement:
            return
        
        mask_right, child_left, child_right = node.split(
            X_node, y_node, criterion,
            feature = feature,
            threshold = threshold,
            improvement = improvement
        )
        
        self._build_nodes(X, y, criterion, indices[~mask_right], child_left)
        self._build_nodes(X, y, criterion, indices[mask_right], child_right)
        
        
    def _get_nodes_predictions(self, X, predictions, indices, node):
        '''
        Builds tree recursively
        
        Parameters
        ----------
        X : samples in node, np.ndarray.shape = (n_samples, n_features)
        predictions : result matrix to be feild,
            - classification : np.ndarray.shape = (n_samples, n_classes)
            - regression : np.ndarray.shape = (n_samples, )
        indices : samples' indices in node,
            np.ndarray.shape = (n_samples, )
            nd.ndarray.dtype = int
        node : current node to split, TreeNode
        '''

        if node.is_terminal:
            predictions[indices] = node.predict_val
            return

        X_node = X[indices]
        mask_right = node.get_best_split_mask(X_node)

        self._get_nodes_predictions(X, predictions, indices[mask_right], node.child_right)
        self._get_nodes_predictions(X, predictions, indices[~mask_right], node.child_left)

    
    @property
    def feature_importances_(self):
        '''
        Returns
        -------
        importance : cummulative improvement per feature, np.ndarray.shape = (n_features, )
        '''
        
        importance = np.zeros(self.n_features_)
        quene = deque()
        quene.append(self.root_)
        
        while len(quene):
            node = quene.popleft()

            if node.is_terminal:
                continue
                
            importance[node.feature] += node.improvement
            quene.append(node.child_left)
            quene.append(node.child_right)
            
        return importance
    

class ClassificationDecisionTree(DecisionTree):
    def __init__(self, criterion='gini', **kwargs):
        super().__init__(**kwargs)
        
        if criterion not in ('gini', 'entropy', ):
            raise ValueError('Unsupported criterion', criterion)
        
        self.criterion = criterion
            
    def fit(self, X, y):
        self.n_classes_ = np.max(y) + 1
        self.n_features_ = X.shape[1]
        
        if self.criterion == 'gini':
            criterion = GiniCriterion(n_classes=self.n_classes_)
        elif self.criterion == 'entropy':
            criterion = EntropyCriterion(n_classes=self.n_classes_)
        else:
            raise ValueError('Unsupported criterion', criterion)

        self.root_ = TreeNode.from_criterion(y, criterion)
        self._build_nodes(X, y, criterion, np.arange(X.shape[0]), self.root_)

        return self
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    def predict_proba(self, X):
        probas = np.zeros(shape=(X.shape[0], self.n_classes_))
        self._get_nodes_predictions(X, probas, np.arange(X.shape[0]), self.root_)

        return probas
    

class RegressionDecisionTree(DecisionTree):
    def __init__(self, criterion='mse', **kwargs):
        super().__init__(**kwargs)
        
        if criterion != 'mse':
            raise ValueError('Unsupported criterion', criterion)
        
        self.criterion = criterion
            
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        
        if self.criterion == 'mse':
            criterion = MSECriterion()
        else:
            raise ValueError('Unsupported criterion', criterion)

        self.root_ = TreeNode.from_criterion(y, criterion)
        self._build_nodes(X, y, criterion, np.arange(X.shape[0]), self.root_)

        return self
    
    def predict(self, X):
        prediction = np.zeros(shape=X.shape[0])
        self._get_nodes_predictions(X, prediction, np.arange(X.shape[0]), self.root_)

        return prediction