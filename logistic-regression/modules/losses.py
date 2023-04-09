import numpy as np
from scipy.special import expit


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """

        return np.mean((X @ w[1:] + w[0] - y)**2)

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """

        return 2 * np.mean(X.T @ (X @ w[1:]) + w[0] - y) + 2 * np.mean(X @ w[1:] + w[0] - y)


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """

        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : 2d numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """

        return np.mean(np.logaddexp(0, -y * (X @ w[1:] + w[0]))) + self.l2_coef * w[1:] @ w[1:].T

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : 2d numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """

        e = expit(y * (X @ w[1:] + w[0]))
        grad_bias = y @ (e - 1) / X.shape[0]
        grad_weights = X.T @ (y * (e - 1)) / X.shape[0] +  2 * self.l2_coef * w[1:]

        return np.r_[grad_bias, grad_weights]

