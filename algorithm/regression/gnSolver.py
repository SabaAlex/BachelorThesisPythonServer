import logging

import numpy as np
from numpy.linalg import pinv

from typing import Callable

logger = logging.getLogger(__name__)

class GNSolver:

    def __init__(self,
                fit_function : Callable,
                initial_values :  np.ndarray,
                xValues : np.ndarray,
                yValues : np.ndarray,
                tolerance_difference : float = 5,
                ):

        self.fit_function = fit_function
        self.tolerance_difference = tolerance_difference
        self.initial_values = initial_values
    
        self.coefficients = initial_values
        self.xValues = xValues
        self.yValues = yValues
    
        self.rmse = np.inf

    def get_coefficients(self):
        return self.coefficients

    def predict(self, x: np.ndarray):
        return self.fit_function(x, self.coefficients)

    def restart_fit_algorithm(self):
        self.rmse = np.inf
        self.coefficients = self.initial_values
    
    def fitNext(self): 
        residual = self.get_residual()

        jacobian = self._compute_jacobian(self.coefficients, step=10 ** (-6))

        self.coefficients = self.coefficients - self._compute_pseudoinverse(jacobian) @ residual

        rmse = np.sqrt(np.sum(residual ** 2))

        if self.tolerance_difference is not None:
            diff = np.abs(rmse - self.rmse)
            if diff < self.tolerance_difference:
                logger.info(f"RMSE difference {diff} between iterations smaller than tolerance. Fit terminated.")
                return True, self.coefficients

        self.rmse = rmse

        return False, self.coefficients

    def get_residual(self) -> np.ndarray:
        return self._compute_residual(self.coefficients)

    def _compute_residual(self, coefficients: np.ndarray) -> np.ndarray:
        y_fit = self.fit_function(self.xValues, coefficients)
        return y_fit - self.yValues

    def _compute_jacobian(self,
                            x0: np.ndarray,
                            step: float = 10 ** (-6)) -> np.ndarray:
        y0 = self._compute_residual(x0)

        jacobian = []
        for i, parameter in enumerate(x0):
            x = x0.copy()
            x[i] += step
            y = self._compute_residual(x)
            derivative = (y - y0) / step
            jacobian.append(derivative)
        jacobian = np.array(jacobian).T

        return jacobian

    @staticmethod
    def _compute_pseudoinverse(x: np.ndarray) -> np.ndarray:
        return pinv(x.T @ x) @ x.T