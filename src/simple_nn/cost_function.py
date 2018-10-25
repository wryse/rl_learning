# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

# cost function


class CostFunction(abc.ABC):
    
    @abc.abstractmethod
    def loss(self, y, y_predict):
        """Calculate cost values for generated results.
        
        Args:
            y (np.array): ground truth values
            y_predict (np.array): generated results
        
        Returns:
            Loss between ground truth and generated results
        """
        pass
    
    @abc.abstractmethod
    def derivative(self, y, y_predict):
        """Calculate cost values for generated results.
        
        Args:
            y (np.array): ground truth values
            y_predict (np.array): generated results
        
        Returns:
            Derivative of cost function using ground truth and generated results
        """
        pass


class SquaredLoss(CostFunction):
    """Squared loss."""
    
    def loss(self, y, y_predict):
        """Calculate cost values using squared loss.
        
        Args:
            y (np.array): ground truth values
            y_predict (np.array): generated results
        
        Returns:
            sum((y_predict - y)^2)
        """
        return (np.power(y_predict - y, 2) / 2).sum(axis=1)
    
    def derivative(self, y, y_predict):
        """Calculate cost values for generated results.
        
        Args:
            y (np.array): ground truth values
            y_predict (np.array): generated results
        
        Returns:
            y_predict - y
        """
        return y_predict - y


class CrossEntropy(CostFunction):
    """Cross entropy for multi-category."""
    
    def loss(self, y, y_predict):
        """Calculate cost values using squared loss.
        
        Args:
            y (np.array): ground truth values
            y_predict (np.array): generated results
        
        Returns:
            sum(-y * log(y_predict))
        """
        return (-y*np.log(np.clip(y_predict, 1e-8, None))).sum(axis=1)
    
    def derivative(self, y, y_predict):
        """Calculate cost values for generated results.
        
        Args:
            y (np.array): ground truth values
            y_predict (np.array): generated results
        
        Returns:
            -y / y_predict
        """
        return -y / np.clip(y_predict, 1e-8, None)


class SigmoidCrossEntropy(CostFunction):
    """Cross entropy for binary category only."""
    
    def loss(self, y, y_predict):
        """Calculate cost values using squared loss.
        
        Args:
            y (np.array): ground truth values
            y_predict (np.array): generated results
        
        Returns:
            sum(-y * log(y_predict))
        """
        return -y*np.log(np.clip(y_predict, 1e-8, None)) \
                   -(1-y)*np.log(np.clip(1-y_predict, 1e-8, None))
    
    def derivative(self, y, y_predict):
        """Calculate cost values for generated results.
        
        Args:
            y (np.array): ground truth values
            y_predict (np.array): generated results
        
        Returns:
            (y_predict-y) / (y_predict*(1-y_predict))
        """
        y_predict = np.clip(y_predict, 1e-8, 1-1e-8)
        return (y_predict - y) / (y_predict*(1-y_predict))
