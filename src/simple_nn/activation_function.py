# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')


# activation function


class ActivationFunction(abc.ABC):
    """Base class for activation functions."""
    
    def __init__(self):
        """Init function. Nothing needs to do in the base class."""
        pass
    
    @property
    @abc.abstractmethod
    def derivative_use_activated(self):
        """Boolean: If derivative of the activation function can be calculated
        using activated value.
        """
        pass
    
    @abc.abstractmethod
    def apply(self, v):
        """Apply activation function on the data.
        
        Args:
            v (np.array): Data
        
        Returns:
            Activated values
        """
        pass
    
    @abc.abstractmethod
    def derivative(self, v):
        """Calculate derivative wrt input of activation function.
        
        Args:
            v (np.array): input of activation function if derivative_use_activated is false,
                output of activation function otherwise
        
        Returns:
            Derivative wrt input of activation function
        """
        pass


class ActivationNone(ActivationFunction):
    """For not using an activation function"""
    
    def __init__(self):
        """Init function. Do nothing."""
        pass
    
    @property
    def derivative_use_activated(self):
        """Arbitrary. Not used."""
        return True
    
    def apply(self, v):
        """Return input as output as no activation needs to be applied.
        
        Args:
            v (np.array): Data
        
        Returns:
            The same as input v
        """
        return v
    
    def derivative(self, v):
        """Calculate derivative wrt input of activation function.
        
        Args:
            v (np.array): input of activation function if derivative_use_activated is false,
                output of activation function otherwise
        
        Returns:
            Always 1
        """
        return 1


class ActivationSigmoid(ActivationFunction):
    """Sigmoid function. Data out of bounds will be clipped."""
    
    def __init__(self, x_upper_bound=None, x_lower_bound=None):
        """Init function.
        
        Args:
            x_upper_bound (int): Upper bound of sigmoid input
            x_lower_bound (int): Lower bound of sigmoid input
        """
        self.x_upper_bound = x_upper_bound
        self.x_lower_bound = x_lower_bound
    
    @property
    def derivative_use_activated(self):
        """Use sigmoid result to calculate derivative."""
        return True
    
    def apply(self, v):
        """Apply sigmoid function to data.
        
        Args:
            v (np.array): Data
        
        Returns:
            1 / (1 + exp(v)). v will be clipped if bounds are set.
        """
        if self.x_upper_bound or self.x_upper_bound:
            return 1.0 / (1.0 + np.exp(-v.clip(max=self.x_upper_bound, min=self.x_lower_bound)))
        return 1.0 / (1.0 + np.exp(-v))
    
    def derivative(self, v):
        """Calculate derivative wrt input of activation function.
        
        Args:
            v (np.array): input of activation function if derivative_use_activated is false,
                output of activation function otherwise
        
        Returns:
            v * (1 - v)
        """
        return v * (1 - v)


class ActivationRelu(ActivationFunction):
    """Relu function."""
    def __init__(self):
        """Init function. Do nothing."""
        pass
    
    @property
    def derivative_use_activated(self):
        """Use result to calculate derivative."""
        return True
    
    def apply(self, v):
        """Apply Relu to data.
        
        Args:
            v (np.array): Data
        
        Returns:
            v if v > 0, otherwise 0
        """
        return v.clip(min=0)
    
    def derivative(self, v):
        """Calculate derivative wrt input of activation function.
        
        Args:
            v (np.array): input of activation function if derivative_use_activated is false,
                output of activation function otherwise
        
        Returns:
            1 if v > 0, otherwise 0
        """
        return np.where(v>0, 1, 0)

