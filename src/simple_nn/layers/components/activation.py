# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

from .base import LayerComponent


# activation function


class Activation(LayerComponent):
    """Base class for activation functions."""
    
    def setup(self, **kwargs):
        """Nothing need to be setup for an activation function.
        
        Args:
            v (np.array): Data
        """
        pass
    
    def forward(self, v):
        """Apply activation function on the data.
        
        Args:
            v (np.array): Data
        
        Returns:
            Activated values
        """
        return self.activate(v)
    
    def back_propagation(self, prev_delta):
        """Back propagation to calculate gradient with.
        
        Args:
            prev_delta (np.array): delta values(derivatives) from the next layer
                wrt output of current layer
        
        Returns:
            Derivatives of current layer wrt input
        """
        return prev_delta * self.derivative()
    
    @abc.abstractmethod
    def activate(self, v):
        """Activate the data.
        
        Args:
            v (np.array): Data
        
        Returns:
            Activated values
        """
        pass
    
    @abc.abstractmethod
    def derivative(self):
        """Calculate derivative wrt input of activation function.
        
        The subclass need to implement the derivative with original data or activated data.
        
        Returns:
            Derivative wrt input of activation function
        """
        pass


class ActivationNone(Activation):
    """For not using an activation function"""
    
    def activate(self, v):
        """Return input as output as no activation needs to be applied.
        
        Args:
            v (np.array): Data
        
        Returns:
            The same as input v
        """
        return v
    
    def derivative(self):
        """Calculate derivative wrt input of activation function.
        
        Returns:
            Always 1
        """
        return 1


class ActivationSigmoid(Activation):
    """Sigmoid function. Data out of bounds will be clipped."""
    
    def __init__(self, x_upper_bound=500, x_lower_bound=-500):
        """Init function.
        
        Args:
            x_upper_bound (int): Upper bound of sigmoid input
            x_lower_bound (int): Lower bound of sigmoid input
        """
        super(ActivationSigmoid, self).__init__()
        self.x_upper_bound = x_upper_bound
        self.x_lower_bound = x_lower_bound
    
    def activate(self, v):
        """Apply sigmoid function to data.
        
        Args:
            v (np.array): Data
        
        Returns:
            1 / (1 + exp(v)). v will be clipped if bounds are set.
        """
        if self.x_upper_bound or self.x_upper_bound:
            return 1.0 / (1.0 + np.exp(-v.clip(max=self.x_upper_bound, min=self.x_lower_bound)))
        return 1.0 / (1.0 + np.exp(-v))
    
    def derivative(self):
        """Calculate derivative wrt input of activation function.
        
        Returns:
            activated * (1 - activated)
        """
        return self.output * (1 - self.output)


class ActivationRelu(Activation):
    """Relu."""
    
    def activate(self, v):
        """Apply Relu to data.
        
        Args:
            v (np.array): Data
        
        Returns:
            v if v > 0, otherwise 0
        """
        return v.clip(min=0)
    
    def derivative(self):
        """Calculate derivative wrt input of activation function.
        
        Returns:
            1 if activated > 0, otherwise 0
        """
        return np.where(self.output>0, 1, 0)


class ActivationTanh(Activation):
    """Tanh."""
    
    def activate(self, v):
        """Apply Tanh to data.
        
        Args:
            v (np.array): Data
        
        Returns:
            tanh(v)
        """
        return np.tanh(v)
    
    def derivative(self):
        """Calculate derivative wrt input of activation function.
        
        Returns:
            1 - activated^2
        """
        return 1 - self.output**2


class ActivationLeakyRelu(Activation):
    """Leaky Relu."""
    
    def __init__(self, alpha=0.3):
        """Init function. Set alpha value."""
        super(ActivationLeakyRelu, self).__init__()
        self.alpha = alpha
    
    def activate(self, v):
        """Apply Leaky Relu to data.
        
        Args:
            v (np.array): Data
        
        Returns:
            v if v > 0, otherwise 0
        """
        return np.where(v>=0, v, self.alpha*v)
    
    def derivative(self):
        """Calculate derivative wrt input of activation function.
        
        Returns:
            1 if activated > 0, otherwise 0
        """
        return np.where(self.output>=0, 1, self.alpha)


class ActivationSoftmax(Activation):
    """Softmax"""
    
    def activate(self, v):
        """Apply Softmax to data.
        
        Args:
            v (np.array): Data
        
        Returns:
            exp(v) / sum(exp(v))
        """
        v = v - v.max(axis=v.ndim-1, keepdims=True)
        return np.exp(v) / np.clip(np.sum(np.exp(v), axis=v.ndim-1, keepdims=True), 1e-8, None)
    
    def back_propagation(self, prev_delta):
        """Back propagation to calculate gradient with.
        
        Args:
            prev_delta (np.array): delta values(derivatives) from the next layer
                wrt output of current layer
        
        Returns:
            Derivatives of current layer wrt input
            Specific for Softmax: −∑k(yk/pk)*(∂pk/∂oi)
            Reference:
                https://deepnotes.io/softmax-crossentropy
                https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
        """
        m = self.output.shape[0]
        return np.tensordot(prev_delta, self.derivative(), (1, 2))[range(m), range(m)]
    
    def derivative(self):
        """Calculate derivative wrt input of activation function.
        
        Returns:
            activated * (1 - activated)
        """
        m, n = self.output.shape[0], self.output.shape[-1]
        p = np.repeat(self.output, n, axis=0).reshape(m,n,n)
        return p.transpose(0,2,1) * (np.eye(n) - p)

