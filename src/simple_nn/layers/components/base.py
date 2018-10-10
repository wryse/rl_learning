# coding: utf-8

import abc

# Base class of layer components


class LayerComponent(abc.ABC):
    
    def __init__(self):
        """Init function.
        
        Attributes:
            input (np.array): last input data
            output (np.array): last output data
        """
        self.c_id = id(self)
        self.input = None
        self.output = None
    
    @abc.abstractmethod
    def setup(self, **kwargs):
        """Setup runtime parameters.
        
        Args:
            v (np.array): Data
        """
        pass
    
    def fit_forward(self, v):
        """Forward data for learning period. Input and output will be kept.
        
        Args:
            v (np.array): Data
        
        Returns:
            Forward results
        """
        self.input = v
        self.output = self.forward(v)
        return self.output
    
    @abc.abstractmethod
    def forward(self, v):
        """Forward data.
        
        Args:
            v (np.array): Data
        
        Returns:
            Forward results
        """
        pass
    
    @abc.abstractmethod
    def back_propagation(self, prev_delta):
        """Back propagation to calculate gradient in last forward.
        
        Args:
            prev_delta (np.array): delta values(derivatives) from the next step
                wrt output of current step
        
        Returns:
            Derivatives of current step wrt input
        """
        pass
    
    def update(self):
        """Update current component. Only necessary for part of components such as Dense.
        """
        pass

