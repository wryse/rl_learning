# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

# nn layer weight initializer

class Initializer(abc.ABC):
    @staticmethod 
    def xavier_normal(shape, has_bias):
        """Initialize weight using Xavier init.
        
        Args:
            shape (tuple): shape of weight metrix to be initialized
            has_bias (boolean): flags if bias terms are added
        
        Returns:
            Initialized weights and bias terms
        """
        prev_node_count, cur_node_count = shape
        weights = np.random.randn(prev_node_count, cur_node_count)/np.sqrt(prev_node_count)
        bias = np.random.randn(cur_node_count)/np.sqrt(prev_node_count) \
            if has_bias else np.zeros(cur_node_count)
        return weights, bias

    @staticmethod 
    def zeros(shape, has_bias):
        """Initialize weight to all zeros.
        
        Args:
            shape (tuple): shape of weight metrix to be initialized
            has_bias (boolean): flags if bias terms are added
        
        Returns:
            Initialized weights and bias terms
        """
        prev_node_count, cur_node_count = shape
        weights = np.zeros(shape)
        bias = np.zeros(cur_node_count)
        return weights, bias
