#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:39:21 2023

@author: fabrizio
"""

class InitException(Exception):
    """
    Exception raised when users try to access a certain field that hasn't been
    intialized yet
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message
    
    
    
class ForwardException(Exception):
    """
    Exception raised when the user-provided forward function raises an error
    """
    
    def __init__(self, original_exc):
        self.message = original_exc.message if hasattr(original_exc, "message") \
            else "Error occurred when running the forward function"
        super().__init__(self.message)
    
    def __str__(self):
        return self.message


class DimensionalityException(Exception):
    """
    Exception raised when trying to add/remove a Voronoi site in a 
    parameterization that has already reached the maximum/minimum number of
    allowed voronoi cells
    """
    
    def __init__(self, move):
        message = ("Error occured when trying to %s a Voronoi site."
                   " The %s number of Voronoi cells had already been reached.")
        words = ("add", "maximum") if move=="Birth" else ("remove", "minimum")
        self.message = message%words            
        super().__init__(self.message)
    
    def __str__(self):
        return self.message
    
    
