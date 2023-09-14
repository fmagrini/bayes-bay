#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:39:21 2023

@author: fabrizio
"""

class InitException(Exception):
    """
    Exception raised when a dispersion curve could not be extracted from the 
    data
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message
    
    
    
