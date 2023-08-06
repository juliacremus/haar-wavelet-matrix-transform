#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascade algo - Haar MRA

Created on Mon Jul 17 20:45:41 2023

@author: julia
"""

#%% import bibs

import numpy as np
import matplotlib.pyplot as plt

#%% parameters

f = 1 / np.sqrt(2)

N = 2500

v = f * np.sin( 2 * np.linspace(0, np.pi, N))

decomposition_level = 5

#%% algorithm

"""

- first, we need to build the matrix H'_(N, 2^k) and make the multiplication

k = np.log2(N) + 1 - l
nf = np.log2(N) - 1



"""

#%%

k = np.log2(N) + 1 - decomposition_level
nf = np.log2(N) - 1

#%%

multiplier_result = np.zeros((N, N))

while k < nf:
    
    H_2k = 
    
    multiplier_result = 1
    
    
    k += 1 

#%% plot results