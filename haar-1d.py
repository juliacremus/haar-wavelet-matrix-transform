#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haar 1D

Created on Mon Jul 17 20:13:08 2023

@author: julia
"""

#%% bib

import numpy as np
import matplotlib.pyplot as plt

#%% parameters 

f = 1 / np.sqrt(2)

N = 2500

v = f * np.sin( 2 * np.linspace(0, np.pi, N))

#%% Algorithm

"""
j = 0

for i into range(0, N):
    
    if i < N/2 :
        
        w[i] = v[j] + v[j+1]
        
        j += 2
        
    if i >= N / 2:
        
        if i == N/2:
            
            j = 0
            
        w[i] = v[j] - v[j+1]

"""

#%% Code

w = np.zeros(N)

j = 0

for i in range(0, N):
    
    if i < N/2 :
        
        w[i] = v[j] + v[j+1]
        
        j += 2
        
    if i >= N / 2:
        
        if i == N/2:
            
            j = 0
            
        w[i] = v[j] - v[j+1]
        
        j += 2
        
#%% Plot result

fig, ax = plt.subplots()

ax.plot(v, label='Initial vector')
ax.plot(w, label='Transformed vector')

ax.legend()