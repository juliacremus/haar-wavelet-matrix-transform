#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haar generalized

Created on Wed Jul 19 20:55:39 2023

@author: julia
"""

#%% bib

import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% Haar Transform

class HaarTransform:
    
    def __init__(self, input_vec, decomp_level, norm_component):
        
        self.input_vec = input_vec
        
        self.decomposition_level = decomp_level
        
        self.norm_factor = norm_component
        
        self.inputs_parameters()
        
        self.haar_matrix, self.inv_haar_matrix = self.build_haar_matrix()
        
    def inputs_parameters(self):
        """
        Function to get input vectors 

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.input_vec_shape = np.shape(self.input_vec)
        
        # checks if is np array or, at least, a list
       
        if np.isscalar(self.input_vec) or type(self.input_vec) == 'string':
            raise Exception('Input vector must be an numpy array or a list, type given: %s' % (type(self.input_vec)))
        
        self.is_2D = True if len(self.input_vec_shape) > 1 else False
        
        if self.is_2D and (self.input_vec_shape[0] != self.input_vec_shape[1]):
            
            raise Exception('The input vector should have N x N dimensions if 2D!')
            
        # if 2D have to have N, N
            
        self.N = self.input_vec_shape[0]
        
        return 
    
    def build_haar_matrix(self, N = 0):
        """
        Build matrix T with detail and means 

        Returns
        -------
        None.

        """
        
        # If N is not given by user, use self.N
        N = self.N if N == 0 else math.ceil(N)
        
        # Start matrix to be completed with values
        haar_matrix = np.zeros(shape = (N, N))
        
        j = 0
               
        for i in range(0, N):
            
            if i < N / 2 :
                
                haar_matrix[i, 2 * j] = 1
                haar_matrix[i, 2 * j + 1] = 1
                
            if i >= N / 2:
                
                if i == N/2:
                    
                    j = 0
                    
                haar_matrix[i, 2 * j] = 1
                haar_matrix[i, 2 * j + 1] = -1
                
            j += 1
            
        # Set the inverse

        # inv_haar_matrix = np.linalg.inv(haar_matrix)
        
        return self.norm_factor * haar_matrix, 1
    
      
    def run_foward_transform(self):
        
        output_vector = np.zeros(shape=self.input_vec_shape)
        
        if self.is_2D:
            
            output_vector = self.haar_matrix @ self.input_vec @ self.inv_haar_matrix
            
        else:
            
            output_vector = self.haar_matrix @ self.input_vec 
        
        return output_vector 
    
    def run_inverse_transform(self):
        
        output_vector = np.zeros(shape=self.input_vec_shape)
        
        if self.is_2D:
            
            output_vector = self.inv_haar_matrix @ self.input_vec @ self.haar_matrix
            
        else:
            
            output_vector = self.inv_haar_matrix @ self.input_vec 
        
        return output_vector
    
    def build_multi_resolution_matrix(self, k):
        
        """
        Build H_{N, 2^k} matrix 
        -----------------------
        
        Parameters:
            k (int): iteration of producer, number of non null values
        
        
        """
     
        # Initialize H_{N, 2k} 
        partial_hn = np.zeros(shape = (self.N, self.N))
               
        # Get T matrix
        haar_matrix, inv = self.build_haar_matrix(np.power(2, k))
        
        # Complete upper right corner 
        partial_hn[
            0:math.ceil(np.power(2, k)), 0:math.ceil(np.power(2, k))
        ] = haar_matrix.copy()
        
        # Complete lower left corner
       
        partial_hn[
            -math.ceil(self.N - np.power(2, k))-1:-1,  
            -math.ceil(self.N - np.power(2, k))-1:-1
        ] = 1
        
        # fig, ax = plt.subplots()
        # plt.imshow(partial_hn, cmap='hot', interpolation='nearest')
        # plt.show()
                
        return partial_hn
    
    
    def run_cascade_multiresolution_transform(self):
        
               
        # Define start and stop of loop
        
        loop_from = np.log2(self.N) + 1 - self.decomposition_level
        loop_to = np.log2(self.N) - 1
        
        # Start temp variable for producer
        temp_Hn = self.build_multi_resolution_matrix(loop_from)
        
        i = loop_from
        
        while i <= loop_to:
        
            print(i)    
        
            partial_hn = self.build_multi_resolution_matrix(i)
            
            temp_Hn = np.copy(np.dot(temp_Hn, partial_hn))
            
            i += 1

        
        final_haar, a = self.build_haar_matrix(self.N)
        
        final_multi_matrix = np.dot(partial_hn, final_haar)
        
        output = np.dot(final_multi_matrix, self.input_vec )
        
        return output 
    

    def build_packet_multi_resolution_matrix(self, k):
        
        packet_multi_matrix = np.zeros(shape = (self.N, self.N))
        
        two_in_k = np.power(2, k)
        
        # Get T matrix
        haar_matrix = self.build_haar_matrix(np.power(2, k))
        
        i = 0
        
        while i < (self.N - two_in_k):
            
            packet_multi_matrix[
                    i : i + two_in_k, 
                    i : i + two_in_k
                ] = haar_matrix.copy()
            
            i += two_in_k
        
        
        return 
    
    @staticmethod
    def run_non_decimated_tranform(self):
        
        return 
    
#%% Testes de script

#%% parameters / inputs from user

f = 1 

N = 1024

v = f * np.sin(5 * np.linspace(0, np.pi, N)) 

v += np.random.rand(N)

levels = 4

haar = HaarTransform(v, levels, f)

#%%

# foward_transform_array = haar.run_foward_transform()
multi_resolution = haar.run_cascade_multiresolution_transform()

#%%

fig, ax = plt.subplots()

ax.plot(v, label='original signal')
ax.plot(multi_resolution, label='transformed signal', marker='o')

ax.legend()