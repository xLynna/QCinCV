#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:39:52 2020

@author: cv_marcel
"""


# ------ Import necessary packages ----
import pickle
import itertools
import dwave.inspector
import neal
import numpy as np
from dimod.reference.samplers import ExactSolver
from dwave.system import (DWaveSampler, EmbeddingComposite,
                          VirtualGraphComposite)


def baseline(N, W, c):
    """Baseline model for the QGM problem
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph. The side length of the square permutation 
        matrix X.
    W : numpy.ndarray, size of (N**2, N**2), symmetric
        Matrix of weights.
    c : numpy.ndarray, size of (N**2, 1)
        Vector of biases.

    Returns
    -------
    Result : list
        List of results from the D-Wave sampler.
    response.info : dict
        Dictionary with information about the D-Wave sampler.
    SimulatedResult : list
        List of results from the simulated annealing sampler.
    
    """

    # lower bound of penalty parameter
    lambda0 = np.sum(np.abs(W)) + np.sum(np.abs(c)) 
    lambda0 /= 2

    # constraints
    a1 = np.tensordot(np.eye(N), np.ones((1, N)), axes=0).reshape(N,-1)
    a2 = np.tensordot(np.ones((1, N)), np.eye(N), axes=0).reshape(-1, N).T
    A = np.vstack((a1, a2))
    b = np.ones((2*N, 1))

    # regularisation matrix and vector
    regularisationMatrix = lambda0 * A.T @ A
    regularisationVector = -2 * lambda0 * (b.T @ A).reshape(-1, 1)

    W += regularisationMatrix # W is symmetric
    c += regularisationVector

    # switch x to {-1, 1}
    Q= W/4 
    np.fill_diagonal(Q, 0)
    qu= 0.5 * (c + np.sum(W, axis= 1, keepdims=True))

    bias=qu.reshape(N**2).tolist()
    indicies = np.arange(N**2)
    indices_pairs = itertools.product(indicies, indicies)
    J = dict(zip(indices_pairs, Q.flatten()))
    
    #sampler = ExactSolver()
   # response = sampler.sample_ising(bias,J)    
    
   # numberPrintedEnergies=0
  #  for datum in response.data(['sample', 'energy']): 
   #     if numberPrintedEnergies<3:    
           # print(datum.sample, datum.energy)
    #    numberPrintedEnergies=numberPrintedEnergies+1
    
    
   # solver = neal.SimulatedAnnealingSampler()
   # response = solver.sample_ising(bias, J, num_reads=50)
   # Result=[]
   # for datum in response.data(['sample', 'energy', 'num_occurrences']):   
   #             print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
   #             Result.append([datum.sample,  datum.energy,  datum.num_occurrences]) 

    
    # Local Solver
    solver = neal.SimulatedAnnealingSampler()
    response = solver.sample_ising(bias, J, num_reads=500)
    SimulatedResult=[]
    for datum in response.data(['sample', 'energy', 'num_occurrences']):   
                print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
                SimulatedResult.append([datum.sample,  datum.energy,  datum.num_occurrences]) 
    

    # chain = np.max(bias)

    # sampler = EmbeddingComposite(DWaveSampler())
    # response = sampler.sample_ising(bias, J, num_reads=500, chain_strength=chain, return_embedding=True, anneal_schedule=((0.0,0.0),(40.0,0.5),(140.0,0.5),(180.0,1.0)))

    # dwave.inspector.show(response)
    # Result=[]
    # for datum in response.data(['sample', 'energy', 'num_occurrences','chain_break_fraction']):   
    #         print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
    #         Result.append([datum.sample,  datum.energy,  datum.num_occurrences, datum.chain_break_fraction])

    # return [Result, response.info, SimulatedResult]
    return [response.info, SimulatedResult]


