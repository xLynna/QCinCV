

# ------ Import necessary packages ----
import pickle
import itertools
import dwave.inspector
import neal
import numpy as np
from dimod.reference.samplers import ExactSolver
from dwave.system import (DWaveSampler, EmbeddingComposite,
                          VirtualGraphComposite)

def rowwise(N, W, c): 
    """Rowwise penalty model for the QGM problem
    
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

    optimizing = np.sum(np.abs(W + W.T), axis = 1, keepdims = True) - np.abs(np.diag(W)).reshape(-1, 1) + np.abs(c)
    MaxGrad= np.max(optimizing)
    
    # column and row-wise optimization:
    columnSum = np.tensordot(np.ones((1, N)), np.eye(N), axes=0).reshape(-1, N)
    rowSum = np.tensordot(np.eye(N), np.ones((1, N)), axes=0).reshape(N,-1).T
    
    Lambdaj = np.zeros((2,N))
    Lambdaj[0, :]= np.max(optimizing * columnSum, axis=0)
    Lambdaj[1, :]= np.max(optimizing * rowSum, axis=0)
    
    optimized_cs = (Lambdaj[0,:] + 1/2 * MaxGrad) * columnSum
    optimized_rs = np.tensordot(Lambdaj[1, :] + 1/2 * MaxGrad, np.ones((N, 1)), axes=0).reshape(-1, 1) * rowSum
    regularisationMatrix = optimized_cs @ columnSum.T + optimized_rs @ rowSum.T

    regularisationVector = -2 * np.sum(optimized_cs + optimized_rs, axis = 1, keepdims = True)
    
    W = W + regularisationMatrix # W is symmetric
    c = c + regularisationVector

    # switch x to {-1, 1}
    Q= W/4 
    np.fill_diagonal(Q, 0)
    qu= 0.5 * (c + np.sum(W, axis= 1, keepdims=True))

    bias=qu.reshape(N**2).tolist()
    indicies = np.arange(N**2)
    indices_pairs = itertools.product(indicies, indicies)
    J = dict(zip(indices_pairs, Q.flatten()))
    
    
   # sampler = ExactSolver()
    #response = sampler.sample_ising(bias,J)    
    
   # numberPrintedEnergies=0
   # for datum in response.data(['sample', 'energy']): 
   #     if numberPrintedEnergies<3:    
    #        print(datum.sample, datum.energy)
    #    numberPrintedEnergies=numberPrintedEnergies+1
    
    
  #  solver = neal.SimulatedAnnealingSampler()
  #  response = solver.sample_ising(bias, J, num_reads=50)
  #  Result=[]
   # for datum in response.data(['sample', 'energy', 'num_occurrences']):   
   #             print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
   #             Result.append([datum.sample,  datum.energy,  datum.num_occurrences]) 

    
    
    
   # sampler = EmbeddingComposite(DWaveSampler(annealing_time=40))
    
    # Local Solver
    solver = neal.SimulatedAnnealingSampler()
    response = solver.sample_ising(bias, J, num_reads=500)
    SimulatedResult=[]
    for datum in response.data(['sample', 'energy', 'num_occurrences']):   
        #print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
        SimulatedResult.append([datum.sample,  datum.energy,  datum.num_occurrences]) 

    # chain = np.max (bias)


    # sampler = EmbeddingComposite(DWaveSampler())
    # response = sampler.sample_ising(bias,J,chain_strength=chain  ,num_reads=500, return_embedding=True,anneal_schedule=((0.0,0.0),(40.0,0.5),(140.0,0.5),(180.0,1.0)))

    # dwave.inspector.show(response)
    # Result=[]
    # for datum in response.data(['sample', 'energy', 'num_occurrences','chain_break_fraction']):   
    #         print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
    #         Result.append([datum.sample,  datum.energy,  datum.num_occurrences, datum.chain_break_fraction])

    return [response.info,SimulatedResult]
