from dwave.system import DWaveSampler, EmbeddingComposite, VirtualGraphComposite

import neal
import numpy as np
from dimod.reference.samplers import ExactSolver
import dwave.inspector

import pickle

#sum of Pauli matrices equals constant. 
#Since variables were inserted we are in a lower dimension


def reducedWandC(N, W, c):
    """Returns the reduced W matrix.
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph. The side length of the square permutation
        matrix X.
    W : numpy.ndarray, size of (N**2, N**2), symmetric
        Matrix of weights.
    
    Returns
    -------
    W_reduced : numpy.ndarray, size of ((N-1)**2, (N-1)**2), symmetric
        Matrix of weights in the reduced dimension.

    """

    # The new reduced matrix composes of 4 parts (divided by columns) of the 
    # original matrix, each part consists of 4 subparts (divided by rows).
    # The parts are divided due to how we reudced the dimension of the matrix.
    # And we vectorise the matrix

    ONE = np.array([0])
    ROWS = np.array(range(1, N)) # if symmetrical, this is columns
    COLUMNS = np.array(range(N, N**2, N)) # if symmetrical, this is rows
    REST = np.delete(np.array(range(N, N**2)), COLUMNS-N)

    # We highly manipulate the fact of symmetry to accelerate the fetch of data.
    # Variation from the first column of original W

    #REST, we start here first to define the shape of W_reduced
    ROW_REST = W[REST, :]
    W_reduced = ROW_REST[:, REST] \
                - np.kron(np.ones(N-1), ROW_REST[:, ROWS]) \
                - np.repeat(ROW_REST[:, COLUMNS], N-1, axis = 1)
    
    # integrate the weight of the first column(row) into the reduced matrix
    # broadcating
    ROW_ONE = W[ONE, :]
    W_reduced += ROW_ONE[:, REST] \
                - np.kron(np.ones(N-1), ROW_ONE[:, ROWS]) \
                - np.repeat(ROW_ONE[:, COLUMNS], N-1, axis = 1)
    
    # second to N columns(rows)
    ROW_2_N = W[ROWS, :]
    W_reduced -= np.kron(np.ones((N-1, 1)), 
                            ROW_2_N[:, REST]
                            - np.kron(np.ones(N-1), ROW_2_N[:, ROWS])
                            - np.repeat(ROW_2_N[:, COLUMNS], N-1, axis = 1))
    
    # First column(row) of each batch, corresponding to the summation of the first row
    ROW_EVERY_N = W[COLUMNS, :]
    W_reduced -= np.repeat(ROW_EVERY_N[:, REST]
                            - np.kron(np.ones(N-1), ROW_EVERY_N[:, ROWS])
                            - np.repeat(ROW_EVERY_N[:, COLUMNS], N-1, axis = 1)
                            , N-1, axis = 0)
    
    W_reduced += W[ONE, ONE] - (np.kron(np.ones(N-1), W[ONE, ROWS]) + np.repeat(W[ONE, COLUMNS], N-1) - W[REST, ONE]).reshape(-1, 1)

    # From the linear term during W reduction
    cols_W = (2 - N) * W[:, ONE] + np.sum(W[:, ROWS] + W[:, COLUMNS], axis = 1, keepdims = True)
    C_reduced = cols_W[REST] - np.kron(np.ones((N-1, 1)), cols_W[ROWS]) - np.repeat(cols_W[COLUMNS], N-1, axis = 0) + cols_W[ONE]

    # Reduce C
    C_reduced = 2 * C_reduced + c[ONE] - np.kron(np.ones((N-1, 1)), c[ROWS]) - np.repeat(c[COLUMNS], N-1, axis = 0) + c[REST]

    return W_reduced, C_reduced
     

def newPauli(position, N):
    """Returns the Pauli matrix of a given position in the permutation matrix X.
    
    Parameters
    ----------
    position : int
        Position of the Pauli matrix in the permutation matrix X.
    N : int
        Number of nodes in the graph. The side length of the square permutation
        matrix X.
    
    Returns
    -------
    result : numpy.ndarray, size of (N**2, )

    """
    if position == 0:
        result = np.ones(((N-1)**2 + 1))
        result[(N-1)**2] = 2-N

    elif position < N:
        pos_vec = np.zeros(N-1)
        pos_vec[position-1] = -1
        result = np.hstack((np.tensordot(np.ones(N-1), pos_vec, axes=0).reshape(-1), 1))

    elif position % N==0:
        pos_vec = np.zeros(N-1)
        pos_vec[position//N-1] = -1
        result = np.hstack((np.repeat(pos_vec, N-1), 1))
    
    else:
        result = np.zeros(((N-1)**2 + 1))
        result[position - position//N - N]= 1 # 

    return result


   
def inserted(N,W,c):

    Wnew=np.zeros(((N-1)**2,(N-1)**2))
    cnew= np.zeros((N-1)**2)
    
    
    
    
    for i in range(N**2):
        for j in range(N**2):
            Wnew += W[i,j]* np.kron(newPauli(i, N)[0:(N-1)**2].reshape(1,(N-1)**2).T, newPauli(j, N )[0: (N-1)**2].reshape(1,(N-1)**2))
            cnew += W[i,j]* newPauli(i, N)[(N-1)**2] * newPauli(j, N )[0:(N-1)**2]
            cnew += W[i,j]* newPauli(i, N)[0:(N-1)**2]  * newPauli(j, N )[(N-1)**2]       
    
    
        cnew+= c[i,0]* newPauli(i,N)[0:(N-1)**2]
    
    
    
    
  
    #rowwise
    
    optimizing=np.zeros((N-1)**2)
    
    for k in range((N-1)**2):
        optimizing[k]+= -np.abs(Wnew[k,k]) + np.abs(cnew[k]) #Minus because in the next line we get += 2W[k,k]
        for i in range((N-1)**2):
    
            optimizing[k]+=np.abs( Wnew[k,i]+ Wnew[i,k])
    
    
    
    MaxGrad= np.max(optimizing)
    
    # row and collumn-wise optimization
    Lambdaj=  np.zeros((2,N))
    
    
    
    
    columnSum= np.zeros(((N-1)**2,N-1))
    rowSum= np.zeros(((N-1)**2,N-1))
    for i in range(N-1):
        for j in range(N-1):
            for k in range(N-1):
    
                if j==k:
                    columnSum[(N-1)*i+j,k]=  1        
                if i==k:
                    rowSum[(N-1)*i+j,k]=1
    
    for j in range(N-1):
        Lambdaj[0,j]= np.max(optimizing* columnSum[:,j])
        Lambdaj[1,j]= np.max(optimizing* rowSum[:,j])
    
    
    
    regularisationMatrix= np.zeros(((N-1)**2,(N-1)**2 ))
    regularisationVector= np.zeros(((N-1)**2,1 ))
    
    
    
    for i in range(N-1):
                regularisationMatrix +=( 1/2 *Lambdaj[0,i] + 1/2 *MaxGrad)* columnSum[:,i].reshape((N-1)**2,1)@  columnSum[:,i].reshape(1,(N-1)**2)
                regularisationVector += -( ( 1/2 * Lambdaj[0,i] + 1/2* MaxGrad) * columnSum[:, i ]).reshape(((N-1)**2,1))
                regularisationMatrix +=( 1/2 *Lambdaj[1,i] + 1/2 *MaxGrad)* rowSum[:,i].reshape((N-1)**2,1)@  rowSum[:,i].reshape(1,(N-1)**2)
                regularisationVector += -( ( 1/2 *Lambdaj[1,i] + 1/2* MaxGrad) * rowSum[:, i ]).reshape(((N-1)**2,1))
    
    # diff
    regularisationMatrix+= MaxGrad/2 * np.ones(((N-1)**2,(N-1)**2))
    regularisationVector+= -MaxGrad/2 * (N-1+N-2 ) * np.ones(( (N-1)**2,1) )
    
    
    
    
    
    
    Wnew= Wnew+  regularisationMatrix
    cnew= cnew+  regularisationVector.T
    Qnew= Wnew/4 
    
    qunew= cnew.T/2 + (np.sum( Wnew, axis=0, keepdims= True ).T + np.sum(Wnew,axis= 1, keepdims=True) )/4
    bias=qunew.reshape((N-1)**2).tolist()
    
    J={}
    
    for i in range((N-1)**2):
    
        for j in range((N-1)**2):
    
            J.update( {(i,j): Qnew[i,j]})
    
    
    
  #  sampler = ExactSolver()
  #  response = sampler.sample_ising(bias,J)    
    
   # numberPrintedEnergies=0
  #  for datum in response.data(['sample', 'energy']): 
  #      if numberPrintedEnergies<2:    
       #     print(datum.sample, datum.energy)
     #   numberPrintedEnergies=numberPrintedEnergies+1
    
    
    solver = neal.SimulatedAnnealingSampler()
    response = solver.sample_ising(bias, J, num_reads=500)
    SimulatedResult=[]
    for datum in response.data(['sample', 'energy', 'num_occurrences']):   
              #  print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
                SimulatedResult.append([datum.sample,  datum.energy,  datum.num_occurrences]) 
   
    # chain = np.max (bias)


    # sampler = EmbeddingComposite(DWaveSampler())
    # response = sampler.sample_ising(bias,J,chain_strength=chain ,num_reads=500, return_embedding=True, anneal_schedule=((0.0,0.0),(40.0,0.5),(140.0,0.5),(180.0,1.0)))
    # dwave.inspector.show(response)
    # Result=[]
    # for datum in response.data(['sample', 'energy', 'num_occurrences','chain_break_fraction']):   
    #         print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
    #         Result.append([datum.sample,  datum.energy,  datum.num_occurrences, datum.chain_break_fraction])


    return [response.info,SimulatedResult]
