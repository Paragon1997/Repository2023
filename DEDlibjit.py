## module DEDlib
''' DEDlib is a Distributional Exact Diagonalization tooling library for study of Anderson (multi-)impurity model in Graphene Nanoribbons'''

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tqdm.auto import trange
from time import time
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import kwant
from numpy import sqrt
import scipy
from itertools import repeat
from numba import njit

def Jordan_wigner_transform(j, lattice_length):
    """Jordan_wigner_transform(j, lattice_length). 
Defines the Jordan Wigner transformation for a 1D lattice."""
    operators = sigmaz()
    for _ in range(j-1): operators = tensor(operators, sigmaz())
    if j == 0: operators = sigmam()
    else: operators = tensor(operators, sigmam())
    for _ in range(lattice_length - j - 1): operators = tensor(operators, identity(2))
    return operators

@njit
def Lorentzian(omega, Gamma, poles,Ed=-3/2,Sigma=3/2):
    """Lorentzian(omega, Gamma, poles,Ed=-3/2,Sigma=3/2). 
Defines the non-interacting DOS (rho0) and selects random sites based on the number of sites in the 1D lattice model and the calculated distribution."""
    p = np.random.uniform(0, 1, poles)
    return -np.imag(1/(omega-Ed-Sigma+1j*Gamma))/np.pi, np.array([Gamma * np.tan(np.pi * (p[i] - 1 / 2))+Ed+Sigma for i in range(poles)])

@njit
def Startrans(poles,select,row,omega, eta):
    """Startrans(poles,select,row,omega, eta). 
Function to transform 1D lattice matrices in order to calculates parameters impengergy, bathenergy and Vkk from random sampling distribution."""
    Pbath,pbar,G=np.zeros((poles, poles)),np.zeros((poles, poles)),np.zeros(omega.shape,dtype = 'complex_')
    for i in range(poles-1):
        for j in range(poles-1):
            if j>=i:
                Pbath[i+1][j+1]=-1/sqrt((poles-i-1)*(poles-i))
        Pbath[i+1][i]=sqrt(poles-i-1)/sqrt(poles-i)
    Pbath[row,:]=1/sqrt(poles)
    Dbath=np.zeros((poles,poles))
    for i, _ in enumerate(select):
        Dbath[i][i]=select[i]
    pbar[1:,1:]=np.linalg.eig(np.dot(Pbath,np.dot(Dbath,Pbath.T))[1:,1:])[1]
    pbar[row][row]=1
    for i, _ in enumerate(select): G+=1 / len(select) / (omega - select[i] + 1.j * eta)
    return np.dot(pbar.T,np.dot(np.dot(Pbath,np.dot(Dbath,Pbath.T)),pbar)),G,select

def HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma, H = 0):
    """HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma). 
Based on energy parameters calculates the Hamiltonian of a single-impurity system."""
    for i in range(2):
        H += impenergy * (c[i].dag() * c[i])
        for j, bathE in enumerate(bathenergy):
            H += Vkk[j] * (c[i].dag() * c[2 * j + i + 2] + c[2 * j + i + 2].dag() * c[i])+ bathE * (c[2 * j + i + 2].dag() * c[2 * j + i + 2])
    return H,H+U * (c[0].dag() * c[0] * c[1].dag() * c[1])-Sigma * (c[0].dag() * c[0] + c[1].dag() * c[1])

def AIMsolver(impenergy, bathenergy, Vkk, U, Sigma, omega, eta, c, n, ctype,Tk):
    """AIMsolver(impenergy, bathenergy, Vkk, U, Sigma, omega, eta, c, n, ctype). 
Gives Green's function for the impurity level in the full interacting system (up and down spin)."""
    H0,H= HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma)
    try:
        return Constraint(ctype,H0,H,omega,eta,c,n,Tk)
    except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence):
        return (np.zeros(len(omega),dtype = 'complex_'),np.zeros(len(Tk)),np.array([])),False
    
def Constraint(ctype,H0,H,omega,eta,c,n,Tk):
    """Constraint(ctype,H0,H,omega,eta,c,n). 
Constraint implementation function for DED method with various possible constraints."""
    if ctype=='snb':
        vecs=scipy.linalg.eigh(H0.data.toarray(),eigvals=[0, 0])[1][:,0]
        evals, evecs =scipy.linalg.eigh(H.data.toarray())
        return MBGAIM(omega, H, c, eta,Tk,np.exp(-abs(evals[find_nearest(np.diag(np.conj(evecs).T@n.data@evecs),np.conj(vecs)@n.data@vecs.T)]-evals[0])/Tk),evals, evecs,0.00001),True
    elif ctype[0]=='n':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.sparse.linalg.eigsh(np.real(H0.data), k=1, which='SA')[1][:,0],
                                                scipy.sparse.linalg.eigsh(np.real(H.data), k=1, which='SA')[1][:,0])))
        exp=np.conj(vecs)@n.data@vecs.T
        if ctype=='n%2' and int(np.round(exp[0,0]))%2==int(np.round(exp[1,1]))%2:
            return MBGAIM(omega, H, c, eta,Tk,np.ones(len(Tk))),True
        elif ctype=='n' and np.round(exp[0,0])==np.round(exp[1,1]):
            return MBGAIM(omega, H, c, eta,Tk,np.ones(len(Tk))),True
        else:
            return (np.zeros(len(omega),dtype = 'complex_'),np.zeros(len(Tk)),np.array([])),False
    elif ctype[0]=='d':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.linalg.eigh(H.data.toarray(),eigvals=[0, 0])[1][:,0],
                                                scipy.linalg.eigh(H0.data.toarray(),eigvals=[0, 0])[1][:,0])))
        exp=np.conj(vecs)@n.data@vecs.T
        if ctype=='dn' and np.round(exp[0,0])==np.round(exp[1,1]):
            return MBGAIM(omega, H, c, eta,Tk,np.ones(len(Tk))),True
        else:
            return (np.zeros(len(omega),dtype = 'complex_'),np.zeros(len(Tk)),np.array([])),False
    else:
        return MBGAIM(omega, H, c, eta,Tk,np.ones(len(Tk))),True