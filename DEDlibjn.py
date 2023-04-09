import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import tqdm
import time
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import kwant
import math
from math import  sqrt
import scipy
from itertools import repeat
from numba import jit

# Defining the Jordan Wigner transformation for a 1D lattice
def jordan_wigner_transform(j, lattice_length):
    operators = sigmaz()
    for _ in range(j-1): operators = tensor(operators, sigmaz())
    if j == 0:
        operators = sigmam()
    else:
        operators = tensor(operators, sigmam())
    for _ in range(lattice_length - j - 1): operators = tensor(operators, identity(2))
    return operators

# Based on energy parameters calculates the Hamiltonian of the impurity system
def HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma):
    H = 0
    for i in range(2):
        H += impenergy * (c[i].dag() * c[i])
        for j, _ in enumerate(bathenergy):
            H += Vkk[j] * (c[i].dag() * c[2 * j + i + 2] + c[2 * j + i + 2].dag() * c[i])+bathenergy[j] * (c[2 * j + i + 2].dag() * c[2 * j + i + 2])
    H0=H.copy()
    H += U * (c[0].dag() * c[0] * c[1].dag() * c[1])-Sigma * (c[0].dag() * c[0] + c[1].dag() * c[1])
    return H0,H

# Calculates the many body Green's function based on the eigenvalues
def MBGAIM(omega, H, c, eta,start_time,attempts,i):
    evals, evecs =scipy.linalg.eigh(H.data.toarray())
    print("--- part 3 %s s attempt %d" % ((time.time() - start_time) ,attempts)+" "+str(i)+"---",end='\r')
    eigm=scipy.sparse.csr_matrix(evecs)
    exp,exp2=np.conj(eigm[:,1:]).T@c[0].data@eigm[:,0],np.conj(eigm[:,1:]).T@c[0].dag().data@eigm[:,0]
    return sum([abs(exp.toarray()[i][0])** 2 / (omega + evals[i+1] - evals[0] + 1.j * eta) + 
                      abs(exp2.toarray()[i][0])** 2 / (omega + evals[0] - evals[i+1] + 1.j * eta) for i,_ in enumerate(exp.toarray())])

# Gives Green’s function for the impurity level in the full interacting system (up and down spin)
def AIMsolver(impenergy, bathenergy, Vkk, U, Sigma, omega, eta, c, n, ctype,start_time,attempts,i):
    H0,H= HamiltonianAIM(c, n, impenergy, bathenergy, Vkk, U, Sigma)
    try:
        return constraint(ctype,H0,H,omega,eta,c,n,start_time,attempts,i)
    except np.linalg.LinAlgError or ValueError:
        return np.zeros(len(omega),dtype = 'complex_'),False

# Constraint implementation
def constraint(ctype,H0,H,omega,eta,c,n,start_time,attempts,i):
    print("--- part 1 %s s attempt %d" % ((time.time() - start_time) ,attempts)+" "+str(i)+"---",end='\r')
    if ctype[0]=='n':
        #_, vecs0 =scipy.sparse.linalg.eigs(np.real(H0.data), k=1, which='SR')
        _, vecs0 =scipy.sparse.linalg.eigh(np.real(H0.data))
        exp2=np.real(np.dot(np.conj(vecs0[:,0]).T,n.data*vecs0[:,0]))
        _, vecs =scipy.sparse.linalg.eigh(np.real(H.data))
        #_, vecs =scipy.sparse.linalg.eigs(np.real(H.data), k=1, which='SR')
        exp=np.dot(np.conj(vecs[:,0]).T,n.data*vecs[:,0])
        print("--- part 2 %s s attempt %d" % ((time.time() - start_time) ,attempts)+" "+str(i)+"---",end='\r')
        if ctype=='n%2' and int(np.round(exp))%2==int(np.round(exp2))%2:
            return MBGAIM(omega, H, c, eta,start_time,attempts,i),True
        elif ctype=='n' and np.round(exp)==np.round(exp2):
            return MBGAIM(omega, H, c, eta,start_time,attempts,i),True
        else:
            return np.zeros(len(omega),dtype = 'complex_'),False
    else:
        return MBGAIM(omega, H, c, eta,start_time,attempts,i),True

#Functions to transfor matrices and calculates parameters impengergy,bathenergy and Vkk from random sampling distribution
def startrans(poles,select,row,omega, eta):
    di=np.full((poles-1, poles), np.zeros(poles))
    for i in range(poles-1):
        for j in (j for j in range(poles-1) if j>=i): di[i][j+1]=-1/sqrt((poles-i-1)*(poles-i))
        di[i][i]=sqrt(poles-i-1)/sqrt(poles-i)
    Pbath,Dbath=np.insert(di, row,1/sqrt(poles),axis=0),np.zeros((poles,poles))
    for i, _ in enumerate(select): Dbath[i][i]=select[i]
    ham_mat=np.dot(Pbath,np.dot(Dbath,Pbath.T))
    pbar=np.insert(np.insert(np.linalg.eig(np.delete(np.delete(ham_mat,row,axis=0),row,axis=1))[1], row,0,axis=0),row,0,axis=1)
    pbar[row][row]=1
    return np.dot(pbar.T,np.dot(ham_mat,pbar)),sum([1 / len(select) / (omega - select[i] + 1.j * eta) for i, _ in enumerate(select)])

# Define rho0 distribution
def Lorentzian(omega, Gamma, poles):
    p = np.random.uniform(0, 1, poles)
    return Gamma / np.pi / (omega ** 2 + Gamma ** 2), np.array([Gamma * math.tan(np.pi * (p[i] - 1 / 2)) for i in range(poles)])

def main(N=1000000,poles=4,U=3,Sigma=3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39], ctype='n'):
    omega,eta,AvgSigmadat= np.linspace(-3,3,SizeO),etaco[0]*abs(np.linspace(-3,3,SizeO))+etaco[1],np.zeros(SizeO,dtype = 'complex_')
    c=[jordan_wigner_transform(i, 2*poles) for i in range(2*poles)]
    selectpT,selectpcT,n=[],np.zeros((N,poles),dtype = 'float'),sum([c[i].dag()*c[i] for i in range(2*poles)])
    for i in tqdm(range(N)):
        start_time,attempts,reset = time.time(),0,False
        while not reset:
            select=sorted(Lorentzian(omega, Gamma, poles)[1])
            NewM,nonG=startrans(poles,select,0,omega,eta)
            MBGdat,reset=AIMsolver(NewM[0][0], [NewM[k+1][k+1] for k in range(len(NewM)-1)], 
                                   NewM[0,1:], U,Sigma,omega,eta,c, n, ctype,start_time,attempts,i)
            print("--- part 4 %s s attempt %d" % ((time.time() - start_time) ,attempts)+" "+str(i)+"---",end='\r')
            if np.isnan(1/nonG-1/MBGdat+Sigma).any() or any(i >= 1000 for i in np.real(1/nonG-1/MBGdat+Sigma)): reset=False
            attempts+=1
            selectpT.append(select)
        selectpcT[i,:]=select
        AvgSigmadat+=(1/nonG-1/MBGdat+Sigma)/N
    fDOS=-np.imag(np.nan_to_num(1/(omega-AvgSigmadat+AvgSigmadat[500]+1j*Gamma)))/np.pi
    return fDOS, Lorentzian(omega,Gamma,poles)[0], omega,selectpT,selectpcT

def PolestoDOS(select,selectnon,ratio=200):
    bar=int(len(select)/ratio)
    bomega=np.linspace(-3,3,bar)
    DOSp=[((bomega[i] < select) & (select <= bomega[i+1])).sum() for i in range(0,bar-1)]
    return np.linspace(-3,3,bar-1),DOSp/(6/(bar-1)*sum(DOSp)),[np.mean(DOSp[j-1:j+1]) 
                                                         for j in range(1,bar-2)],[((bomega[i] < selectnon) & (selectnon <= bomega[i+1])).sum() 
                                                                                   for i in range(0,bar-1)]

def DOSplot(fDOS,Lor,omega,name,labels):
    font = {'size'   : 25}
    plt.rc('legend', fontsize=17)
    plt.rc('font', **font)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    axis_font = {'fontname':'Calibri', 'size':'25'}
    plt.figure(figsize=(10,8))
    plt.xlim(min(omega), max(omega))
    plt.gca().set_ylim(bottom=0,top=1.2)
    plt.xlabel("$\\omega$ [-]", **axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom", rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,Lor, '--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,'-b',label=labels)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png')
    plt.savefig(name+'.svg', format='svg', dpi=3600)
    plt.draw()
    return plt

def DOSmultiplot(omega,omegap,DOST,plotp,labels,name):
    font = {'size'   : 18}
    colors=['crimson','darkorange','lime','turquoise','cyan','dodgerblue','darkviolet','deeppink']
    plt.rc('legend', fontsize=18)
    plt.rc('font', **font)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    axis_font = {'fontname':'Calibri', 'size':'18'}
    plt.figure(figsize=(10,8))
    plt.xlim(min(omega), max(omega))
    plt.gca().set_ylim(bottom=0,top=1.2)
    plt.xlabel("$\\omega$ [-]", **axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom", rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,Lorentzian(omega,0.3,4)[0], '--',color='black',linewidth=4,label='$\\rho_0$')
    for i,p in enumerate(plotp):
        plt.plot(omegap[i,:p],DOST[i,:p],colors[i],linewidth=2,label=labels[i])
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png')
    plt.savefig(name+'.svg', format='svg', dpi=3600)
    plt.draw()
    return plt

def textfileW(omega,selectpT,selectpcT,fDOS,name):
    np.savetxt(name,np.transpose([omega,fDOS]), fmt='%.18g', delimiter='\t', newline='\n')
    np.savetxt(name+'polesC',selectpcT, delimiter='\t', newline='\n')
    np.savetxt(name+'poles',selectpT, delimiter='\t', newline='\n')

def textfileR(name):
    text_file = open(name, "r")
    lines = text_file.read().split('\n')
    text_file.close()
    return np.array([np.array(l,dtype=object).astype(np.float) for l in [lines[i].split('\t') for i, _ in enumerate(lines[1:])]])