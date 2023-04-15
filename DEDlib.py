## module DEDlib
''' DEDlib is a Distributional Exact Diagonalization tooling library for study of Anderson (multi-)impurity model in Graphene Nanoribbons'''

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tqdm import tqdm
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

#class DED:

def Jordan_wigner_transform(j, lattice_length):
    """Jordan_wigner_transform(j, lattice_length). 
Defines the Jordan Wigner transformation for a 1D lattice."""
    operators = sigmaz()
    for _ in range(j-1): operators = tensor(operators, sigmaz())
    if j == 0: operators = sigmam()
    else: operators = tensor(operators, sigmam())
    for _ in range(lattice_length - j - 1): operators = tensor(operators, identity(2))
    return operators

def Lorentzian(omega, Gamma, poles,Ed=-3/2,Sigma=3/2):
    """Lorentzian(omega, Gamma, poles,Ed=-3/2,Sigma=3/2). 
Defines the non-interacting DOS (rho0) and selects random sites based on the number of sites in the 1D lattice model and the calculated distribution."""
    p = np.random.uniform(0, 1, poles)
    return -np.imag(1/(omega-Ed-Sigma+1j*Gamma))/np.pi, np.array([Gamma * math.tan(np.pi * (p[i] - 1 / 2))+Ed+Sigma for i in range(poles)])

def Startrans(poles,select,row,omega, eta):
    """Startrans(poles,select,row,omega, eta). 
Function to transform 1D lattice matrices in order to calculates parameters impengergy, bathenergy and Vkk from random sampling distribution."""
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

def HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma, H = 0):
    """HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma). 
Based on energy parameters calculates the Hamiltonian of a single-impurity system."""
    for i in range(2):
        H += impenergy * (c[i].dag() * c[i])
        for j, bathE in enumerate(bathenergy):
            H += Vkk[j] * (c[i].dag() * c[2 * j + i + 2] + c[2 * j + i + 2].dag() * c[i])+ bathE * (c[2 * j + i + 2].dag() * c[2 * j + i + 2])
    return H,H+U * (c[0].dag() * c[0] * c[1].dag() * c[1])-Sigma * (c[0].dag() * c[0] + c[1].dag() * c[1])

def MBGAIM(omega, H, c, eta,Tk,Boltzmann,evals=[],evecs=[],etaoffset=0.0001):
    """MBGAIM(omega, H, c, eta). 
Calculates the many body Green's function based on the Hamiltonian eigenenergies/-states."""
    if evals==[]: evals, evecs =scipy.linalg.eigh(H.data.toarray())
    if Tk==[0]:
        vecn=np.conj(evecs[:,1:]).T
        exp,exp2=vecn@c[0].data.tocoo()@evecs[:,0],vecn@c[0].dag().data.tocoo()@evecs[:,0]
        return sum([abs(expi)** 2 / (omega + evals[i+1] - evals[0] + 1.j * eta) + 
                        abs(exp2[i])** 2 / (omega + evals[0] - evals[i+1] + 1.j * eta) for i,expi in enumerate(exp)]),1,evecs[:,0]
    else:
        MGdat,eta[int(np.round(len(eta)/2))]=np.zeros((len(Tk),len(omega)),dtype = 'complex_'),etaoffset
        for k,T in enumerate(Tk):
            if Boltzmann[k]!=0:
                eevals=np.exp(-evals/T-scipy.special.logsumexp(-evals/T))
                vecn=np.conj(evecs).T
                exp,exp2=vecn@c[0].data.tocoo()@evecs,vecn@c[0].dag().data.tocoo()@evecs
                MGdat[k,:]=sum([(exp[i][j]*exp2[j][i]/ (omega + evi - evj + 1.j * eta) + 
                            exp[j][i]*exp2[i][j]/ (omega + evj - evi + 1.j * eta))*eevals[i] for i,evi in enumerate(evals) for j,evj in enumerate(evals)])*Boltzmann[k]
        return MGdat.squeeze(),Boltzmann,evecs[:,0]

def AIMsolver(impenergy, bathenergy, Vkk, U, Sigma, omega, eta, c, n, ctype,Tk):
    """AIMsolver(impenergy, bathenergy, Vkk, U, Sigma, omega, eta, c, n, ctype). 
Gives Green's function for the impurity level in the full interacting system (up and down spin)."""
    H0,H= HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma)
    try:
        return Constraint(ctype,H0,H,omega,eta,c,n)
    except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence):
        return (np.zeros(len(omega),dtype = 'complex_'),np.zeros(len(Tk)),np.array([])),False

def Constraint(ctype,H0,H,omega,eta,c,n,Tk):
    """Constraint(ctype,H0,H,omega,eta,c,n). 
Constraint implementation function for DED method with various possible constraints."""
    if ctype=='snb':
        vecs=scipy.linalg.eigh(H0.data.toarray(),eigvals=[0, 0])[1][:,0]
        evals, evecs =scipy.linalg.eigh(H.data.toarray())
        return MBGAIM(omega, H, c, eta,Tk,np.exp(-abs(evals[find_nearest(np.diag(np.conj(evecs)@n.data@evecs.T),np.conj(vecs)@n.data@vecs.T)]-evals[0])/Tk),evals, evecs,1e-24),True
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
    
def find_nearest(array,value):
    for i in (i for i,arrval in enumerate(array) if np.isclose(arrval, value, atol=0.1)): return i

def main(N=200000,poles=4,U=3,Sigma=3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39], ctype='n',Ed='AS',bound=3,Tk=[0]):
    """main(N=1000000,poles=4,U=3,Sigma=3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39], ctype='n',Ed='AS'). 
The main DED function simulating the Anderson impurity model for given parameters."""
    omega,eta,selectpcT,selectpT= np.linspace(-bound,bound,SizeO),etaco[0]*abs(np.linspace(-bound,bound,SizeO))+etaco[1],np.zeros((N,poles),dtype = 'float'),[]
    c=[Jordan_wigner_transform(i, 2*poles) for i in range(2*poles)]
    n,AvgSigmadat,Nfin,nd=sum([c[i].dag()*c[i] for i in range(2*poles)]),np.zeros((len(Tk),SizeO),dtype = 'complex_'),np.zeros(len(Tk),dtype = 'float'),np.zeros(len(Tk),dtype = 'complex_')
    for i in tqdm(range(N)):
        reset = False
        while not reset:
            if Ed == 'AS': select=sorted(Lorentzian(omega, Gamma, poles)[1])
            else: select=sorted(Lorentzian(omega, Gamma, poles,Ed,Sigma)[1])
            NewM,nonG=Startrans(poles,select,0,omega,eta)
            (MBGdat,Boltzmann,Ev0),reset=AIMsolver(NewM[0][0], [NewM[k+1][k+1] for k in range(len(NewM)-1)], 
                                   NewM[0,1:], U,Sigma,omega,eta,c, n, ctype,Tk)
            if np.isnan(1/nonG-1/MBGdat+Sigma).any() or np.array([i >= 1000 for i in np.real(1/nonG-1/MBGdat+Sigma)]).any(): reset=False
            selectpT.append(select)
        selectpcT[i,:]=select
        Nfin+=Boltzmann
        AvgSigmadat+=(1/nonG-1/MBGdat+Sigma)
        nd+=np.conj(Ev0).T@(c[0].dag() * c[0] + c[1].dag() * c[1]).data.tocoo()@Ev0
    if Ed == 'AS': return np.real(nd/Nfin).squeeze(),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]+AvgSigmadat[int(np.round(SizeO/2))]/Nfin[:,None]+1j*Gamma)))/np.pi).squeeze(),Lorentzian(omega,Gamma,poles)[0],omega,selectpT,selectpcT
    else: return np.real(nd/Nfin).squeeze(),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]-Ed+1j*Gamma)))/np.pi).squeeze(),Lorentzian(omega,Gamma,poles,Ed,Sigma)[0],omega,selectpT,selectpcT

def GrapheneAnalyzer(imp,fsyst,colorbnd,filename,omega=np.linspace(-8,8,4001),etaco=[0.02,1e-24],omegastat=100001):
    """GrapheneAnalyzer(imp,fsyst,colorbnd,filename,omega=np.linspace(-8,8,4001),etaco=[0.02,1e-24],omegastat=100001).
Returns data regarding a defined graphene circular structure such as the corresponding Green's function."""
    def plotsize(i): return 0.208 if i == imp else 0.125
    def family_color(i):
        if i == imp: return 'purple'
        elif i<colorbnd: return (31/255,119/255,180/255,255/255)
        else: return (255/255,127/255,14/255,255/255)
    plt.figure(figsize=(10,8))
    plt.rc('legend', fontsize=25)
    plt.rc('font', size=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plot=kwant.plot(fsyst,unit=1.2 ,hop_lw=0.05,site_size=plotsize,site_color=family_color,site_lw=0.02,fig_size=[10,8])
    plot.tight_layout()
    plot.savefig(filename+'NR.svg', format='svg', dpi=3600)
    plt.draw()
    plt.pause(0.5)
    eig,P=scipy.linalg.eigh(fsyst.hamiltonian_submatrix(sparse=False))
    return np.abs(P[imp][:])**2/np.linalg.norm(np.abs(P[imp][:])),[np.sum([(abs(Pv[i])**2)/(omega-eigv+1.j*(etaco[0]*abs(omega)+etaco[1])) 
                                    for i,eigv in enumerate(eig)],axis=0) for _,Pv in enumerate(P)][imp],eig,[np.sum([(abs(Pv[i])**2)
                                    /(np.linspace(min(omega),max(omega),omegastat)-eigv+1.j*(etaco[0]*abs(np.linspace(min(omega),max(omega),omegastat))+etaco[1])) 
                                    for i,eigv in enumerate(eig)],axis=0) for _,Pv in enumerate(P)][imp]

def GrapheneNRzigzagstruct(W=2.5,L=12,x=-11.835680518387328,dy=0.5,Wo=0, Lo=0,t=1):
    lat,sys=kwant.lattice.Polyatomic([[sqrt(3)/2, 0.5], [0, 1]],[[-1/sqrt(12), -0.5], [1/sqrt(12), -0.5]]),kwant.Builder()
    sys[lat.shape(ribbon(W, L), (0, 0))],sys[lat.neighbors(1)]  = 0, -t
    del sys[lat.shape(ribbon(W, dy,x, 0), (x, 0))],sys[lat.shape(ribbon(Wo, Lo,-x, -W), (-x, -W))],sys[lat.shape(ribbon(Wo, Lo,-x, W), (-x, W))]
    return sys.finalized()

def GrapheneNRarmchairstruct(W=3,L=12,y=-2.8867513459481287,Wo=0, Lo=0,t=1):
    lat,sys=kwant.lattice.Polyatomic([[1, 0], [0.5, sqrt(3)/2]],[[0, 1/sqrt(3)], [0, 0]]),kwant.Builder()
    sys[lat.shape(ribbon(W,L), (0, 0))],sys[lat.neighbors(1)] = 0,-t
    del sys[lat.shape(ribbon(Wo, Lo,L,y), (L, y))], sys[lat.shape(ribbon(Wo, Lo,-L,y), (-L, y))]
    return sys.finalized()

def ribbon(W, L,x=0,y=0):
    def shape(pos):
        return (-L <= pos[0]-x <= L and -W <= pos[1]-y <= W)
    return shape

def Graphenecirclestruct(r=1.5, t=1):
    def circle(pos):
        return pos[0]**2 + pos[1]**2 < r**2
    lat,syst=kwant.lattice.honeycomb(norbs=1),kwant.Builder()
    syst[lat.shape(circle, (0, 0))],syst[lat.neighbors()] = 0,-t
    return syst.finalized()

def Graphene_main(psi,SPG,eig,SPrho0,N=200000,poles=4,U=3,Sigma=3/2,SizeO=4001,etaco=[0.02,1e-24], ctype='n',Ed='AS',bound=8,eigsel=False,nd=0):
    """Graphene_main(graphfunc,args,imp,colorbnd,name,N=200000,poles=4,U=3,Sigma=3/2,SizeO=4001,etaco=[0.02,1e-24], ctype='n',Ed='AS',bound=8,eigsel=False). 
The main Graphene nanoribbon DED function simulating the Anderson impurity model on a defined graphene structure for given parameters."""
    omega,eta,AvgSigmadat,selectpcT,selectpT= np.linspace(-bound,bound,SizeO),etaco[0]*abs(np.linspace(-bound,bound,SizeO))+etaco[1],np.zeros(SizeO,dtype = 'complex_'),np.zeros((N,poles),dtype = 'float'),[]
    c,rhoint=[Jordan_wigner_transform(i, 2*poles) for i in range(2*poles)],-np.imag(SPrho0)/np.pi*((max(omega)-min(omega))/len(SPrho0))/sum(-np.imag(SPrho0)/np.pi*((max(omega)-min(omega))/len(SPrho0)))
    n=sum([c[i].dag()*c[i] for i in range(2*poles)])
    for i in tqdm(range(N)):
        reset = False
        while not reset:
            if eigsel: select=sorted(np.random.choice(eig, poles,p=psi,replace=False))
            else: select=sorted(np.random.choice(np.linspace(-bound,bound,len(rhoint)),poles,p=rhoint,replace=False))
            NewM,nonG=Startrans(poles,select,0,omega,eta)
            (MBGdat,Ev0),reset=AIMsolver(NewM[0][0], [NewM[k+1][k+1] for k in range(len(NewM)-1)], 
                                   NewM[0,1:], U,Sigma,omega,eta,c, n, ctype)
            if np.isnan(1/nonG-1/MBGdat+Sigma).any() or any(abs(i) >= 1000 for i in np.real(1/nonG-1/MBGdat+Sigma)) or any(float(i) >= 500 for i in np.abs(1/nonG-1/MBGdat+Sigma)): reset=False
            selectpT.append(select)
        selectpcT[i,:]=select
        AvgSigmadat+=(1/nonG-1/MBGdat+Sigma)/N
        nd+=1/N*np.conj(Ev0).T@(c[0].dag() * c[0] + c[1].dag() * c[1]).data.tocoo()@Ev0
    if Ed == 'AS': return np.real(nd),AvgSigmadat,-np.imag(1/(1/SPG-AvgSigmadat+AvgSigmadat[int(np.round(SizeO/2))]))/np.pi,-np.imag(SPG)/np.pi,omega,selectpT,selectpcT
    else: return np.real(nd),AvgSigmadat,-np.imag(1/(1/SPG-AvgSigmadat-Ed))/np.pi,-np.imag(SPG)/np.pi,omega,selectpT,selectpcT

def PolestoDOS(select,selectnon,ratio=200,bound=3):
    """PolestoDOS(select,selectnon,ratio=200). 
Function with calculated distribution of selected sites based on the results of the DED algorithm."""
    bar=int(len(select)/ratio)
    bomega=np.linspace(-bound,bound,bar)
    DOSp=[((bomega[i] < select) & (select <= bomega[i+1])).sum() for i in range(0,bar-1)]
    return np.linspace(-bound,bound,bar-1),DOSp/(2*bound/(bar-1)*sum(DOSp)),[np.mean(DOSp[j-1:j+1]) 
                                                         for j in range(1,bar-2)],[((bomega[i] < selectnon) & (selectnon <= bomega[i+1])).sum() 
                                                                                   for i in range(0,bar-1)]

def DOSplot(fDOS,Lor,omega,name,labels,log=False):
    """DOSplot(fDOS,Lor,omega,name,labels). 
A plot function to present results from the AIM moddeling for a single results with a comparison to the non-interacting DOS."""
    plt.figure(figsize=(10,8))
    plt.rc('legend', fontsize=17)
    plt.rc('font', size=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    axis_font = {'fontname':'Calibri', 'size':'25'}
    plt.xlim(min(omega), max(omega))
    if not log: plt.gca().set_ylim(bottom=0,top=1.2)
    else: 
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=10)
        plt.gca().set_xticks([-8,-6,-4,-2,0,2,4,6,8], minor=False)
    plt.xlabel("$\\omega$ [-]", **axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom", rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,Lor, '--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,'-b',label=labels)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png', format='png')
    plt.savefig(name+'.svg', format='svg', dpi=3600)
    plt.draw()
    plt.pause(0.5)
    return plt

def DOSmultiplot(omega,omegap,DOST,plotp,labels,name,rho0,log=False):
    """DOSmultiplot(omega,omegap,DOST,plotp,labels,name).
Multi plot function to combine datasets in one graph for comparison including a defined non-interacting DOS."""
    colors=['crimson','darkorange','lime','turquoise','cyan','dodgerblue','darkviolet','deeppink']
    plt.figure(figsize=(10,8))
    plt.rc('legend', fontsize=18)
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    axis_font = {'fontname':'Calibri', 'size':'18'}
    plt.xlim(min(omega), max(omega))
    if not log: plt.gca().set_ylim(bottom=0,top=1.2)
    else: 
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=10)
        plt.gca().set_xticks([-8,-6,-4,-2,0,2,4,6,8], minor=False)
    plt.xlabel("$\\omega$ [-]", **axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom", rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,rho0, '--',color='black',linewidth=4,label='$\\rho_0$')
    for i,p in enumerate(plotp): plt.plot(omegap[i,:p],DOST[i,:p],colors[i],linewidth=2,label=labels[i])
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png', format='png')
    plt.savefig(name+'.svg', format='svg', dpi=3600)
    plt.draw()
    plt.pause(0.5)
    return plt

def textfileW(omega,selectpT,selectpcT,fDOS,name):
    """textfileW(omega,selectpT,selectpcT,fDOS,name).
File writing function for DED results."""
    np.savetxt(name,np.transpose([omega,fDOS]), fmt='%.18g', delimiter='\t', newline='\n')
    np.savetxt(name+'polesC',selectpcT, delimiter='\t', newline='\n')
    np.savetxt(name+'poles',selectpT, delimiter='\t', newline='\n')

def textfileR(name):
    """textfileR(name).
File reader to read DED data writen by textfileW(...)."""
    text_file = open(name, "r")
    lines = text_file.read().split('\n')
    text_file.close()
    return np.array([np.array(l,dtype=object).astype(np.float) for l in [lines[i].split('\t') for i, _ in enumerate(lines[1:])]])