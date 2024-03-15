## module DEDlib
''' DEDlib is a Distributional Exact Diagonalization tooling library for study of Anderson (multi-)impurity model in Graphene Nanoribbons'''

import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
from tqdm.auto import trange
from qutip import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import kwant
import scipy
from numba import njit
from numpy._typing import NDArray
from kwant.builder import FiniteSystem
from matplotlib.figure import Figure

def Jordan_wigner_transform(j:int,lattice_length:int)->Qobj:
    """``Jordan_wigner_transform(j,lattice_length)``.\n 
Defines the Jordan Wigner transformation for a 1D lattice."""
    operators=sigmaz()
    for _ in range(j-1):operators=tensor(operators,sigmaz())
    if j == 0:operators=sigmam()
    else:operators=tensor(operators,sigmam())
    for _ in range(lattice_length - j - 1):operators=tensor(operators,identity(2))
    return operators

@njit
def Lorentzian(omega:NDArray[np.float64],Gamma:float,poles:int,Ed:float=-3/2,Sigma:float=3/2)->tuple[NDArray[np.float64],NDArray[np.float64]]:
    """``Lorentzian(omega,Gamma,poles,Ed=-3/2,Sigma=3/2)``.\n 
Defines the non-interacting DOS (rho0) and selects random sites based on the number of sites in the 1D lattice model and the calculated distribution."""
    return -np.imag(1/(omega-Ed-Sigma+1j*Gamma))/np.pi,np.array([Gamma*np.tan(np.pi*(pi-1/2))+Ed+Sigma for pi in np.random.uniform(0,1,poles)])

@njit
def Startrans(poles:int,select:NDArray[np.float64],omega:NDArray[np.float64],eta:NDArray[np.float64],row:int=0)->tuple[NDArray[np.float64],NDArray[np.complex128],NDArray[np.float64]]:
    """``Startrans(poles,select,omega,eta,row=0)``.\n 
Function to transform 1D lattice matrices in order to calculates parameters impengergy, bathenergy and Vkk from random sampling distribution."""
    Pbath,Dbath,pbar,G=np.zeros((poles,poles)),np.zeros((poles,poles)),np.zeros((poles,poles)),np.zeros(omega.shape,dtype='complex_')
    for i in range(poles-1):
        for j in range(poles-1):
            if j>=i: Pbath[i+1][j+1]=-1/np.sqrt((poles-i-1)*(poles-i))
        Pbath[i+1][i]=np.sqrt(poles-i-1)/np.sqrt(poles-i)
    Pbath[row,:]=1/np.sqrt(poles)
    for i, _ in enumerate(select):Dbath[i][i]=select[i]
    pbar[1:,1:]=np.linalg.eig((Pbath@Dbath@Pbath.T)[1:,1:])[1]
    pbar[row][row]=1
    for i, _ in enumerate(select):G+=1/len(select)/(omega-select[i]+1.j*eta)
    return pbar.T@Pbath@Dbath@Pbath.T@pbar,G,select

def Operators(c:list[Qobj],Nimpurities:int,poles:int)->tuple[tuple[list[Qobj],Qobj],list[Qobj]]:
    """``Operators(c,Nimpurities,poles)``.\n
Calculates various operators to construct the required (non-)interacting Hamiltonians."""
    posimp=[int(2*poles/Nimpurities*i) for i in range(Nimpurities)]
    impn=[sum([c[posimp[k]+i].dag()*c[posimp[k]+i] for i in range(2)]) for k in range(Nimpurities)]
    bathn=[[sum([c[2*j+i+2+posimp[k]].dag()*c[2*j+i+2+posimp[k]] for i in range(2)]) for j in range(int(poles/Nimpurities)-1)] for k in range(Nimpurities)]
    crossn=[[sum([c[posimp[k]+i].dag()*c[2*j+i+2+posimp[k]]+c[2*j+i+2+posimp[k]].dag()*c[posimp[k]+i] for i in range(2)]) for j in range(int(poles/Nimpurities)-1)] for k in range(Nimpurities)]
    Un=sum([c[posimp[k]].dag()*c[posimp[k]]*c[posimp[k]+1].dag()*c[posimp[k]+1] for k in range(Nimpurities)])
    Sigman=sum([c[posimp[k]].dag()*c[posimp[k]]+c[posimp[k]+1].dag()*c[posimp[k]+1] for k in range(Nimpurities)])
    U2n=sum([c[posimp[k]+i].dag()*c[posimp[k]+i]*c[posimp[l]+j].dag()*c[posimp[l]+j] for i in range(2) for j in range(2) for l in range(Nimpurities) for k in range(Nimpurities) if k !=l])
    Jn=sum([1/2*(c[posimp[k]].dag()*c[posimp[k]+1]*c[posimp[l]+1].dag()*c[posimp[l]]+c[posimp[k]+1].dag()*c[posimp[k]]*c[posimp[l]].dag()*c[posimp[l]+1])
                    +1/4*(c[posimp[k]].dag()*c[posimp[k]]-c[posimp[k]+1].dag()*c[posimp[k]+1])*(c[posimp[l]].dag()*c[posimp[l]]-c[posimp[l]+1].dag()*c[posimp[l]+1]) for l in range(Nimpurities) for k in range(Nimpurities) if k!=l])
    n=[sum([c[j+posimp[i]].dag()*c[j+posimp[i]] for j in range(2*int(poles/Nimpurities))]) for i in range(Nimpurities)]
    return (impn,bathn,crossn,Un,Sigman,U2n,Jn),n

def HamiltonianAIM(impenergy:NDArray[np.float64],bathenergy:NDArray[np.float64],Vkk:NDArray[np.float64],U:float,Sigma:float,U2:float,J:float,Hn:tuple[list[Qobj],Qobj],H0:Qobj=0)->tuple[Qobj,Qobj]:
    """``HamiltonianAIM(impenergy,bathenergy,Vkk,U,Sigma,U2,J,Hn,H0=0)``.\n 
Based on energy parameters calculates the Hamiltonian of a single-impurity system."""
    for k in range(len(impenergy)):
        H0+=impenergy[k]*Hn[0][k]
        for j in range(len(bathenergy[k])): H0+=bathenergy[k][j]*Hn[1][k][j]+Vkk[k][j]*Hn[2][k][j]
    return H0,H0+U*Hn[3]-Sigma*Hn[4]+(U2/2-J/4)*Hn[5]-J*Hn[6]

@njit
def MBGT0(omega:NDArray[np.float64],eta:NDArray[np.float64],evals:NDArray[np.float64],exp:NDArray[np.complex128],exp2:NDArray[np.complex128])->NDArray[np.complex128]:
    """``MBGT0(omega,eta,evals,exp,exp2)``.\n
Determines the many body Green's function for the T=0 case given the eigen-values and -vectors."""
    G=np.zeros(len(omega),dtype='complex_')
    for i,expi in enumerate(exp): G+=abs(expi)**2/(omega+evals[i+1]-evals[0]+1.j*eta)+abs(exp2[i])**2/(omega+evals[0]-evals[i+1]+1.j*eta)
    return G

@njit
def MBGTnonzero(omega:NDArray[np.float64],eta:NDArray[np.float64],evals:NDArray[np.float64],exp:NDArray[np.complex128],exp2:NDArray[np.complex128],eevals:NDArray[np.float64])->NDArray[np.complex128]:
    """``MBGTnonzero(omega,eta,evals,exp,exp2,eevals)``.\n
Determines the many body Green's function for the T>0 case given the eigen-values and -vectors."""
    G=np.zeros(len(omega),dtype='complex_')
    for i,evi in enumerate(evals):
        for j,evj in enumerate(evals):
            G+=(exp[i][j]*exp2[j][i]/(omega+evi-evj+1.j*eta)+exp[j][i]*exp2[i][j]/(omega+evj-evi+1.j*eta))*eevals[i]
    return G

def MBGAIM(omega:NDArray[np.float64],H:Qobj,c:list[Qobj],eta:NDArray[np.float64],Tk:list[float],Boltzmann:NDArray[np.float64],poleDOS:bool,evals:NDArray[np.float64]=[],evecs:NDArray[np.complex128]=[],etaoffset:float=1e-4,posoffset:NDArray[np.int32]=np.zeros(1,dtype='int'))->tuple[NDArray[np.float64],NDArray[np.complex128]]:
    """``MBGAIM(omega,H,c,eta,Tk,Boltzmann,poleDOS,evals=[],evecs=[],etaoffset=1e-4,posoffset=np.zeros(1,dtype='int'))``.\n 
Calculates the many body Green's function based on the Hamiltonian eigenenergies/-states for given temperatures."""
    if poleDOS: return np.zeros(len(omega),dtype='complex_'),Boltzmann,np.array([])
    else:
        if ~np.any(evals):evals,evecs=scipy.linalg.eigh(H.data.toarray())
        if Tk==[0]:
            vecn=np.conj(evecs[:,1:]).T
            exp,exp2=vecn@c[0].data.tocoo()@evecs[:,0],vecn@c[0].dag().data.tocoo()@evecs[:,0]
            return MBGT0(omega,eta,evals,exp,exp2),Boltzmann,evecs[:,0]
        else:
            MGdat,eta[int(np.round(len(eta)/2))+posoffset]=np.ones((len(Tk),len(omega)),dtype='complex_'),etaoffset
            for k,T in enumerate(Tk):
                if Boltzmann[k]!=0:
                    eevals,vecn=np.exp(-evals/T-scipy.special.logsumexp(-evals/T)),np.conj(evecs).T
                    exp,exp2=vecn@c[0].data.tocoo()@evecs,vecn@c[0].dag().data.tocoo()@evecs
                    MGdat[k,:]=MBGTnonzero(omega,eta,evals,exp,exp2,eevals)
            return MGdat.squeeze(),Boltzmann,evecs[:,0]

def Constraint(ctype:str,H0:Qobj,H:Qobj,omega:NDArray[np.float64],eta:NDArray[np.float64],c:list[Qobj],n:list[Qobj],Tk:list[float],Nfin:NDArray[np.float64],poleDOS:bool)->tuple[tuple[NDArray[np.float64],NDArray[np.complex128]],bool]:
    """``Constraint(ctype,H0,H,omega,eta,c,n,Tk,Nfin,poleDOS)``.\n 
Constraint implementation function for DED method with various possible constraints."""
    if ctype[0]=='m':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.sparse.linalg.eigsh(np.real(H0.data),k=1,which='SA')[1][:,0],
                                        scipy.sparse.linalg.eigsh(np.real(H.data),k=1,which='SA')[1][:,0])))
        exp=np.conj(vecs)@n[0].data@vecs.T
        if ctype=='mosn' and np.round(exp[0,0])==np.round(exp[1,1]):
            return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk)),poleDOS),True
        else:
            return (np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
    elif ctype[0]=='s':
        vecs=scipy.linalg.eigh(H0.data.toarray(),eigvals=[0,0])[1][:,0]
        evals,evecs=scipy.linalg.eigh(H.data.toarray())
        if ctype=='ssn':
            Boltzmann=np.exp(-abs(evals[find_nearest(np.diag(np.conj(evecs).T@n[0].data@evecs),np.conj(vecs)@n[0].data@vecs.T)]-evals[0])/Tk)*Nfin.astype('int')
            return MBGAIM(omega,H,c,eta,Tk,Boltzmann,poleDOS,evals,evecs,4e-4,np.array([-2,-1,0,1,2])),True
        else:
            return MBGAIM(omega,H,c,eta,Tk,np.exp(-abs(evals[find_nearest(np.diag(np.conj(evecs).T@n[0].data@evecs),np.conj(vecs)@n[0].data@vecs.T)]-evals[0])/Tk),poleDOS,evals,evecs,4e-4,np.array([-2,-1,0,1,2])),True
    elif ctype[0]=='n':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.sparse.linalg.eigsh(np.real(H0.data),k=1,which='SA')[1][:,0],
                                                scipy.sparse.linalg.eigsh(np.real(H.data),k=1,which='SA')[1][:,0])))
        exp=[np.conj(vecs)@ni.data@vecs.T for ni in n]
        if ctype=='n%2' and all([int(np.round(expi[0,0]))%2==int(np.round(expi[1,1]))%2 for expi in exp]):
            return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk)),poleDOS),True
        elif ctype=='n' and all([np.round(expi[0,0])==np.round(expi[1,1]) for expi in exp]):
            return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk)),poleDOS),True
        else:
            return (np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
    elif ctype[0]=='d':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.linalg.eigh(H.data.toarray(),eigvals=[0,0])[1][:,0],
                                                scipy.linalg.eigh(H0.data.toarray(),eigvals=[0,0])[1][:,0])))
        exp=[np.conj(vecs)@ni.data@vecs.T for ni in n]
        if ctype=='dn' and all([np.round(expi[0,0])==np.round(expi[1,1]) for expi in exp]):
            return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk)),poleDOS),True
        else:
            return (np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
    else:
        return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk)),poleDOS),True
    
def find_nearest(array:NDArray[np.number],value:np.number)->int:
    """``find_nearest(array,value)``.\n
For an array finds the index of the number closest to the given value."""
    for i in (i for i,arrval in enumerate(array) if np.isclose(arrval,value,atol=0.1)): return i

def main(N:int=200000,poles:int=4,U:float=3,Sigma:float=3/2,Ed:float=-3/2,Gamma:float=0.3,SizeO:int=1001,etaco:list[float]=[0.02,1e-39],ctype:str='n',Edcalc:str='',bound:float=3,Tk:list[float]=[0],Nimpurities:int=1,U2:float=0,J:float=0,posb:int=1,log:bool=False,base:float=1.5,poleDOS:bool=False)->tuple[tuple[NDArray[np.float64],NDArray[np.complex128]],NDArray[np.complex128],NDArray[np.float64],list[NDArray[np.float64]],float]:
    """``main(N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5,poleDOS=False)``.\n 
The main DED function simulating the Anderson impurity model for given parameters."""
    if log:omega,selectpcT,selectpT,Npoles=np.concatenate((-np.logspace(np.log(bound)/np.log(base),np.log(1e-5)/np.log(base),int(np.round(SizeO/2)),base=base),np.logspace(np.log(1e-5)/np.log(base),np.log(bound)/np.log(base),int(np.round(SizeO/2)),base=base))),[],[],int(poles/Nimpurities)
    else:omega,selectpcT,selectpT,Npoles=np.linspace(-bound,bound,SizeO),[],[],int(poles/Nimpurities)
    c,pbar,eta=[Jordan_wigner_transform(i,2*poles) for i in range(2*poles)],trange(N,position=posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),etaco[0]*abs(omega)+etaco[1]
    (Hn,n),AvgSigmadat,Nfin,nd=Operators(c,Nimpurities,poles),np.zeros((len(Tk),SizeO),dtype='complex_'),np.zeros(len(Tk),dtype='float'),np.zeros(len(Tk),dtype='complex_')
    while pbar.n<N:
        reset=False
        while not reset:
            NewM,nonG,select=Startrans(Npoles,np.sort(Lorentzian(omega,Gamma,Npoles,Ed,Sigma)[1]),omega,eta)
            H0,H=HamiltonianAIM(np.repeat(NewM[0][0],Nimpurities),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(Nimpurities,1)),np.tile(NewM[0,1:],(Nimpurities,1)),U,Sigma,U2,J,Hn)
            try:(MBGdat,Boltzmann,Ev0),reset=Constraint(ctype,H0,H,omega,eta,c,n,Tk,np.array([ar<N for ar in Nfin]),poleDOS)
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence):(MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
            if np.isnan(1/nonG-1/MBGdat+Sigma).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+Sigma)]).any():reset=False
            selectpT.append(select)
        Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+Sigma)*Boltzmann[:,None],nd+np.conj(Ev0).T@sum(Hn[0]).data.tocoo()@Ev0*Boltzmann
        selectpcT.append(select)
        if ctype=='sn':pbar.n+=1
        else:pbar.n=int(min(Nfin))
        pbar.refresh()
    pbar.close()
    if Edcalc=='AS': return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]+(AvgSigmadat[:,int(np.round(SizeO/2))]/Nfin)[:,None]+1j*Gamma)))/np.pi).squeeze(),Lorentzian(omega,Gamma,poles)[0],omega,selectpT,selectpcT,pbar.format_dict["elapsed"]
    else: return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]-Ed+1j*Gamma)))/np.pi).squeeze(),Lorentzian(omega,Gamma,poles,Ed,Sigma)[0],omega,selectpT,selectpcT,pbar.format_dict["elapsed"]

def ConstraintS(ctype:str,H0:Qobj,H:Qobj,n:list[Qobj],Tk:NDArray[np.float64],Nfin:NDArray[np.float64]=0)->tuple[NDArray[np.float64]]:
    """``ConstraintS(ctype,H0,H,n,Tk,Nfin=0)``.\n
Constraint implementation function for Entropy DED calculation method with various possible constraints."""
    if ctype[0]=='s':
        vecs=scipy.linalg.eigh(H0.data.toarray(),eigvals=[0,0])[1][:,0]
        evals,evecs=scipy.linalg.eigh(H.data.toarray())
        if ctype=='ssn':
            Boltzmann=np.exp(-abs(evals[find_nearest(np.diag(np.conj(evecs).T@n.data@evecs),np.conj(vecs)@n.data@vecs.T)]-evals[0])/Tk)*Nfin.astype('int')
            return Boltzmann,evals
        else: return np.exp(-abs(evals[find_nearest(np.diag(np.conj(evecs).T@n.data@evecs),np.conj(vecs)@n.data@vecs.T)]-evals[0])/Tk),evals
    elif ctype[0]=='n':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.sparse.linalg.eigsh(np.real(H0.data),k=1,which='SA')[1][:,0],
                                                scipy.sparse.linalg.eigsh(np.real(H.data),k=1,which='SA')[1][:,0])))
        exp=np.conj(vecs)@n.data@vecs.T
        if ctype=='n%2' and int(np.round(exp[0,0]))%2==int(np.round(exp[1,1]))%2: return np.ones(len(Tk)),[]
        elif ctype=='n' and np.round(exp[0,0])==np.round(exp[1,1]): return np.ones(len(Tk)),[]
        else: return np.zeros(len(Tk)),[]
    elif ctype[0]=='d':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.linalg.eigh(H.data.toarray(),eigvals=[0,0])[1][:,0],
                                                scipy.linalg.eigh(H0.data.toarray(),eigvals=[0,0])[1][:,0])))
        exp=np.conj(vecs)@n.data@vecs.T
        if ctype=='dn' and np.round(exp[0,0])==np.round(exp[1,1]): return np.ones(len(Tk)),[]
        else: return np.zeros(len(Tk)),[]
    else: return np.ones(len(Tk)),[]

@njit
def SAIM(evals:NDArray[np.float64],Z_tot:NDArray[np.float64],Tk:NDArray[np.float64],kb:float,E_k:NDArray[np.float64],constr:NDArray[np.float64],S_t:NDArray[np.float64],S_b:NDArray[np.float64],S_imp:NDArray[np.float64],Nfin:NDArray[np.float64])->tuple[NDArray[np.float64]]:
    """``SAIM(evals,Z_tot,Tk,kb,E_k,constr,S_t,S_b,S_imp,Nfin)``.\n
Calculates the impurity entropy based on AIM derivation of entropy."""
    S_tot,S_bath=kb*(Z_tot+evals@np.exp(np.outer(-evals,1/Tk)-Z_tot)/Tk),np.zeros(len(Tk))
    for ek in E_k:S_bath+=2*kb*(np.logaddexp(np.zeros(len(Tk)),-ek/Tk)+ek/np.exp(np.logaddexp(np.zeros(len(Tk)),ek/Tk))/Tk)
    return S_t+S_tot*constr,S_b+S_bath*constr,S_imp+(S_tot-S_bath)*constr,Nfin+constr

def Entropyimp_main(N:int=200000,poles:int=4,U:float=3,Sigma:float=3/2,Ed:float=-3/2,Gamma:float=0.3,SizeO:int=1001,etaco:list[float]=[0.02,1e-39],ctype:str='n',bound:float=3,Tk:NDArray[np.float64]=np.logspace(-6,2,801,base=10),kb:float=1,posb:int=1)->tuple[NDArray[np.float64],tuple[int,float]]:
    """``Entropyimp_main(N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',bound=3,Tk=np.logspace(-6,2,801,base=10),kb=1,posb=1)``.\n
The main impurity entropy DED function simulating the Anderson impurity model for given parameters."""
    omega,eta,selectpcT,selectpT,S_imp,S_t,S_b,c=np.linspace(-bound,bound,SizeO),etaco[0]*abs(np.linspace(-bound,bound,SizeO))+etaco[1],[],[],np.zeros(len(Tk),dtype=np.float64),np.zeros(len(Tk),dtype=np.float64),np.zeros(len(Tk),dtype=np.float64),[Jordan_wigner_transform(i, 2*poles) for i in range(2*poles)]
    (Hn,n),Nfin,pbar=Operators(c,1,poles),np.zeros(len(Tk),dtype='float'),trange(N,position=posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    while pbar.n<N:
        constr=np.zeros(len(Tk),dtype='float')
        while np.array([con==0 for con in constr]).all():
            NewM,_,select=Startrans(poles,np.sort(Lorentzian(omega,Gamma,poles,Ed,Sigma)[1]),omega,eta)
            E_k=np.array([NewM[k+1][k+1] for k in range(len(NewM)-1)])
            H0,H=HamiltonianAIM([NewM[0][0]],[E_k],[NewM[0,1:]],U,Sigma,0,0,Hn)
            constr,evals=ConstraintS(ctype,H0,H,n[0],Tk)
            selectpT.append(select)
        selectpcT.append(select)
        if ~np.any(evals):evals=scipy.linalg.eigvalsh(H.data.toarray())
        Z_tot=scipy.special.logsumexp(np.outer(-evals,1/Tk),axis=0)
        if (Z_tot>2e+08).any():continue
        else:S_t,S_b,S_imp,Nfin=SAIM(evals,Z_tot,Tk,kb,E_k,constr,S_t,S_b,S_imp,Nfin)
        if ctype=='sn':pbar.n+=1
        else:pbar.n=int(min(Nfin))
        pbar.refresh()
    pbar.close()
    return np.abs(S_imp/Nfin).squeeze(),np.real(S_t/Nfin).squeeze(),np.real(S_b/Nfin).squeeze(),Nfin.squeeze(),Tk,(pbar.format_dict["n"],pbar.format_dict["elapsed"])

def GrapheneAnalyzer(imp:int,fsyst:FiniteSystem,colorbnd:int,filename:str,SizeO:int=4001,bound:float=8,etaco:list[float]=[0.02,1e-24],omegastat:int=100001,log:bool=False,base:float=1.5,save:bool=True)->tuple[NDArray[np.float64],NDArray[np.complex128]]:
    """``GrapheneAnalyzer(imp,fsyst,colorbnd,filename,SizeO=4001,bound=8,etaco=[0.02,1e-24],omegastat=100001,log=False,base=1.5,save=True)``.\n
Returns data regarding a defined graphene structure fsyst such as the corresponding Green's function."""
    if log:omega=np.concatenate((-np.logspace(np.log(bound)/np.log(base),np.log(1e-5)/np.log(base),int(np.round(SizeO/2)),base=1.5),np.logspace(np.log(1e-5)/np.log(base),np.log(bound)/np.log(base),int(np.round(SizeO/2)),base=base)))
    else:omega=np.linspace(-bound,bound,SizeO)
    def plotsize(i):return 0.208 if i == imp else 0.125
    def family_color(i):
        if i == imp:return 'purple'
        elif i<colorbnd:return (31/255,119/255,180/255,255/255)
        else: return (255/255,127/255,14/255,255/255)
    plt.ion()
    plt.rc('legend',fontsize=25)
    plt.rc('font',size=25)
    plt.rc('xtick',labelsize=25)
    plt.rc('ytick',labelsize=25)
    plot=kwant.plot(fsyst,unit=1.2,hop_lw=0.05,site_size=plotsize,site_color=family_color,site_lw=0.02,fig_size=[10,8])
    plot.tight_layout()
    plt.show()
    if save: plot.savefig(filename+'NR.svg',format='svg',dpi=3600)
    plt.pause(5)
    plt.close()
    (eig,P),eta=scipy.linalg.eigh(fsyst.hamiltonian_submatrix(sparse=False)),etaco[0]*abs(omega)+etaco[1]
    eta[int(np.round(len(omega)/2))]=1e-6
    return np.abs(P[imp][:])**2/np.linalg.norm(np.abs(P[imp][:])),np.sum([(abs(P[imp][i])**2)/(omega-eigv+1.j*eta) 
                                    for i,eigv in enumerate(eig)],axis=0),eig,np.sum([(abs(P[imp][i])**2)
                                    /(np.linspace(min(omega),max(omega),omegastat)-eigv+1.j*(etaco[0]*abs(np.linspace(min(omega),max(omega),omegastat))+etaco[1])) 
                                    for i,eigv in enumerate(eig)],axis=0)

def GrapheneNRzigzagstruct(W:float=2.5,L:float=12,x:float=-11.835680518387328,dy:float=0.5,Wo:float=0,Lo:float=0,t:float=1)->FiniteSystem:
    """``GrapheneNRzigzagstruct(W=2.5,L=12,x=-11.835680518387328,dy=0.5,Wo=0,Lo=0,t=1)``.\n
Defines graphene zigzag structure based on given parameters."""
    lat,sys=kwant.lattice.Polyatomic([[np.sqrt(3)/2,0.5],[0,1]],[[-1/np.sqrt(12),-0.5],[1/np.sqrt(12),-0.5]]),kwant.Builder()
    sys[lat.shape(ribbon(W,L),(0,0))],sys[lat.neighbors(1)]=0,-t
    del sys[lat.shape(ribbon(W,dy,x,0),(x,0))],sys[lat.shape(ribbon(Wo,Lo,-x,-W),(-x,-W))],sys[lat.shape(ribbon(Wo,Lo,-x,W),(-x,W))]
    return sys.finalized()

def GrapheneNRarmchairstruct(W:float=3,L:float=12,y:float=-2.8867513459481287,Wo:float=0,Lo:float=0,t:float=1)->FiniteSystem:
    """``GrapheneNRarmchairstruct(W=3,L=12,y=-2.8867513459481287,Wo=0,Lo=0,t=1)``.\n
Defines graphene armchair structure based on given parameters."""
    lat,sys=kwant.lattice.Polyatomic([[1,0],[0.5,np.sqrt(3)/2]],[[0,1/np.sqrt(3)],[0,0]]),kwant.Builder()
    sys[lat.shape(ribbon(W,L),(0,0))],sys[lat.neighbors(1)]=0,-t
    del sys[lat.shape(ribbon(Wo,Lo,L,y),(L,y))],sys[lat.shape(ribbon(Wo,Lo,-L,y),(-L,y))]
    return sys.finalized()

def ribbon(W:float,L:float,x:float=0,y:float=0)->bool:
    """``ribbon(W,L,x=0,y=0)``.\n
Returns dimensions of ribbon structure."""
    def shape(pos):
        return (-L<=pos[0]-x<=L and -W<=pos[1]-y<=W)
    return shape

def Graphenecirclestruct(r:float=1.5,t:float=1)->FiniteSystem:
    """``Graphenecirclestruct(r=1.5,t=1)``.\n
Defines graphene circular structure based on given parameters."""
    def circle(pos):
        return pos[0]**2+pos[1]**2<r**2
    lat,syst=kwant.lattice.honeycomb(norbs=1),kwant.Builder()
    syst[lat.shape(circle,(0,0))],syst[lat.neighbors()]=0,-t
    return syst.finalized()

def Graphene_main(psi:NDArray[np.float64],SPG:NDArray[np.complex128],eig:NDArray[np.float64],SPrho0:NDArray[np.complex128],N:int=200000,poles:int=4,U:float=3,Sigma:float=3/2,Ed:float=-3/2,SizeO:int=4001,etaco:list[float]=[0.02,1e-24],ctype:str='n',Edcalc:str='',bound:float=8,eigsel:bool=False,Tk:list[float]=[0],Nimpurities:int=1,U2:float=0,J:float=0,posb:int=1,log:bool=False,base:float=1.5,poleDOS:bool=False)->tuple[tuple[NDArray[np.float64],NDArray[np.complex128]],NDArray[np.complex128],NDArray[np.float64],list[NDArray[np.float64]],float]:
    """``Graphene_main(psi,SPG,eig,SPrho0,N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,SizeO=4001,etaco=[0.02,1e-24],ctype='n',Edcalc='',bound=8,eigsel=False,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5,poleDOS=False)``.\n 
The main Graphene nanoribbon DED function simulating the Anderson impurity model on a defined graphene structure for given parameters."""
    if log: omega,selectpcT,selectpT,Npoles,pbar=np.concatenate((-np.logspace(np.log(bound)/np.log(base),np.log(1e-5)/np.log(base),int(np.round(SizeO/2)),base=base),np.logspace(np.log(1e-5)/np.log(base),np.log(bound)/np.log(base),int(np.round(SizeO/2)),base=base))),[],[],int(poles/Nimpurities),trange(N,position=posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    else: omega,selectpcT,selectpT,Npoles,pbar=np.linspace(-bound,bound,SizeO),[],[],int(poles/Nimpurities),trange(N,position=posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    c,eta,rhoint=[Jordan_wigner_transform(i,2*poles) for i in range(2*poles)],etaco[0]*abs(omega)+etaco[1],-np.imag(SPrho0)/np.pi*((max(omega)-min(omega))/len(SPrho0))/sum(-np.imag(SPrho0)/np.pi*((max(omega)-min(omega))/len(SPrho0)))
    (Hn,n),AvgSigmadat,Nfin,nd=Operators(c,Nimpurities,poles),np.zeros((len(Tk),SizeO),dtype='complex_'),np.zeros(len(Tk),dtype='complex_'),np.zeros(len(Tk),dtype='complex_')
    while pbar.n<N:
        reset = False
        while not reset:
            if eigsel:NewM,nonG,select=Startrans(Npoles,np.sort(np.random.choice(eig,Npoles,p=psi,replace=False)),omega,eta)
            else:NewM,nonG,select=Startrans(Npoles,np.sort(np.random.choice(np.linspace(-bound,bound,len(rhoint)),Npoles,p=rhoint,replace=False)),omega,eta)
            H0,H=HamiltonianAIM(np.repeat(NewM[0][0],Nimpurities),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(Nimpurities,1)),np.tile(NewM[0,1:],(Nimpurities,1)),U,Sigma,U2,J,Hn)
            try:(MBGdat,Boltzmann,Ev0),reset=Constraint(ctype,H0,H,omega,eta,c,n,Tk,np.array([ar<N for ar in Nfin]),poleDOS)
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence):(MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
            if np.isnan(1/nonG-1/MBGdat+Sigma).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+Sigma)]).any() or np.array([i>=500 for i in np.abs(1/nonG-1/MBGdat+Sigma)]).any():reset=False
            selectpT.append(select)
        Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+Sigma)*Boltzmann[:,None],nd+np.conj(Ev0).T@sum(Hn[0]).data.tocoo()@Ev0*Boltzmann
        selectpcT.append(select)
        if ctype=='sn':pbar.n+=1
        else:pbar.n=int(min(Nfin))
        pbar.refresh()
    pbar.close()
    if Edcalc=='AS': return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(1/(1/SPG-AvgSigmadat/Nfin[:,None]+(AvgSigmadat[:,int(np.round(SizeO/2))]/Nfin)[:,None]))/np.pi).squeeze(),-np.imag(SPG)/np.pi,omega,selectpT,selectpcT,pbar.format_dict["elapsed"]
    else: return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(1/(1/SPG-AvgSigmadat/Nfin[:,None]-Ed))/np.pi).squeeze(),-np.imag(SPG)/np.pi,omega,selectpT,selectpcT,pbar.format_dict["elapsed"]

def PolestoDOS(select:NDArray[np.float64],selectnon:NDArray[np.float64]=[],ratio:float=200,bound:float=3)->tuple[NDArray[np.float64],list[np.float64]]:
    """``PolestoDOS(select,selectnon=np.zeros(4),ratio=200,bound=3)``.\n 
Function that calculates distribution of selected sites based on the results of the DED algorithm."""
    bar=int(len(select)/ratio)
    bomega=np.linspace(-bound,bound,bar)
    if selectnon==[]:selectnon=np.zeros(len(select))
    DOSp=[((bomega[i]<select)&(select<=bomega[i+1])).sum() for i in range(0,bar-1)]
    return np.linspace(-bound,bound,bar-1),DOSp/(2*bound/(bar-1)*sum(DOSp)),[np.mean(DOSp[j-1:j+1]) 
                                                         for j in range(1,bar-2)],[((bomega[i]<selectnon)&(selectnon<=bomega[i+1])).sum() 
                                                                                   for i in range(0,bar-1)]

def polesfDOS(DOSp:NDArray[np.float64],bound:float=3,corrfactor:float=1)->NDArray[np.float64]:
    """``polesfDOS(DOSp,bound=3,corrfactor=1)``.\n 
Function that calculates distribution of selected sites based counts of poles derived from DED simulation."""
    return np.nan_to_num(DOSp/(2*bound/len(DOSp)*sum(DOSp))*corrfactor)

def DOSplot(fDOS:NDArray[np.float64],Lor:NDArray[np.float64],omega:NDArray[np.float64],name:str,labels:str,log:bool=False,ymax:float=1.2,save:bool=True,fDOScolor:str='b',bound:float=0)->Figure:
    """``DOSplot(fDOS,Lor,omega,name,labels,log=False,ymax=1.2,save=True,fDOScolor='b',bound=0)``.\n 
A plot function to present results from the AIM moddeling for a single results with a comparison to the non-interacting DOS."""
    fig,axis_font=plt.figure(figsize=(10,8)),{'fontname':'Calibri','size':'25'}
    if bound==0:bound=max(omega)
    plt.rc('legend',fontsize=17)
    plt.rc('font',size=25)
    plt.rc('xtick',labelsize=25,color='black')
    plt.rc('ytick',labelsize=25,color='black')
    plt.xlim(-bound,bound)
    if not log:
        plt.gca().set_ylim(bottom=0,top=ymax)
        plt.gca().set_xticks(np.linspace(-bound,bound,2*int(bound)+1),minor=False)
    else: 
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=10)
        plt.gca().set_xticks(np.linspace(-bound,bound,int(bound)+int(bound)%2+1),minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font,color='black')
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font,color='black')
    plt.plot(omega,Lor,'--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,fDOScolor,label=labels)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(name+'.png',format='png')
        plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return fig

def DOSmultiplot(omega:NDArray[np.float64],omegap:NDArray[np.float64],DOST:NDArray[np.float64],plotp:NDArray[np.int32],labels:str,name:str,rho0:NDArray[np.float64],log:bool=False,ymax:float=1.2,save:bool=True,colors:list[str]=['crimson','darkorange','lime','turquoise','cyan','dodgerblue','darkviolet','deeppink'])->Figure:
    """``DOSmultiplot(omega,omegap,DOST,plotp,labels,name,rho0,log=False,ymax=1.2,save=True,colors=['crimson','darkorange','lime','turquoise','cyan','dodgerblue','darkviolet','deeppink'])``.\n
Multi plot function to combine datasets in one graph for comparison including a defined non-interacting DOS."""
    fig,axis_font=plt.figure(figsize=(10,8)),{'fontname':'Calibri','size':'18'}
    plt.rc('legend',fontsize=18)
    plt.rc('font',size=18)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    plt.xlim(min(omega),max(omega))
    if not log:
        plt.gca().set_ylim(bottom=0,top=ymax)
        plt.gca().set_xticks(np.linspace(min(omega),max(omega),2*int(max(omega))+1),minor=False)
    else: 
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=ymax)
        plt.gca().set_xticks(np.linspace(min(omega),max(omega),int(max(omega))+int(max(omega))%2+1),minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,rho0,'--',color='black',linewidth=4,label='$\\rho_0$')
    for i,p in enumerate(plotp): plt.plot(omegap[i,:p],DOST[i,:p],colors[i],linewidth=2,label=labels[i],alpha=0.7)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(name+'.png',format='png')
        plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return fig

def DOSxlogplot(fDOS:NDArray[np.float64],Lor:NDArray[np.float64],omega:NDArray[np.float64],name:str,labels:str,ymax:float=1.2,save:bool=True,xloglim:float=1e-3,incneg:bool=True,fDOScolor:str='b')->Figure:
    """``DOSxlogplot(fDOS,Lor,omega,name,labels,ymax=1.2,save=True,xloglim=1e-3,incneg=True,fDOScolor='b')``.\n
A plot function with a logarithmic x-axis to present results from the AIM moddeling for a single results with a comparison to the non-interacting DOS."""
    fig,axis_font=plt.figure(figsize=(10+incneg*10,8)),{'fontname':'Calibri','size':'18'}
    plt.rc('legend',fontsize=17)
    plt.rc('font',size=18)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    if incneg:
        ax1=fig.add_subplot(121)
        ax2=fig.add_subplot(122)
        ax1.set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font)
        ax1.set_xscale('log')
        ax1.invert_xaxis()
        ax1.set_xticks([1,1e-1,1e-2,1e-3,1e-4,1e-5])
        ax1.set_xlim(max(omega),xloglim)
        ax1.set_ylim(0,ymax)
        ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,pos: '$-\\mathdefault{10^{'+f'{int(np.log10(x))}'+'}}$'))
        ax1.plot(-omega[:len(omega)//2],Lor[:len(omega)//2],'--r',linewidth=4,label='$\\rho_0$')
        ax1.plot(-omega[:len(omega)//2],fDOS[:len(omega)//2],fDOScolor,label=labels)
        ax2.set_xlim(xloglim,max(omega))
        ax2.set_xscale('log')
        ax2.set_ylim(0,ymax)
        ax2.yaxis.set_tick_params(labelleft=False)
        ax2.plot(omega[len(omega)//2:],Lor[len(omega)//2:],'--r',linewidth=4,label='$\\rho_0$')
        ax2.plot(omega[len(omega)//2:],fDOS[len(omega)//2:],fDOScolor,label=labels)
        ax1.grid()
        ax2.grid()
        ax1.set_xlabel("$\\omega$ [-]",**axis_font)
        ax2.set_xlabel("$\\omega$ [-]",**axis_font)
    else:
        plt.xscale('log')
        plt.gca().set_xticks([1,1e-1,1e-2,1e-3,1e-4,1e-5],minor=False)
        plt.gca().set_xlim(xloglim,max(omega))
        plt.gca().set_ylim(bottom=0,top=ymax)
        plt.xlabel("$\\omega$ [-]",**axis_font)
        plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font)
        plt.plot(omega[len(omega)//2:],Lor[len(omega)//2:],'--r',linewidth=4,label='$\\rho_0$')
        plt.plot(omega[len(omega)//2:],fDOS[len(omega)//2:],fDOScolor,label=labels)
        plt.grid()
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.tight_layout()
    if save:
        plt.savefig(name+'.png',format='png')
        plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return fig

def Entropyplot(Tk:NDArray[np.float64],S_imp:NDArray[np.float64],labels:list[str],name:str,colors:list[str]=['crimson','darkorange','goldenrod','lime','turquoise','cyan','dodgerblue','darkviolet','deeppink'])->Figure:
    """``Entropyplot(Tk,S_imp,labels,name,colors=['crimson','darkorange','goldenrod','lime','turquoise','cyan','dodgerblue','darkviolet','deeppink'])``.\n
Entropy plot function to present results from the AIM moddeling for a single or multiple results."""
    fig,axis_font=plt.figure(figsize=(10,8)),{'fontname':'Calibri','size':'17'}
    plt.rc('legend',fontsize=17)
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    plt.xlim(min(Tk),max(Tk))
    plt.xscale('log')
    plt.xlabel("$k_BT$ [-]",**axis_font)
    plt.gca().set_ylabel("$S_{imp}$($k_B$)",va="bottom",rotation=0,labelpad=40,**axis_font)
    plt.gca().set_ylim(bottom=0,top=1.4)
    if len(S_imp.shape)==1:plt.plot(Tk,S_imp,'-r',linewidth=2,label=labels)
    else:
        for i,S in enumerate(S_imp):plt.plot(Tk,S,'-',color=colors[i],linewidth=2,label=labels[i])
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png',format='png')
    plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return fig

def stdplot(Nstdev:NDArray[np.int32],stdavg:NDArray[np.float64],name:str,labelname:str,ymax:float=0.012)->Figure:
    """``stdplot(Nstdev,stdavg,name,labelname,ymax=0.012)``.\n
Plotting function for showing the standard deviation of the calculated DED iterations versus the number of iterations."""
    fig,axis_font=plt.figure(figsize=(10,8)),{'fontname':'Calibri','size':'17'}
    plt.rc('legend',fontsize=17)
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    plt.xlim(min(Nstdev),max(Nstdev))
    plt.xscale('log')
    plt.xlabel("$N$ [-]",**axis_font)
    plt.gca().set_ylabel("$\\sigma$($N$)",va="bottom",rotation=0,labelpad=30,**axis_font)
    plt.gca().set_ylim(bottom=0,top=ymax)
    plt.plot(Nstdev,np.mean(stdavg,axis=1),'-',color='black',linewidth=2,label=labelname)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png',format='png')
    plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return fig
    
def textfileW(omega:NDArray[np.float64],selectpT:list[NDArray[np.float64]],selectpcT:list[NDArray[np.float64]],fDOS:NDArray[np.float64],name:str,AvgSigmadat:NDArray[np.complex128]=[],savpoles:bool=True)->None:
    """``textfileW(omega,selectpT,selectpcT,fDOS,name,AvgSigmadat=[],savpoles=True)``.\n
``.txt`` file writing function for DED results."""
    if AvgSigmadat==[]:np.savetxt(name+'.txt',np.transpose([omega,fDOS]),fmt='%.18g',delimiter='\t',newline='\n')
    else:np.savetxt(name+'.txt',np.c_[omega,fDOS,np.real(AvgSigmadat),np.imag(AvgSigmadat)],fmt='%.18f\t%.18f\t(%.18g%+.18gj)',delimiter='\t',newline='\n')
    if savpoles:
        np.savetxt(name+'polesC'+'.txt',selectpcT,delimiter='\t',newline='\n')
        np.savetxt(name+'poles'+'.txt',selectpT,delimiter='\t',newline='\n')

def textfileR(name:str)->NDArray:
    """``textfileR(name)``.\n
``.txt`` file reader to read DED data writen by ``textfileW(...)``."""
    text_file=open(name,"r")
    lines=text_file.read().split('\n')
    text_file.close()
    return np.array([np.array(l,dtype=object).astype(np.complex) for l in [lines[i].split('\t') for i, _ in enumerate(lines[1:])]]).T