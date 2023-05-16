## module DEDlib
''' DEDlib is a Distributional Exact Diagonalization tooling library for study of Anderson (multi-)impurity model in Graphene Nanoribbons'''

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tqdm.auto import trange
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import kwant
from numpy import sqrt
import scipy
from numba import njit

#class DED:

def Jordan_wigner_transform(j,lattice_length):
    """Jordan_wigner_transform(j, lattice_length). 
Defines the Jordan Wigner transformation for a 1D lattice."""
    operators=sigmaz()
    for _ in range(j-1): operators=tensor(operators,sigmaz())
    if j == 0: operators=sigmam()
    else: operators=tensor(operators,sigmam())
    for _ in range(lattice_length - j - 1): operators=tensor(operators,identity(2))
    return operators

@njit
def Lorentzian(omega,Gamma,poles,Ed=-3/2,Sigma=3/2):
    """Lorentzian(omega, Gamma, poles,Ed=-3/2,Sigma=3/2). 
Defines the non-interacting DOS (rho0) and selects random sites based on the number of sites in the 1D lattice model and the calculated distribution."""
    return -np.imag(1/(omega-Ed-Sigma+1j*Gamma))/np.pi,np.array([Gamma*np.tan(np.pi*(pi-1/2))+Ed+Sigma for pi in np.random.uniform(0,1,poles)])

@njit
def Startrans(poles,select,omega,eta,row=0):
    """Startrans(poles,select,row,omega, eta). 
Function to transform 1D lattice matrices in order to calculates parameters impengergy, bathenergy and Vkk from random sampling distribution."""
    Pbath,Dbath,pbar,G=np.zeros((poles,poles)),np.zeros((poles,poles)),np.zeros((poles,poles)),np.zeros(omega.shape,dtype='complex_')
    for i in range(poles-1):
        for j in range(poles-1):
            if j>=i: Pbath[i+1][j+1]=-1/sqrt((poles-i-1)*(poles-i))
        Pbath[i+1][i]=sqrt(poles-i-1)/sqrt(poles-i)
    Pbath[row,:]=1/sqrt(poles)
    for i, _ in enumerate(select): Dbath[i][i]=select[i]
    pbar[1:,1:]=np.linalg.eig((Pbath@Dbath@Pbath.T)[1:,1:])[1]
    pbar[row][row]=1
    for i, _ in enumerate(select): G+=1/len(select)/(omega-select[i]+1.j*eta)
    return pbar.T@Pbath@Dbath@Pbath.T@pbar,G,select

def Operators(c,Nimpurities,poles):
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

def HamiltonianAIM(impenergy,bathenergy,Vkk,U,Sigma,U2,J,Hn,H0=0):
    """HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma). 
Based on energy parameters calculates the Hamiltonian of a single-impurity system."""
    for k in range(len(impenergy)):
        H0+=impenergy[k]*Hn[0][k]
        for j in range(len(bathenergy[k])): H0+=bathenergy[k][j]*Hn[1][k][j]+Vkk[k][j]*Hn[2][k][j]
    return H0,H0+U*Hn[3]-Sigma*Hn[4]+(U2/2-J/4)*Hn[5]-J*Hn[6]

@njit
def MBGT0(omega,eta,evals,exp,exp2):
    G=np.zeros(len(omega),dtype='complex_')
    for i,expi in enumerate(exp): G+=abs(expi)**2/(omega+evals[i+1]-evals[0]+1.j*eta)+abs(exp2[i])**2/(omega+evals[0]-evals[i+1]+1.j*eta)
    return G

@njit
def MBGTnonzero(omega,eta,evals,exp,exp2,eevals):
    G=np.zeros(len(omega),dtype='complex_')
    for i,evi in enumerate(evals):
        for j,evj in enumerate(evals):
            G+=(exp[i][j]*exp2[j][i]/(omega+evi-evj+1.j*eta)+exp[j][i]*exp2[i][j]/(omega+evj-evi+1.j*eta))*eevals[i]
    return G

def MBGAIM(omega,H,c,eta,Tk,Boltzmann,evals=[],evecs=[],etaoffset=1e-4,posoffset=np.zeros(1,dtype='int')):
    """MBGAIM(omega, H, c, eta). 
Calculates the many body Green's function based on the Hamiltonian eigenenergies/-states."""
    if ~np.any(evals): evals,evecs=scipy.linalg.eigh(H.data.toarray())
    if Tk==[0]:
        vecn=np.conj(evecs[:,1:]).T
        exp,exp2=vecn@c[0].data.tocoo()@evecs[:,0],vecn@c[0].dag().data.tocoo()@evecs[:,0]
        return MBGT0(omega,eta,evals,exp,exp2),Boltzmann,evecs[:,0]
    else:
        MGdat,eta[int(np.round(len(eta)/2))+posoffset]=np.ones((len(Tk),len(omega)),dtype='complex_'),etaoffset
        for k,T in enumerate(Tk):
            if Boltzmann[k]!=0:
                eevals=np.exp(-evals/T-scipy.special.logsumexp(-evals/T))
                vecn=np.conj(evecs).T
                exp,exp2=vecn@c[0].data.tocoo()@evecs,vecn@c[0].dag().data.tocoo()@evecs
                MGdat[k,:]=MBGTnonzero(omega,eta,evals,exp,exp2,eevals)
        return MGdat.squeeze(),Boltzmann,evecs[:,0]

def Constraint(ctype,H0,H,omega,eta,c,n,Tk,Nfin):
    """Constraint(ctype,H0,H,omega,eta,c,n). 
Constraint implementation function for DED method with various possible constraints."""
    if ctype[0]=='s':
        vecs=scipy.linalg.eigh(H0.data.toarray(),eigvals=[0,0])[1][:,0]
        evals,evecs=scipy.linalg.eigh(H.data.toarray())
        if ctype=='ssn':
            Boltzmann=np.exp(-abs(evals[find_nearest(np.diag(np.conj(evecs).T@n[0].data@evecs),np.conj(vecs)@n[0].data@vecs.T)]-evals[0])/Tk)*Nfin.astype('int')
            return MBGAIM(omega,H,c,eta,Tk,Boltzmann,evals,evecs,1e-4,np.array([-2,-1,0,1,2])),True
        else:
            return MBGAIM(omega,H,c,eta,Tk,np.exp(-abs(evals[find_nearest(np.diag(np.conj(evecs).T@n[0].data@evecs),np.conj(vecs)@n[0].data@vecs.T)]-evals[0])/Tk),evals,evecs,1e-4,np.array([-2,-1,0,1,2])),True
    elif ctype[0]=='n':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.sparse.linalg.eigsh(np.real(H0.data),k=1,which='SA')[1][:,0],
                                                scipy.sparse.linalg.eigsh(np.real(H.data),k=1,which='SA')[1][:,0])))
        exp=[np.conj(vecs)@ni.data@vecs.T for ni in n]
        if ctype=='n%2' and all([int(np.round(expi[0,0]))%2==int(np.round(expi[1,1]))%2 for expi in exp]):
            return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
        elif ctype=='n' and all([np.round(expi[0,0])==np.round(expi[1,1]) for expi in exp]):
            return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
        else:
            return (np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
    elif ctype[0]=='d':
        vecs=scipy.sparse.csr_matrix(np.vstack((scipy.linalg.eigh(H.data.toarray(),eigvals=[0,0])[1][:,0],
                                                scipy.linalg.eigh(H0.data.toarray(),eigvals=[0,0])[1][:,0])))
        exp=[np.conj(vecs)@ni.data@vecs.T for ni in n]
        if ctype=='dn' and all([np.round(expi[0,0])==np.round(expi[1,1]) for expi in exp]):
            return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
        else:
            return (np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
    else:
        return MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
    
def find_nearest(array,value):
    for i in (i for i,arrval in enumerate(array) if np.isclose(arrval,value,atol=0.1)): return i

def main(N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1):
    """main(N=1000000,poles=4,U=3,Sigma=3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39], ctype='n',Ed='AS'). 
The main DED function simulating the Anderson impurity model for given parameters."""
    omega,eta,selectpcT,selectpT,Npoles=np.linspace(-bound,bound,SizeO),etaco[0]*abs(np.linspace(-bound,bound,SizeO))+etaco[1],[],[],int(poles/Nimpurities)
    c,pbar=[Jordan_wigner_transform(i,2*poles) for i in range(2*poles)],trange(N,position=posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    (Hn,n),AvgSigmadat,Nfin,nd=Operators(c,Nimpurities,poles),np.zeros((len(Tk),SizeO),dtype='complex_'),np.zeros(len(Tk),dtype='float'),np.zeros(len(Tk),dtype='complex_')
    while pbar.n<N:
        reset=False
        while not reset:
            NewM,nonG,select=Startrans(Npoles,np.sort(Lorentzian(omega,Gamma,Npoles,Ed,Sigma)[1]),omega,eta)
            H0,H=HamiltonianAIM(np.repeat(NewM[0][0],Nimpurities),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(Nimpurities,1)),np.tile(NewM[0,1:],(Nimpurities,1)),U,Sigma,U2,J,Hn)
            try: (MBGdat,Boltzmann,Ev0),reset=Constraint(ctype,H0,H,omega,eta,c,n,Tk,np.array([ar<N for ar in Nfin]))
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
            if np.isnan(1/nonG-1/MBGdat+Sigma).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+Sigma)]).any(): reset=False
            selectpT.append(select)
        Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+Sigma)*Boltzmann[:,None],nd+np.conj(Ev0).T@(c[0].dag()*c[0]+c[1].dag()*c[1]).data.tocoo()@Ev0*Boltzmann
        selectpcT.append(select)
        if ctype=='sn': pbar.n+=1
        else: pbar.n=int(min(Nfin))
        pbar.refresh()
    pbar.close()
    if Edcalc == 'AS': return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]+(AvgSigmadat[:,int(np.round(SizeO/2))]/Nfin)[:,None]+1j*Gamma)))/np.pi).squeeze(),Lorentzian(omega,Gamma,poles)[0],omega,selectpT,selectpcT,pbar.format_dict["elapsed"]
    else: return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]-Ed+1j*Gamma)))/np.pi).squeeze(),Lorentzian(omega,Gamma,poles,Ed,Sigma)[0],omega,selectpT,selectpcT,pbar.format_dict["elapsed"]

def ConstraintS(ctype,H0,H,n,Tk,Nfin=0):
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
def SAIM(evals,Z_tot,Tk,kb,E_k,constr,S_t,S_b,S_imp,Nfin):
    S_tot,S_bath=kb*(Z_tot+evals@np.exp(np.outer(-evals,1/Tk)-Z_tot)/Tk),np.zeros(len(Tk))
    for ek in E_k: S_bath+=2*kb*(np.logaddexp(np.zeros(len(Tk)),-ek/Tk)+ek/np.exp(np.logaddexp(np.zeros(len(Tk)),ek/Tk))/Tk)
    return S_t+S_tot*constr,S_b+S_bath*constr,S_imp+(S_tot-S_bath)*constr,Nfin+constr

def Entropyimp_main(N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',bound=3,Tk=np.logspace(-6,2,801,base=10),kb=1,posb=1):
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
        if ~np.any(evals): evals=scipy.linalg.eigvalsh(H.data.toarray())
        Z_tot=scipy.special.logsumexp(np.outer(-evals,1/Tk),axis=0)
        if (Z_tot>2e+08).any(): continue
        else: S_t,S_b,S_imp,Nfin=SAIM(evals,Z_tot,Tk,kb,E_k,constr,S_t,S_b,S_imp,Nfin)
        if ctype=='sn': pbar.n+=1
        else: pbar.n=int(min(Nfin))
        pbar.refresh()
    pbar.close()
    return np.abs(S_imp/Nfin).squeeze(),np.real(S_t/Nfin).squeeze(),np.real(S_b/Nfin).squeeze(),Nfin.squeeze(),Tk,(pbar.format_dict["n"],pbar.format_dict["elapsed"])

def GrapheneAnalyzer(imp,fsyst,colorbnd,filename,omega=np.linspace(-8,8,4001),etaco=[0.02,1e-24],omegastat=100001):
    """GrapheneAnalyzer(imp,fsyst,colorbnd,filename,omega=np.linspace(-8,8,4001),etaco=[0.02,1e-24],omegastat=100001).
Returns data regarding a defined graphene circular structure such as the corresponding Green's function."""
    def plotsize(i): return 0.208 if i == imp else 0.125
    def family_color(i):
        if i == imp: return 'purple'
        elif i<colorbnd: return (31/255,119/255,180/255,255/255)
        else: return (255/255,127/255,14/255,255/255)
    plt.ion()
    plt.rc('legend',fontsize=25)
    plt.rc('font',size=25)
    plt.rc('xtick',labelsize=25)
    plt.rc('ytick',labelsize=25)
    plot=kwant.plot(fsyst,unit=1.2,hop_lw=0.05,site_size=plotsize,site_color=family_color,site_lw=0.02,fig_size=[10,8])
    plot.tight_layout()
    plt.show()
    plot.savefig(filename+'NR.svg',format='svg',dpi=3600)
    plt.pause(5)
    plt.close()
    (eig,P),eta=scipy.linalg.eigh(fsyst.hamiltonian_submatrix(sparse=False)),etaco[0]*abs(omega)+etaco[1]
    eta[int(np.round(len(omega)/2))]=1e-6
    return np.abs(P[imp][:])**2/np.linalg.norm(np.abs(P[imp][:])),np.sum([(abs(P[imp][i])**2)/(omega-eigv+1.j*eta) 
                                    for i,eigv in enumerate(eig)],axis=0),eig,np.sum([(abs(P[imp][i])**2)
                                    /(np.linspace(min(omega),max(omega),omegastat)-eigv+1.j*(etaco[0]*abs(np.linspace(min(omega),max(omega),omegastat))+etaco[1])) 
                                    for i,eigv in enumerate(eig)],axis=0)

def GrapheneNRzigzagstruct(W=2.5,L=12,x=-11.835680518387328,dy=0.5,Wo=0,Lo=0,t=1):
    lat,sys=kwant.lattice.Polyatomic([[sqrt(3)/2,0.5],[0,1]],[[-1/sqrt(12),-0.5],[1/sqrt(12),-0.5]]),kwant.Builder()
    sys[lat.shape(ribbon(W,L),(0,0))],sys[lat.neighbors(1)]=0,-t
    del sys[lat.shape(ribbon(W,dy,x,0),(x,0))],sys[lat.shape(ribbon(Wo,Lo,-x,-W),(-x,-W))],sys[lat.shape(ribbon(Wo,Lo,-x,W),(-x,W))]
    return sys.finalized()

def GrapheneNRarmchairstruct(W=3,L=12,y=-2.8867513459481287,Wo=0,Lo=0,t=1):
    lat,sys=kwant.lattice.Polyatomic([[1,0],[0.5,sqrt(3)/2]],[[0,1/sqrt(3)],[0,0]]),kwant.Builder()
    sys[lat.shape(ribbon(W,L),(0,0))],sys[lat.neighbors(1)]=0,-t
    del sys[lat.shape(ribbon(Wo,Lo,L,y),(L,y))],sys[lat.shape(ribbon(Wo,Lo,-L,y),(-L,y))]
    return sys.finalized()

def ribbon(W,L,x=0,y=0):
    def shape(pos):
        return (-L<=pos[0]-x<=L and -W<=pos[1]-y<=W)
    return shape

def Graphenecirclestruct(r=1.5,t=1):
    def circle(pos):
        return pos[0]**2+pos[1]**2<r**2
    lat,syst=kwant.lattice.honeycomb(norbs=1),kwant.Builder()
    syst[lat.shape(circle,(0,0))],syst[lat.neighbors()]=0,-t
    return syst.finalized()

def Graphene_main(psi,SPG,eig,SPrho0,N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,SizeO=4001,etaco=[0.02,1e-24],ctype='n',Edcalc='',bound=8,eigsel=False,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1):
    """Graphene_main(graphfunc,args,imp,colorbnd,name,N=200000,poles=4,U=3,Sigma=3/2,SizeO=4001,etaco=[0.02,1e-24], ctype='n',Ed='AS',bound=8,eigsel=False). 
The main Graphene nanoribbon DED function simulating the Anderson impurity model on a defined graphene structure for given parameters."""
    omega,AvgSigmadat,selectpcT,selectpT,Npoles,pbar=np.linspace(-bound,bound,SizeO),np.zeros(SizeO,dtype ='complex_'),[],[],int(poles/Nimpurities),trange(N,position=posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    c,eta,rhoint=[Jordan_wigner_transform(i,2*poles) for i in range(2*poles)],etaco[0]*abs(omega)+etaco[1],-np.imag(SPrho0)/np.pi*((max(omega)-min(omega))/len(SPrho0))/sum(-np.imag(SPrho0)/np.pi*((max(omega)-min(omega))/len(SPrho0)))
    (Hn,n),AvgSigmadat,Nfin,nd=Operators(c,Nimpurities,poles),np.zeros((len(Tk),SizeO),dtype='complex_'),np.zeros(len(Tk),dtype='complex_'),np.zeros(len(Tk),dtype='complex_')
    while pbar.n<N:
        reset = False
        while not reset:
            if eigsel: NewM,nonG,select=Startrans(Npoles,np.sort(np.random.choice(eig,Npoles,p=psi,replace=False)),omega,eta)
            else: NewM,nonG,select=Startrans(Npoles,np.sort(np.random.choice(np.linspace(-bound,bound,len(rhoint)),Npoles,p=rhoint,replace=False)),omega,eta)
            H0,H=HamiltonianAIM(np.repeat(NewM[0][0],Nimpurities),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(Nimpurities,1)),np.tile(NewM[0,1:],(Nimpurities,1)),U,Sigma,U2,J,Hn)
            try: (MBGdat,Boltzmann,Ev0),reset=Constraint(ctype,H0,H,omega,eta,c,n,Tk,np.array([ar<N for ar in Nfin]))
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
            if np.isnan(1/nonG-1/MBGdat+Sigma).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+Sigma)]).any() or np.array([i>=500 for i in np.abs(1/nonG-1/MBGdat+Sigma)]).any(): reset=False
            selectpT.append(select)
        Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+Sigma)*Boltzmann[:,None],nd+np.conj(Ev0).T@(c[0].dag()*c[0]+c[1].dag()*c[1]).data.tocoo()@Ev0*Boltzmann
        selectpcT.append(select)
        if ctype=='sn': pbar.n+=1
        else: pbar.n=int(min(Nfin))
        pbar.refresh()
    pbar.close()
    if Edcalc == 'AS': return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(1/(1/SPG-AvgSigmadat/Nfin[:,None]+(AvgSigmadat[:,int(np.round(SizeO/2))]/Nfin)[:,None]))/np.pi).squeeze(),-np.imag(SPG)/np.pi,omega,selectpT,selectpcT,pbar.format_dict["elapsed"]
    else: return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(1/(1/SPG-AvgSigmadat/Nfin[:,None]-Ed))/np.pi).squeeze(),-np.imag(SPG)/np.pi,omega,selectpT,selectpcT,pbar.format_dict["elapsed"]

def PolestoDOS(select,selectnon,ratio=200,bound=3):
    """PolestoDOS(select,selectnon,ratio=200). 
Function with calculated distribution of selected sites based on the results of the DED algorithm."""
    bar=int(len(select)/ratio)
    bomega=np.linspace(-bound,bound,bar)
    DOSp=[((bomega[i]<select)&(select<=bomega[i+1])).sum() for i in range(0,bar-1)]
    return np.linspace(-bound,bound,bar-1),DOSp/(2*bound/(bar-1)*sum(DOSp)),[np.mean(DOSp[j-1:j+1]) 
                                                         for j in range(1,bar-2)],[((bomega[i]<selectnon)&(selectnon<=bomega[i+1])).sum() 
                                                                                   for i in range(0,bar-1)]

def DOSplot(fDOS,Lor,omega,name,labels,log=False):
    """DOSplot(fDOS,Lor,omega,name,labels). 
A plot function to present results from the AIM moddeling for a single results with a comparison to the non-interacting DOS."""
    plt.figure(figsize=(10,8))
    plt.rc('legend',fontsize=17)
    plt.rc('font',size=25)
    plt.rc('xtick',labelsize=25)
    plt.rc('ytick',labelsize=25)
    axis_font={'fontname':'Calibri','size':'25'}
    plt.xlim(min(omega),max(omega))
    if not log: plt.gca().set_ylim(bottom=0,top=1.2)
    else: 
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=10)
        plt.gca().set_xticks([-8,-6,-4,-2,0,2,4,6,8],minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,Lor,'--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,'-b',label=labels)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png',format='png')
    plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return plt

def DOSmultiplot(omega,omegap,DOST,plotp,labels,name,rho0,log=False):
    """DOSmultiplot(omega,omegap,DOST,plotp,labels,name).
Multi plot function to combine datasets in one graph for comparison including a defined non-interacting DOS."""
    colors=['crimson','darkorange','lime','turquoise','cyan','dodgerblue','darkviolet','deeppink']
    plt.figure(figsize=(10,8))
    plt.rc('legend',fontsize=18)
    plt.rc('font',size=18)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    axis_font={'fontname':'Calibri','size':'18'}
    plt.xlim(min(omega),max(omega))
    if not log: plt.gca().set_ylim(bottom=0,top=1.2)
    else: 
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=10)
        plt.gca().set_xticks([-8,-6,-4,-2,0,2,4,6,8],minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,rho0,'--',color='black',linewidth=4,label='$\\rho_0$')
    for i,p in enumerate(plotp): plt.plot(omegap[i,:p],DOST[i,:p],colors[i],linewidth=2,label=labels[i],alpha=0.7)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png',format='png')
    plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return plt

def Entropyplot(Tk,S_imp,labels,name):
    colors=['crimson','darkorange','goldenrod','lime','turquoise','cyan','dodgerblue','darkviolet','deeppink']
    plt.figure(figsize=(10,8))
    plt.rc('legend',fontsize=17)
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    axis_font={'fontname':'Calibri','size':'17'}
    plt.xlim(min(Tk),max(Tk))
    plt.xscale('log')
    plt.xlabel("$k_BT$ [-]",**axis_font)
    plt.gca().set_ylabel("$S_{imp}$($k_B$)",va="bottom",rotation=0,labelpad=40,**axis_font)
    plt.gca().set_ylim(bottom=0,top=1.4)
    if len(S_imp.shape)==1: plt.plot(Tk,S_imp,'-r',linewidth=2,label=labels)
    else:
        for i,S in enumerate(S_imp): plt.plot(Tk,S,'-',color=colors[i],linewidth=2,label=labels[i])
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png',format='png')
    plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return plt

def stdplot(Nstdev,stdavg,name,labelname):
    plt.figure(figsize=(10,8))
    plt.rc('legend',fontsize=17)
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    axis_font={'fontname':'Calibri','size':'17'}
    plt.xlim(min(Nstdev),max(Nstdev))
    plt.xscale('log')
    plt.xlabel("$N$ [-]",**axis_font)
    plt.gca().set_ylabel("$\\sigma$($N$)",va="bottom",rotation=0,labelpad=30,**axis_font)
    plt.gca().set_ylim(bottom=0)
    plt.plot(Nstdev,np.mean(stdavg,axis=1),'-',color='black',linewidth=2,label=labelname)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    plt.savefig(name+'.png',format='png')
    plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.draw()
    plt.pause(5)
    plt.close()
    return plt
    
def textfileW(omega,selectpT,selectpcT,fDOS,name,AvgSigmadat=[]):
    """textfileW(omega,selectpT,selectpcT,fDOS,name).
File writing function for DED results."""
    if AvgSigmadat==[]: np.savetxt(name+'.txt',np.transpose([omega,fDOS]),fmt='%.18g',delimiter='\t',newline='\n')
    else: np.savetxt(name+'.txt',np.c_[omega,fDOS,np.real(AvgSigmadat),np.imag(AvgSigmadat)],fmt='%.18f\t%.18f\t(%.18g%+.18gj)',delimiter='\t',newline='\n')
    np.savetxt(name+'polesC'+'.txt',selectpcT,delimiter='\t',newline='\n')
    np.savetxt(name+'poles'+'.txt',selectpT,delimiter='\t',newline='\n')

def textfileR(name):
    """textfileR(name).
File reader to read DED data writen by textfileW(...)."""
    text_file=open(name,"r")
    lines=text_file.read().split('\n')
    text_file.close()
    return np.array([np.array(l,dtype=object).astype(np.complex) for l in [lines[i].split('\t') for i, _ in enumerate(lines[1:])]]).T