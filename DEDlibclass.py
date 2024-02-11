## module DEDlib
''' DEDlib is a Distributional Exact Diagonalization tooling library for study of Anderson (multi-)impurity model in Graphene Nanoribbons'''

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tqdm.auto import trange
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import kwant
import scipy
from numba import njit

class DEDlib:
    def __init__(self,**kwargs):
        self.Nit,self.poles,self.Uimp,self.Sigma,self.Ed,self.Gamma,self.SizeO,self.etaco,self.ctype,self.Edcalc,self.bound,self.Tk,self.Nimpurities,self.U2imp,self.Jimp,self.posb=200000,4,3,3/2,-3/2,0.3,1001,[0.02,1e-39],'n','',3,[0],1,0,0,1
        for kwarg,val in kwargs.items(): setattr(self,kwarg,val)
        self.omega,self.eta,self.selectpcT,self.selectpT,self.Npoles=np.linspace(-self.bound,self.bound,self.SizeO),self.etaco[0]*abs(np.linspace(-self.bound,self.bound,self.SizeO))+self.etaco[1],[],[],int(self.poles/self.Nimpurities)
        self.c,self.pbar=[self.Jordan_wigner_transform(i,2*self.poles) for i in range(2*self.poles)],trange(self.Nit,position=self.posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        (self.Hn,self.n)=self.Operators(self.c,self.Nimpurities,self.poles)
        self.ndfin,self.NewSigma,self.fDOS,self.Lor,self.tsim=self.main()

    def Jordan_wigner_transform(self,j,lattice_length):
        """Jordan_wigner_transform(j, lattice_length). 
    Defines the Jordan Wigner transformation for a 1D lattice."""
        operators=sigmaz()
        for _ in range(j-1): operators=tensor(operators,sigmaz())
        if j == 0: operators=sigmam()
        else: operators=tensor(operators,sigmam())
        for _ in range(lattice_length - j - 1): operators=tensor(operators,identity(2))
        return operators
    
    def Operators(self,c,Nimpurities,poles):
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

    def main(self):
        AvgSigmadat,Nfin,nd=np.zeros((len(self.Tk),self.SizeO),dtype='complex_'),np.zeros(len(self.Tk),dtype='float'),np.zeros(len(self.Tk),dtype='complex_')
        while self.pbar.n<self.Nit:
            reset=False
            while not reset:
                NewM,nonG,select=self.Startrans(self.Npoles,np.sort(self.Lorentzian(self.omega,self.Gamma,self.Npoles,self.Ed,self.Sigma)[1]),self.omega,self.eta)
                H0,H=self.HamiltonianAIM(np.repeat(NewM[0][0],self.Nimpurities),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(self.Nimpurities,1)),np.tile(NewM[0,1:],(self.Nimpurities,1)),self.Uimp,self.Sigma,self.U2imp,self.Jimp,self.Hn)
                try: (MBGdat,Boltzmann,Ev0),reset=self.Constraint(self.ctype,H0,H,self.omega,self.eta,self.c,self.n,self.Tk,np.array([ar<self.Nit for ar in Nfin]))
                except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(self.omega),dtype='complex_'),np.zeros(len(self.Tk)),np.array([])),False
                if np.isnan(1/nonG-1/MBGdat+self.Sigma).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+self.Sigma)]).any(): reset=False
                self.selectpT.append(select)
            Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+self.Sigma)*Boltzmann[:,None],nd+np.conj(Ev0).T@sum(self.Hn[0]).data.tocoo()@Ev0*Boltzmann
            self.selectpcT.append(select)
            if self.ctype=='sn': self.pbar.n+=1
            else: self.pbar.n=int(min(Nfin))
            self.pbar.refresh()
        self.pbar.close()
        if self.Edcalc == 'AS': return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(np.nan_to_num(1/(self.omega-AvgSigmadat/Nfin[:,None]+(AvgSigmadat[:,int(np.round(self.SizeO/2))]/Nfin)[:,None]+1j*self.Gamma)))/np.pi).squeeze(),self.Lorentzian(self.omega,self.Gamma,self.poles)[0],self.pbar.format_dict["elapsed"]
        else: return (Nfin.squeeze(),np.real(nd/Nfin).squeeze()),(AvgSigmadat/Nfin[:,None]).squeeze(),(-np.imag(np.nan_to_num(1/(self.omega-AvgSigmadat/Nfin[:,None]-self.Ed+1j*self.Gamma)))/np.pi).squeeze(),self.Lorentzian(self.omega,self.Gamma,self.poles,self.Ed,self.Sigma)[0],self.pbar.format_dict["elapsed"]

    #@njit
    def Lorentzian(self,omega,Gamma,poles,Ed=-3/2,Sigma=3/2):
        """Lorentzian(omega, Gamma, poles,Ed=-3/2,Sigma=3/2). 
    Defines the non-interacting DOS (rho0) and selects random sites based on the number of sites in the 1D lattice model and the calculated distribution."""
        return -np.imag(1/(omega-Ed-Sigma+1j*Gamma))/np.pi,np.array([Gamma*np.tan(np.pi*(pi-1/2))+Ed+Sigma for pi in np.random.uniform(0,1,poles)])

    #@njit
    def Startrans(self,poles,select,omega,eta,row=0):
        """Startrans(poles,select,row,omega, eta). 
    Function to transform 1D lattice matrices in order to calculates parameters impengergy, bathenergy and Vkk from random sampling distribution."""
        Pbath,Dbath,pbar,G=np.zeros((poles,poles)),np.zeros((poles,poles)),np.zeros((poles,poles)),np.zeros(omega.shape,dtype='complex_')
        for i in range(poles-1):
            for j in range(poles-1):
                if j>=i: Pbath[i+1][j+1]=-1/np.sqrt((poles-i-1)*(poles-i))
            Pbath[i+1][i]=np.sqrt(poles-i-1)/np.sqrt(poles-i)
        Pbath[row,:]=1/np.sqrt(poles)
        for i, _ in enumerate(select): Dbath[i][i]=select[i]
        pbar[1:,1:]=np.linalg.eig((Pbath@Dbath@Pbath.T)[1:,1:])[1]
        pbar[row][row]=1
        for i, _ in enumerate(select): G+=1/len(select)/(omega-select[i]+1.j*eta)
        return pbar.T@Pbath@Dbath@Pbath.T@pbar,G,select
    
    def HamiltonianAIM(self,impenergy,bathenergy,Vkk,U,Sigma,U2,J,Hn,H0=0):
        """HamiltonianAIM(c, impenergy, bathenergy, Vkk, U, Sigma). 
    Based on energy parameters calculates the Hamiltonian of a single-impurity system."""
        for k in range(len(impenergy)):
            H0+=impenergy[k]*Hn[0][k]
            for j in range(len(bathenergy[k])): H0+=bathenergy[k][j]*Hn[1][k][j]+Vkk[k][j]*Hn[2][k][j]
        return H0,H0+U*Hn[3]-Sigma*Hn[4]+(U2/2-J/4)*Hn[5]-J*Hn[6]
    
    def Constraint(self,ctype,H0,H,omega,eta,c,n,Tk,Nfin):
        """Constraint(ctype,H0,H,omega,eta,c,n). 
    Constraint implementation function for DED method with various possible constraints."""
        if ctype[0]=='m':
            vecs=scipy.sparse.csr_matrix(np.vstack((scipy.sparse.linalg.eigsh(np.real(H0.data),k=1,which='SA')[1][:,0],
                                            scipy.sparse.linalg.eigsh(np.real(H.data),k=1,which='SA')[1][:,0])))
            exp=np.conj(vecs)@n[0].data@vecs.T
            if ctype=='mosn' and np.round(exp[0,0])==np.round(exp[1,1]):
                return self.MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
            else:
                return (np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
        elif ctype[0]=='s':
            vecs=scipy.linalg.eigh(H0.data.toarray(),eigvals=[0,0])[1][:,0]
            evals,evecs=scipy.linalg.eigh(H.data.toarray())
            if ctype=='ssn':
                Boltzmann=np.exp(-abs(evals[self.find_nearest(np.diag(np.conj(evecs).T@n[0].data@evecs),np.conj(vecs)@n[0].data@vecs.T)]-evals[0])/Tk)*Nfin.astype('int')
                return self.MBGAIM(omega,H,c,eta,Tk,Boltzmann,evals,evecs,4e-4,np.array([-2,-1,0,1,2])),True
            else:
                return self.MBGAIM(omega,H,c,eta,Tk,np.exp(-abs(evals[self.find_nearest(np.diag(np.conj(evecs).T@n[0].data@evecs),np.conj(vecs)@n[0].data@vecs.T)]-evals[0])/Tk),evals,evecs,4e-4,np.array([-2,-1,0,1,2])),True
        elif ctype[0]=='n':
            vecs=scipy.sparse.csr_matrix(np.vstack((scipy.sparse.linalg.eigsh(np.real(H0.data),k=1,which='SA')[1][:,0],
                                                    scipy.sparse.linalg.eigsh(np.real(H.data),k=1,which='SA')[1][:,0])))
            exp=[np.conj(vecs)@ni.data@vecs.T for ni in n]
            if ctype=='n%2' and all([int(np.round(expi[0,0]))%2==int(np.round(expi[1,1]))%2 for expi in exp]):
                return self.MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
            elif ctype=='n' and all([np.round(expi[0,0])==np.round(expi[1,1]) for expi in exp]):
                return self.MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
            else:
                return (np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
        elif ctype[0]=='d':
            vecs=scipy.sparse.csr_matrix(np.vstack((scipy.linalg.eigh(H.data.toarray(),eigvals=[0,0])[1][:,0],
                                                    scipy.linalg.eigh(H0.data.toarray(),eigvals=[0,0])[1][:,0])))
            exp=[np.conj(vecs)@ni.data@vecs.T for ni in n]
            if ctype=='dn' and all([np.round(expi[0,0])==np.round(expi[1,1]) for expi in exp]):
                return self.MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
            else:
                return (np.zeros(len(omega),dtype='complex_'),np.zeros(len(Tk)),np.array([])),False
        else:
            return self.MBGAIM(omega,H,c,eta,Tk,np.ones(len(Tk))),True
        
    def MBGAIM(self,omega,H,c,eta,Tk,Boltzmann,evals=[],evecs=[],etaoffset=1e-4,posoffset=np.zeros(1,dtype='int')):
        """MBGAIM(omega, H, c, eta). 
    Calculates the many body Green's function based on the Hamiltonian eigenenergies/-states."""
        if ~np.any(evals): evals,evecs=scipy.linalg.eigh(H.data.toarray())
        if Tk==[0]:
            vecn=np.conj(evecs[:,1:]).T
            exp,exp2=vecn@c[0].data.tocoo()@evecs[:,0],vecn@c[0].dag().data.tocoo()@evecs[:,0]
            return self.MBGT0(omega,eta,evals,exp,exp2),Boltzmann,evecs[:,0]
        else:
            MGdat,eta[int(np.round(len(eta)/2))+posoffset]=np.ones((len(Tk),len(omega)),dtype='complex_'),etaoffset
            for k,T in enumerate(Tk):
                if Boltzmann[k]!=0:
                    eevals,vecn=np.exp(-evals/T-scipy.special.logsumexp(-evals/T)),np.conj(evecs).T
                    exp,exp2=vecn@c[0].data.tocoo()@evecs,vecn@c[0].dag().data.tocoo()@evecs
                    MGdat[k,:]=self.MBGTnonzero(omega,eta,evals,exp,exp2,eevals)
            return MGdat.squeeze(),Boltzmann,evecs[:,0]
        
    #@njit
    def MBGT0(self,omega,eta,evals,exp,exp2):
        G=np.zeros(len(omega),dtype='complex_')
        for i,expi in enumerate(exp): G+=abs(expi)**2/(omega+evals[i+1]-evals[0]+1.j*eta)+abs(exp2[i])**2/(omega+evals[0]-evals[i+1]+1.j*eta)
        return G

    #@njit
    def MBGTnonzero(self,omega,eta,evals,exp,exp2,eevals):
        G=np.zeros(len(omega),dtype='complex_')
        for i,evi in enumerate(evals):
            for j,evj in enumerate(evals):
                G+=(exp[i][j]*exp2[j][i]/(omega+evi-evj+1.j*eta)+exp[j][i]*exp2[i][j]/(omega+evj-evi+1.j*eta))*eevals[i]
        return G
    
    def find_nearest(self,array,value):
        for i in (i for i,arrval in enumerate(array) if np.isclose(arrval,value,atol=0.1)): return i

    def DOSplot(self,name,labels,log=False,ymax=1.2):
        """DOSplot(fDOS,Lor,omega,name,labels). 
    A plot function to present results from the AIM moddeling for a single results with a comparison to the non-interacting DOS."""
        plt.figure(figsize=(10,8))
        plt.rc('legend',fontsize=17)
        plt.rc('font',size=25)
        plt.rc('xtick',labelsize=25)
        plt.rc('ytick',labelsize=25)
        axis_font={'fontname':'Calibri','size':'25'}
        plt.xlim(min(self.omega),max(self.omega))
        if not log:
            plt.gca().set_ylim(bottom=0,top=ymax)
            plt.gca().set_xticks(np.linspace(min(self.omega),max(self.omega),2*int(max(self.omega))+1),minor=False)
        else: 
            plt.yscale('log')
            plt.gca().set_ylim(bottom=0.0001,top=10)
            plt.gca().set_xticks([-8,-6,-4,-2,0,2,4,6,8],minor=False)
        plt.xlabel("$\\omega$ [-]",**axis_font)
        plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font)
        plt.plot(self.omega,self.Lor,'--r',linewidth=4,label='$\\rho_0$')
        plt.plot(self.omega,self.fDOS,'-b',label=labels)
        plt.legend(fancybox=False).get_frame().set_edgecolor('black')
        plt.grid()
        plt.tight_layout()
        plt.savefig(name+'.png',format='png')
        plt.savefig(name+'.svg',format='svg',dpi=3600)
        plt.draw()
        plt.pause(5)
        plt.close()
        return plt