import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tqdm.auto import tqdm,trange
import time
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import kwant
import math
from math import  sqrt
import cmath
from numpy.lib.scimath import sqrt as csqrt
import scipy
from scipy.optimize import fsolve
import copy
from itertools import repeat

import DEDlib

if __name__ == '__main__':
    #Stdev calculator as a function of N
    filename,labelnames,Nstdev,nstd='stdevN4p',['Population $\\rho \it{n=4}$','$\\pm 3\\sigma$','DED \it{n=4}$'],np.logspace(2,5,num=50,base=10,dtype='int'),20
    Npbar,stdev=tqdm(Nstdev,position=0,leave=False,desc='No. SAIM DED stdev(N) calculations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),np.zeros((len(Nstdev),1001))
    for i,N in enumerate(Npbar):
        pbar,DOST=trange(nstd,position=1,leave=False,desc='No. SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),np.zeros((nstd,1001),dtype='complex_')
        for j in pbar:
            _,_,DOST[j],_,omega,_,_,tsim=DEDlib.main(N=N,posb=2)
        pbar.close()
        stdev[i]=np.sqrt(np.sum([(DOS-np.mean(DOST,axis=0))**2 for DOS in DOST],axis=0)/(len(DOST)-1))
    Npbar.close()
    stdavg=stdev/np.sqrt(len(DOST))
    DEDlib.stdplot(Nstdev,stdavg,filename,labelnames[2])
    np.savetxt(filename+'.txt',np.insert(stdev,0,Nstdev,axis=1),delimiter='\t',newline='\n')