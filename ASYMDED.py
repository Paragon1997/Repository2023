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

import DEDlib as DEDlib

if __name__ == '__main__':
    # Interacting DOS of asymmetric Anderson impurity model
    input=[{"N" : 200000, "poles" : 4, "Ed" : -1.5, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -1.65, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -1.8, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -2, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -2.5, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -3, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4}]
    filenames=tqdm(['cN4p-1_5Ed','cN4p-1_65Ed','cN4p-1_8Ed','cN4p-2Ed','cN4p-2_5Ed','cN4p-3Ed'],position=0,leave=False,desc='No. ASAIM DED sims')
    for i,file in enumerate(filenames):
        DOST,labelnames,nd,pbar=np.zeros((8,1001),dtype = 'float'),np.chararray(8, itemsize=23),np.zeros(8,dtype = 'float'),trange(8,position=1,leave=False,desc='Self-consistence iteration')
        for j in pbar:
            (nd[j], NewSigma, DOST[j], Lor, omega, selectpT, selectpcT),labelnames[j]=DEDlib.main(**input[i],posb=2),'$\\rho,\\Sigma_0=%.3f$'%input[i]['Sigma']
            DEDlib.DOSplot(DOST[j], Lor, omega,file+'%.16fSigma'%input[i]['Sigma'],'$\\rho,\\Sigma_0=%.3f$'%input[i]['Sigma'])
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[j],file+'%.16fSigma'%input[i]['Sigma'],NewSigma)
            if np.isclose(input[i]['Sigma'],np.real(NewSigma[500]),rtol=1e-4): break
            input[i]['Sigma']=np.real(NewSigma[500])
        pbar.close()
        np.savetxt(file+'%.16fSigma'%np.real(NewSigma[500])+'nd.txt',nd,delimiter='\t', newline='\n')
        DEDlib.DOSmultiplot(omega,np.tile(omega, (j+1,1)),DOST[~np.all(DOST == 0, axis=1)],np.tile(len(omega), j+1),labelnames[:j+1].astype(str),'Asymtotal'+file,DEDlib.Lorentzian(omega,0.3,4,input[i]['Ed'],3/2)[0])
    filenames.close()