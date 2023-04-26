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
    # Comparison of DED spectra for the symmetric Anderson model for several constaints and sites
    input={"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n'}
    file,labelnames='constraintN2p','$\\rho_{constr.},N,$n=2'
    nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input)
    DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames)
    #DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)

    #input=[{"N" : 2000, "poles" : 2, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 7.5, "Sigma" : 3.75, "Ed" : -7.5/2, "ctype" : 'n'}]
    #filenames,labelnames=tqdm(['cN4p0U','cN4p1_5U','cN4p3U','cN4p4_5U','cN4p6U','cN4p7_5U'],position=0,leave=False,desc='No. SAIM DED U sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\it{U=0.0}$','$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$','$\it{U=7.5}$']
    #DOST=np.zeros((len(filenames),1001),dtype = 'float')
    #for i,file in enumerate(filenames):
    #    nd, _, DOST[i], Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
    #filenames.close()
    #DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'Utotal',DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])
