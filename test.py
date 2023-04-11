import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tqdm.notebook import tqdm
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

import numba
from numba import jit

import DEDlib

input=[{"N" : 200, "poles" : 2, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 100, "poles" : 3, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 4000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 20000, "poles" : 6, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 4000, "poles" : 4, "Ed" : -3/2, "ctype" : ' '},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n%2'}]
filenames,labelnames=['constraintN2p','constraintN3p','constraintN4p','constraintN5p','constraintN6p','noconstraintN4p','constraint%2N4p'],['$\\rho_{constr.},N,$n=2','$\\rho_{constr.},N,$n=3','$\\rho_{constr.},N,$n=4','$\\rho_{constr.},N,$n=5','$\\rho_{constr.},N,$n=6','$\\rho_{no constr.},$n=4','$\\rho_{constr.},$$N\\%$2,n=4']
i=1
file=filenames[i]
nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames[i])
DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)
i=5
file=filenames[i]
nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames[i])
DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)