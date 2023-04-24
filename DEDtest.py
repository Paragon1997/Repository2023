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
    input={"N" : 2000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n'}
    file,labelnames='constraintN2p','$\\rho_{constr.},N,$n=2'
    nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input)
    DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames)
    #DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)

