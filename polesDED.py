import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
from tqdm.auto import tqdm,trange
import time
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import kwant
import scipy
from scipy.optimize import fsolve

import DEDlib

nd,_,fDOS,Lor,omega,selectpT,selectpcT,tsim=DEDlib.main(**{"N":200000,"poles":6,"Ed":-3/2,"ctype":'n'})
DEDlib.DOSplot(fDOS,Lor,omega,'constraintN6p','$\\rho_{constr.},N,$n=6')
DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,'constraintN6p',savpoles=False)