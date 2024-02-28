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
    #Impurity entropy for different Gamma values of the SAIM
    input={"N" : 2000, "poles" : 6, "ctype" : 'sn'}
    Gamma,filenames,labelnames,S_imp,S_tot,S_bath=[0.2,0.3,0.5,0.9],tqdm(['ST0_2G','ST0_3G','ST0_5G','ST0_9G'],position=0,leave=False,desc='No. Gamma Entropy DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\Gamma\it{=0.2}$','$\Gamma\it{=0.3}$','$\Gamma\it{=0.5}$','$\Gamma\it{=0.9}$'],np.zeros((4,801)),np.zeros((4,801)),np.zeros((4,801))
    for i,file in enumerate(filenames):
        S_imp[i],S_tot[i],S_bath[i],Nfin,Tk,tsim=DEDlib.Entropyimp_main(**input,Gamma=Gamma[i])
        DEDlib.Entropyplot(Tk,S_imp[i],labelnames[i],file)
        np.savetxt(file+'.txt',np.c_[Tk,S_imp[i],S_tot[i],S_bath[i]],delimiter='\t', newline='\n')
    filenames.close()
    DEDlib.Entropyplot(Tk,S_imp,labelnames,'STtotalG')