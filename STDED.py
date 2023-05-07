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
    #The impurity entropy calculated by DED for different constraints and site quantities
    input=[{"N" : 200000, "poles" : 2},{"N" : 200000, "poles" : 4},{"N" : 20000, "poles" : 6}]
    ctypes,filenames,labelnames,S_imp,S_tot,S_bath=tqdm(['n',' ','sn'],position=0,leave=False,desc='No. constraints Entropy DED',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),tqdm(['ST2p','ST4p','ST6p'],position=1,leave=False,desc='No. poles Entropy DED',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$constr. N,$n=','$no constr. N,$n=','$soft constr. N,$n='],np.zeros((9,801)),np.zeros((9,801)),np.zeros((9,801))
    for j,c in enumerate(ctypes):
        for i,file in enumerate(filenames):
            S_imp[3*j+i],S_tot[3*j+i],S_bath[3*j+i],Nfin,Tk,tsim=DEDlib.Entropyimp_main(**input[i],ctype=c,posb=2)
            DEDlib.Entropyplot(Tk,S_imp[3*j+i],labelnames[j]+str(input[i]["poles"]),file+c)
            np.savetxt(file+c+'.txt',np.c_[Tk,S_imp[3*j+i],S_tot[3*j+i],S_bath[3*j+i]],delimiter='\t', newline='\n')
        filenames.close()
    ctypes.close()
    DEDlib.Entropyplot(Tk,S_imp,np.char.add(np.repeat(labelnames,len(input)),np.tile([str(inp["poles"]) for inp in input],len(labelnames))),'STtotal')