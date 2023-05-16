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
    #Temperature dependence interacting impurity DOS with modified constraint for the temperature
    input=[{"N" : 200000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : ' ', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : 'sn', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : 'ssn', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]}]
    filenames,labelnames,conname,pbar=['cN4pT1e-12','cN4pT1e-3','cN4pT1e-2','cN4pT1e-1','cN4pT3e-1','cN4pT1'],['$\it{k_bT= %.3f}$'%0.000,'$\it{k_bT= %.3f}$'%0.001,'$\it{k_bT= %.3f}$'%0.010,'$\it{k_bT= %.3f}$'%0.100,'$\it{k_bT= %.3f}$'%0.300,'$\it{k_bT= %.3f}$'%1.000],['no','soft','smartsoft'],tqdm(input,position=0,leave=False,desc='No. T-dependent SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    DOST=np.zeros((len(input),len(input[0]["Tk"]),1001),dtype = 'complex_')
    for j,inpt in enumerate(pbar):
        nd, _, DOST[j], Lor, omega, selectpT, selectpcT,tsim=DEDlib.main(**inpt)
        for i,file in enumerate(filenames):
            DEDlib.DOSplot(DOST[j][i], Lor, omega,conname[j]+file,labelnames[i])
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[j][i],conname[j]+file)
        DEDlib.DOSmultiplot(omega,np.tile(omega, (len(input[0]["Tk"]),1)),DOST[j],np.tile(len(omega), len(input[0]["Tk"])),labelnames,conname[j]+'Ttotal',Lor)
        np.savetxt(conname[j]+'Ttotalnd.txt',nd,delimiter='\t', newline='\n')
    pbar.close()
    #conlabel,fDOS=['$\it{k_bT= %.0f}$, (constr.)'%0,'$\it{k_bT= %.0f}$, (no constr.)'%0,'$\it{k_bT= %.0f}$, (constr.)'%1,'$\it{k_bT= %.0f}$, (no constr.)'%1],[DOST[0][0],DOST[1][0],DOST[0][5],DOST[1][5]]
    #DEDlib.DOSmultiplot(omega,np.tile(omega, (4,1)),fDOS,np.tile(len(omega), 4),conlabel,'constrTtotal',Lor)