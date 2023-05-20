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
    #Two-orbital Anderson model impurity DOS
    input=[{"N":200000,"poles":4,"Gamma":0.096,"U":2,"Sigma":1,"Ed":-1,"U2":2,"J":0,"ctype":'n'},
        {"N":20000,"poles":6,"Gamma":0.096,"U":2,"Sigma":1,"Ed":-1,"U2":2,"J":0,"ctype":'n'},
        {"N":200000,"poles":4,"Gamma":0.3,"U":3,"Sigma":3/2*3,"Ed":-3/2*3,"U2":3,"J":0,"ctype":'n',"bound":4},
        {"N":20000,"poles":6,"Gamma":0.3,"U":3,"Sigma":3/2*3,"Ed":-3/2*3,"U2":3,"J":0,"ctype":'n',"bound":4},
        {"N":200000,"poles":4,"Gamma":0.3,"U":3.5,"Sigma":4.0,"Ed":-4.0,"U2":2.5,"J":0.5,"ctype":'n',"bound":5},
        {"N":20000,"poles":6,"Gamma":0.3,"U":3.5,"Sigma":4.0,"Ed":-4.0,"U2":2.5,"J":0.5,"ctype":'n',"bound":5}]
    filenames,DOST,labelnames,nd,ymax=trange(['4p2U2U\'-1Ed','6p2U2U\'-1Ed','4p3U3U\'-4_5Ed','6p3U3U\'-4_5Ed','4p3_5U2_5U\'0_5J-4Ed','6p3_5U2_5U\'0_5J-4Ed'],position=0,leave=False,desc='No. Multi-orbital DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),np.zeros((len(input),1001),dtype='float'),['$\it{U,U\'=2,\\epsilon_d=-1}$,n=4','$\it{U,U\'=2,\\epsilon_d=-1}$,n=6','$\it{U,U\'=3,\\epsilon_d=-4.5}$,n=4','$\it{U,U\'=3,\\epsilon_d=-4.5}$,n=6','$\it{U=3.5,J=0.5,\\epsilon_d=-4}$,n=4','$\it{U=3.5,J=0.5,\\epsilon_d=-4}$,n=6'],np.zeros((len(input),2),dtype='float'),[4,4,1.2,1.2,1.2,1.2]      
    for i,file in enumerate(filenames):
        nd[i],NewSigma,DOST[i],Lor,omega,selectpT,selectpcT,tsim=DEDlib.main(**input,Nimpurities=2)
        DEDlib.DOSplot(DOST[i],Lor,omega,'multiorb2cN'+file,labelnames[i],ymax=ymax[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],'multiorb2cN'+file,NewSigma)
        if i%2==1: DEDlib.DOSmultiplot(omega,np.tile(omega,(2,1)),DOST[i-1:i+1],np.tile(len(omega),2),labelnames[i-1:i+1],'multiorb2cN'+str(int(i/2)+1),Lor,ymax=ymax[i])
    filenames.close()
    np.savetxt('multiorb2cNnd.txt',nd,delimiter='\t',newline='\n')
