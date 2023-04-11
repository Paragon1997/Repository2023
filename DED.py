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

import DEDlib as DEDlib

if __name__ == '__main__':
    
    #Stop here###############################################################################
    #Interacting graphene impurity DOS of Anderson impurity model
    input=[{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8},
            {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8},
            {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8},
            {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8}]
    radius,colorbnd,imp,nd=[1.5,2.3,3.1,4.042,5.1],[7,19,37,61,91],[3,9,18,30,45],np.zeros((5,4),dtype = 'float')
    for j,r in enumerate(radius):
        filenames,labelnames=['GrapheneCirc'+str(r)+'r1_5U','GrapheneCirc'+str(r)+'r3U','GrapheneCirc'+str(r)+'r4_5U','GrapheneCirc'+str(r)+'r6U'],['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
        DOST=np.zeros((len(filenames),4001),dtype = 'float')
        psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp[j],DEDlib.Graphenecirclestruct(r,1),colorbnd[j],'GrapheneCirc'+str(r)+'r')
        for i,file in enumerate(filenames):
            if j==1: input[i]['ctype']='dn'
            nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**input[i])
            DEDlib.DOSplot(DOST[i], nonintrho, omega,file,labelnames[i],log=True)
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file)
        DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneCirc'+str(r)+'r',nonintrho,log=True)
    np.savetxt('GrapheneCircnd',nd,delimiter='\t', newline='\n')
    
    #Interacting graphene nanoribbon center/edge DOS of Anderson impurity model
    posimp,func,args,colorbnd,structname,nd=[[85,248],[74,76]],[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((2,4),dtype = 'float')
    for k,pos in enumerate(posimp):
        for j,imp in enumerate(pos):
            filenames,labelnames=['GrapheneNR'+structname[k]+str(imp)+'pos1_5U','GrapheneNR'+structname[k]+str(imp)+'pos3U','GrapheneNR'+structname[k]+str(imp)+'pos4_5U','GrapheneNR'+structname[k]+str(imp)+'pos6U'],['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
            DOST=np.zeros((len(filenames),4001),dtype = 'float')
            psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[k](*args[k]),colorbnd[k],'GrapheneNR'+structname[k]+str(imp)+'pos')
            for i,file in enumerate(filenames):
                nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**input[i])
                DEDlib.DOSplot(DOST[i], nonintrho, omega,file,labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file)
            DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneNR'+structname[k]+str(imp)+'pos',nonintrho,log=True)
        np.savetxt('GrapheneNR'+structname[k]+'nd',nd,delimiter='\t', newline='\n')
    
    #Temperature dependence interacting impurity DOS
    Tk=[0.000000000001,0.001,0.01,0.1,1]
    
    #add graphene U sim for all structures, vary t=1 to check impact nanoribbons

    # Need redos of all single impurity with Ed=Ed not sigma dependent to check if this influences symmetry U=6 for example as test
    # Problem was the index of sigmadat for Ed (now corrected)

    ######################################################################### Extra simulation to check if no constraint is correct for n=6 (does not work yet needs more testing)##########################################################################

    ######################################################################### Improve to n=7 ####################################################################

    