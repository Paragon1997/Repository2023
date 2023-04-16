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
    #Interacting graphene impurity DOS and Interacting graphene nanoribbon center/edge DOS of Anderson impurity model
    input=[[{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8, "eigsel" : False}],
            [{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8, "eigsel" : True}]]
    for l,inp in enumerate(input):
        radius,colorbnd,ip,nd,selecm=[1.5,2.3,3.1,4.042,5.1],[7,19,37,61,91],[3,9,18,30,45],np.zeros((5,4),dtype = 'float'),['','eigval']
        for j,r in enumerate(radius):
            filenames,labelnames=['GrapheneCirc'+str(r)+'r1_5U','GrapheneCirc'+str(r)+'r3U','GrapheneCirc'+str(r)+'r4_5U','GrapheneCirc'+str(r)+'r6U'],['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
            DOST=np.zeros((len(filenames),4001),dtype = 'float')
            psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(ip[j],DEDlib.Graphenecirclestruct(r,1),colorbnd[j],'GrapheneCirc'+str(r)+'r')
            for i,file in enumerate(filenames):
                if j==1: inp[i]['ctype'],inp[i]['Ed']='dn','AS'
                else: inp[i]['ctype'],inp[i]['Ed']='n',-inp[i]['U']/2
                nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp[i])
                DEDlib.DOSplot(DOST[i], nonintrho, omega,file+selecm[l],labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file+selecm[l])
            DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneCirc'+str(r)+'r'+selecm[l],nonintrho,log=True)
        np.savetxt('GrapheneCirc'+selecm[l]+'nd',nd,delimiter='\t', newline='\n')
        posimp,func,args,colorbnd,structname,nd=[[85,248],[74,76]],[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((2,4),dtype = 'float')
        for k,pos in enumerate(posimp):
            for j,imp in enumerate(pos):
                filenames,labelnames=['GrapheneNR'+structname[k]+str(imp)+'pos1_5U','GrapheneNR'+structname[k]+str(imp)+'pos3U','GrapheneNR'+structname[k]+str(imp)+'pos4_5U','GrapheneNR'+structname[k]+str(imp)+'pos6U'],['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
                DOST=np.zeros((len(filenames),4001),dtype = 'float')
                psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[k](*args[k]),colorbnd[k],'GrapheneNR'+structname[k]+str(imp)+'pos')
                for i,file in enumerate(filenames):
                    nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp[i])
                    DEDlib.DOSplot(DOST[i], nonintrho, omega,file+selecm[l],labelnames[i],log=True)
                    DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file+selecm[l])
                DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneNR'+structname[k]+str(imp)+'pos'+selecm[l],nonintrho,log=True)
            np.savetxt('GrapheneNR'+structname[k]+selecm[l]+'nd',nd,delimiter='\t', newline='\n')
    
    #Stop here###############################################################################


    
    #add graphene U sim for all structures, vary t=1 to check impact nanoribbons

    # Need redos of all single impurity with Ed=Ed not sigma dependent to check if this influences symmetry U=6 for example as test
    # Problem was the index of sigmadat for Ed (now corrected)

    ######################################################################### Extra simulation to check if no constraint is correct for n=6 (does not work yet needs more testing)##########################################################################

    ######################################################################### Improve to n=7 ####################################################################

    