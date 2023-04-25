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
    #Interacting graphene impurity DOS and Interacting graphene nanoribbon center/edge DOS of Anderson impurity model
    input=tqdm([[{"N" : 2000, "poles" : 4, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n', "bound" : 8, "eigsel" : False}],
            [{"N" : 2000, "poles" : 4, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n', "bound" : 8, "eigsel" : True}]],position=0,leave=False,desc='No. selection type sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for l,inp in enumerate(input):
        radius,colorbnd,ip,nd,selecm=tqdm([1.5,2.3,3.1,4.042,5.1],position=1,leave=False,desc='No. Graphene circular NR SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),[7,19,37,61,91],[3,9,18,30,45],np.zeros((5,4),dtype = 'float'),['','eigval']
        for j,r in enumerate(radius):
            filenames,labelnames=tqdm(['GrapheneCirc'+str(r)+'r0U'],position=2,leave=False,desc='No. U variation sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\it{U=0}$']
            DOST=np.zeros((len(filenames),4001),dtype = 'float')
            psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(ip[j],DEDlib.Graphenecirclestruct(r,1),colorbnd[j],'GrapheneCirc'+str(r)+'r')
            for i,file in enumerate(filenames):
                if j==1: inp[i]['Edcalc']='AS'
                else: inp[i]['Edcalc']=''
                nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp[i],posb=3)
                DEDlib.DOSplot(DOST[i], nonintrho, omega,file+selecm[l],labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file+selecm[l])
            filenames.close()
            DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneCirc'+str(r)+'r'+selecm[l],nonintrho,log=True)
