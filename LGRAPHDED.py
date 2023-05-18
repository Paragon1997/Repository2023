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
    #Interacting DOS of Anderson impurity model in the center of a large graphene nanoribbon structure
    input=tqdm([[{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
        {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
        {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
        {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8, "eigsel" : False}],
        [{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
        {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
        {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
        {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8, "eigsel" : True}]],position=0,leave=False,desc='No. selection type sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    posimp,func,args,colorbnd,structname,selecm,labelnames=3823,DEDlib.GrapheneNRarmchairstruct,(41,40,-40.991869112463434),7647,'LargeStruct',['','eigval'],['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
    psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(posimp,func(*args),colorbnd,'Graphene'+structname)
    for l,inp in enumerate(input):
        filenames,DOST,nd=tqdm(['GrapheneNR'+structname+'1_5U','GrapheneNR'+structname+'3U','GrapheneNR'+structname+'4_5U','GrapheneNR'+structname+'6U'],position=1,leave=False,desc='No. U variation sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),np.zeros((len(filenames),4001),dtype = 'float'),np.zeros((4,2),dtype = 'float')
        for i,file in enumerate(filenames):
            nd[i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT,tsim=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp[i],posb=2)
            DEDlib.DOSplot(DOST[i], nonintrho, omega,file+selecm[l],labelnames[i],log=True)
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file+selecm[l])
        filenames.close()
        DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneNR'+structname+selecm[l],nonintrho,log=True)
        np.savetxt('GrapheneNR'+structname+selecm[l]+'nd.txt',nd,delimiter='\t', newline='\n')
    input.close()