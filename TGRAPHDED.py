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
    #Temperature dependence interacting graphene nanoribbon center DOS of Anderson impurity model
    input=tqdm([{"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'ssn', "bound" : 8, "eigsel" : False,"Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]},
           {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'ssn', "bound" : 8, "eigsel" : True,"Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]}],position=0,leave=False,desc='No. selection type sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    filenames,labelnames,selecm=['4pT1e-12','4pT1e-3','4pT1e-2','4pT1e-1','4pT3e-1','4pT1'],['$\it{k_bT= %.3f}$'%0.000,'$\it{k_bT= %.3f}$'%0.001,'$\it{k_bT= %.3f}$'%0.010,'$\it{k_bT= %.3f}$'%0.100,'$\it{k_bT= %.3f}$'%0.300,'$\it{k_bT= %.3f}$'%1.000],['','eigval']
    func,args,colorbnd,structname,nd=[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((2,2,6),dtype = 'float')
    for l,inp in enumerate(input):
        posimp=tqdm([85,74],position=1,leave=False,desc='No. Graphene A/Z NR SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for j,imp in enumerate(posimp):
            psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[j](*args[j]),colorbnd[j],'GrapheneNR'+structname[j]+str(imp)+'pos')
            nd[j], AvgSigmadat, DOST, nonintrho, omega, selectpT, selectpcT,tsim=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp,posb=2)
            for i,file in enumerate(filenames):
                DEDlib.DOSplot(DOST[i], nonintrho, omega,'GrapheneNR'+file+structname[j]+selecm[l],labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],'GrapheneNR'+file+structname[j]+selecm[l])
            DEDlib.DOSmultiplot(omega,np.tile(omega, (len(inp["Tk"]),1)),DOST,np.tile(len(omega), len(inp["Tk"])),labelnames,'GTtotal'+structname[j]+selecm[l],nonintrho,log=True)
        posimp.close()
        np.savetxt('TGrapheneNR'+selecm[l]+'nd.txt',nd,delimiter='\t', newline='\n')
    input.close()   