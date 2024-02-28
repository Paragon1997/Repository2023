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
    #Interacting graphene nanoribbon center DOS of Anderson impurity model for various t values
    input=tqdm([{"N":200000,"poles":4,"Ed":-3/2,"ctype":'n',"bound":8,"eigsel":False},
    {"N":200000,"poles":4,"Ed":-3/2,"ctype":'n',"bound":8,"eigsel":True}],position=0,leave=False,desc='No. selection type sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    labelnames,selecm=['$\it{t= 0.1}$','$\it{t= 0.5}$','$\it{t= 1.0}$','$\it{t= 1.5}$','$\it{t= 2.0}$'],['','eigval']
    t,func,args,colorbnd,structname,nd=[0.1,0.5,1.0,1.5,2.0],[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((2,5,2),dtype='float')
    for l,inp in enumerate(input):
        posimp=tqdm([85,74],position=1,leave=False,desc='No. Graphene A/Z NR SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for j,imp in enumerate(posimp):
            filenames,DOST,nonintrho=tqdm(['cssnt0_1','cssnt0_5','cssnt1','cssnt1_5','cssnt2'],position=2,leave=False,desc='No. t variation sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),np.zeros((len(t),4001),dtype='float'),np.zeros((len(t),4001),dtype='float')
            for i,file in enumerate(filenames):
                psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[j](*args[j],t=t[i]),colorbnd[j],'GrapheneNR'+structname[j]+str(imp)+'pos')
                nd[j,i],AvgSigmadat,DOST[i],nonintrho[i],omega,selectpT,selectpcT,tsim=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp,posb=3)
                DEDlib.DOSplot(DOST[i],nonintrho[i],omega,'GrapheneNR'+file+structname[j]+selecm[l],labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],'GrapheneNR'+file+structname[j]+selecm[l])
            filenames.close()
            DEDlib.DOSmultiplot(omega,np.tile(omega,(len(filenames),1)),DOST,np.tile(len(omega),len(filenames)),labelnames,'GrapheneNRt'+structname[l]+str(imp)+'pos'+selecm[l],nonintrho[2],log=True)
        posimp.close()
        np.savetxt('tGrapheneNR'+selecm[l]+'nd.txt',nd,delimiter='\t',newline='\n')
    input.close()