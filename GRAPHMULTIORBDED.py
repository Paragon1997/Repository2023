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
    #Interacting graphene nanoribbon DOS of quarter/half-filled (including and without Hund's rule coupling) two-orbital Anderson impurity model
    input=np.array([[{"U":1,"Sigma":0.5,"Ed":-0.5,"U2":1,"J":0},{"U":2,"Sigma":1,"Ed":-1,"U2":2,"J":0},{"U":3,"Sigma":1.5,"Ed":-1.5,"U2":3,"J":0},{"U":4,"Sigma":2,"Ed":-2,"U2":4,"J":0}],
        [{"U":1.5,"Sigma":2.25,"Ed":-2.25,"U2":1.5,"J":0},{"U":3,"Sigma":4.5,"Ed":-4.5,"U2":3,"J":0},{"U":4.5,"Sigma":6.75,"Ed":-6.75,"U2":4.5,"J":0},{"U":6,"Sigma":9,"Ed":-9,"U2":6,"J":0}],
        [{"U":3.5,"Sigma":4.625,"Ed":-4.625,"U2":3,"J":0.25},{"U":3.5,"Sigma":4,"Ed":-4,"U2":2.5,"J":0.5},{"U":3.5,"Sigma":3.375,"Ed":-3.375,"U2":2,"J":0.75},{"U":3.5,"Sigma":2.75,"Ed":-2.75,"U2":1.5,"J":1}],
        [{"U":1.75,"Sigma":2,"Ed":-2,"U2":1.25,"J":0.25},{"U":3.5,"Sigma":4,"Ed":-4,"U2":2.5,"J":0.5},{"U":5.25,"Sigma":6,"Ed":-6,"U2":3.75,"J":0.75},{"U":7,"Sigma":8,"Ed":-8,"U2":5,"J":1}]])
    imps,pbar=[85,74],tqdm(input,position=0,leave=False,desc='No. Multi-orbital Graphene DED scenarios',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    func,args,colorbnd,structname,nd=[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((len(imps),input.shape[1],2),dtype='float')
    for l,inp in enumerate(pbar):
        txtfile,posimp=open('GrapheneNRmultiorb2cN4p'+str(l+1)+'nd.txt','w'),tqdm(imps,position=1,leave=False,desc='No. Graphene A/Z NR SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        if input[l][0]["J"]==0: labelnames=['$\it{U,U\'='+str(ip["U"])+',\\epsilon_d='+str(ip["Ed"])+'}$' for ip in inp]
        else: labelnames=['$\it{U='+str(ip["U"])+',J='+str(ip["J"])+',\\epsilon_d='+str(ip["Ed"])+'}$' for ip in inp]
        for j,imp in enumerate(posimp):
            filenames,DOST=tqdm([str(ip["U"])+'U'+str(ip["U2"])+'U\''+str(ip["J"])+'J'+str(ip["Ed"])+'Ed' for ip in input[l]],position=2,leave=False,desc='No. U,U\',J variation sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),np.zeros((len(input[l]),4001),dtype='float')
            psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[j](*args[j]),colorbnd[j],'GrapheneNR'+structname[j]+str(imp)+'pos')
            for i,file in enumerate(filenames):
                nd[j,i],AvgSigmadat,DOST[i],nonintrho,omega,selectpT,selectpcT,tsim=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**{"N":200000,"poles":4,"ctype":'n'}|inp[i],Nimpurities=2,posb=3)
                if i==2 and l==3: file+='v2'
                DEDlib.DOSplot(DOST[i],nonintrho,omega,'GrapheneNRmultiorb2cN4p'+structname[j]+file,labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],'GrapheneNRmultiorb2cN4p'+structname[j]+file)
            filenames.close()
            DEDlib.DOSmultiplot(omega,np.tile(omega,(len(filenames),1)),DOST,np.tile(len(omega),len(filenames)),labelnames,'GrapheneNRmultiorb2CN'+structname[j]+str(l+1),nonintrho,log=True)
            np.savetxt(txtfile,nd[j],delimiter='\t',newline='\n')
            txtfile.write('\n')
        posimp.close()
        txtfile.close()
    pbar.close()