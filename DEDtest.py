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
    input=tqdm([{"N":20000,"poles":4,"Ed":-3/2,"ctype":'ssn',"bound":8,"eigsel":False,"Tk":[0.000000000001,0.001,0.01,0.1,0.3,1]},
        {"N":20000,"poles":4,"Ed":-3/2,"ctype":'ssn',"bound":8,"eigsel":True,"Tk":[0.000000000001,0.001,0.01,0.1,0.3,1]}],position=0,leave=False,desc='No. selection type sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    filenames,posimp,labelnames,selecm=['4pT1e-12','4pT1e-3','4pT1e-2','4pT1e-1','4pT3e-1','4pT1'],[85,74],['$\it{k_bT= %.3f}$'%0.000,'$\it{k_bT= %.3f}$'%0.001,'$\it{k_bT= %.3f}$'%0.010,'$\it{k_bT= %.3f}$'%0.100,'$\it{k_bT= %.3f}$'%0.300,'$\it{k_bT= %.3f}$'%1.000],['','eigval']
    func,args,colorbnd,structname,nd=[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((len(posimp),2,len(filenames)),dtype='float')
    for l,inp in enumerate(input):
        txtfile,posimp=open('test'+selecm[l]+'nd.txt','w'),tqdm(posimp,position=1,leave=False,desc='No. Graphene A/Z NR SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for j,imp in enumerate(posimp):
            psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[j](*args[j]),colorbnd[j],'test'+structname[j]+str(imp)+'pos')
            nd[j],AvgSigmadat,DOST,nonintrho,omega,selectpT,selectpcT,tsim=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp,posb=2)
            for i,file in enumerate(filenames):
                DEDlib.DOSplot(DOST[i],nonintrho,omega,'GrapheneNR'+file+structname[j]+selecm[l],labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],'GrapheneNR'+file+structname[j]+selecm[l])
            DEDlib.DOSmultiplot(omega,np.tile(omega,(len(inp["Tk"]),1)),DOST,np.tile(len(omega),len(inp["Tk"])),labelnames,'GTtotal'+structname[j]+selecm[l],nonintrho,log=True)
            np.savetxt(txtfile,nd[j],delimiter='\t',newline='\n')
            txtfile.write('\n')    
        posimp.close()
        txtfile.close()
    input.close()


    # Comparison of DED spectra for the symmetric Anderson model for several constaints and sites
    #input={"N" : 4000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n'}
    #file,labelnames='constraintN2p','$\\rho_{constr.},N,$n=2'
    #nd, _, fDOS, Lor, omega, selectpT, selectpcT,tsim=DEDlib.main(**input)
    DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames)
    #DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)

    #input=[{"N" : 2000, "poles" : 2, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n'},
    #{"N" : 2000, "poles" : 2, "U" : 7.5, "Sigma" : 3.75, "Ed" : -7.5/2, "ctype" : 'n'}]
    #filenames,labelnames=tqdm(['cN4p0U','cN4p1_5U','cN4p3U','cN4p4_5U','cN4p6U','cN4p7_5U'],position=0,leave=False,desc='No. SAIM DED U sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\it{U=0.0}$','$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$','$\it{U=7.5}$']
    #DOST=np.zeros((len(filenames),1001),dtype = 'float')
    #for i,file in enumerate(filenames):
    #    nd, _, DOST[i], Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
    #filenames.close()
    #DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'Utotal',DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])

    #input=[{"N" : 10, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : 'ssn', "Tk" : [0.000000000001,0.001,0.01,0.1,1]}]
    #filenames,labelnames,conname=['cN4pT1e-12','cN4pT1e-3','cN4pT1e-2','cN4pT1e-1','cN4pT1'],['$\it{k_bT= %.3f}$'%0.000,'$\it{k_bT= %.3f}$'%0.001,'$\it{k_bT= %.3f}$'%0.010,'$\it{k_bT= %.3f}$'%0.100,'$\it{k_bT= %.3f}$'%1.000],['','no','soft']
    #DOST=np.zeros((len(input),len(input[0]["Tk"]),1001),dtype = 'complex_')
    #j=0#vary this###############################
    #inpt=input[j]
    #nd, Avgs, DOST[j], Lor, omega, selectpT, selectpcT=DEDlib.main(**inpt)
    #for i,file in enumerate(filenames):
    #    DEDlib.DOSplot(DOST[j][i], Lor, omega,file,labelnames[i])

    print(tsim)
    #print(len(selectpcT))

    #print(selectpcT)

"""     filename,labelnames,Nstdev,stdev='stdevN4p',['Population $\\rho \it{n=4}$','$\\pm 3\\sigma$','DED \it{n=4}$'],np.logspace(2, 5, num=50, base=10,dtype='int'),np.zeros((50,1001))
    Npbar,pbar=tqdm(Nstdev,position=0,leave=False,desc='No. SAIM DED stdev(N) calc',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),trange(20,position=1,leave=False,desc='No. SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i,N in enumerate(Npbar):
        DOST=np.zeros((20,1001),dtype = 'complex_')
        for j in pbar:
            _, _, DOST[j], _, omega, _, _=DEDlib.main(N=N,posb=2)
        pbar.close()
        stdev[i]=np.sqrt(np.sum([(DOS-np.mean(DOST,axis=0))**2 for DOS in DOST],axis=0)/(len(DOST)-1))
        print(stdev)
    Npbar.close()
    stdavg=stdev/np.sqrt(len(DOST))
    DEDlib.stdplot(Nstdev,stdavg,filename,labelnames[2])
    np.savetxt(filename+'.txt',(Nstdev,stdev),delimiter='\t', newline='\n') """