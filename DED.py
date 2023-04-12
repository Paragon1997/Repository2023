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
    # Comparison of DED spectra for the symmetric Anderson model for several constaints and sites
    input=[{"N" : 200000, "poles" : 2, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 3, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 20000, "poles" : 6, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : ' '},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n%2'}]
    filenames,labelnames=['constraintN2p','constraintN3p','constraintN4p','constraintN5p','constraintN6p','noconstraintN4p','constraint%2N4p'],['$\\rho_{constr.},N,$n=2','$\\rho_{constr.},N,$n=3','$\\rho_{constr.},N,$n=4','$\\rho_{constr.},N,$n=5','$\\rho_{constr.},N,$n=6','$\\rho_{no constr.},$n=4','$\\rho_{constr.},$$N\\%$2,n=4']
    for i,file in enumerate(filenames):
        nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)

    # Series of sampled non-interacting DOS for different number of sites compared to the original Lorentzian non-interacting DOS
    input=[{"N" : 200000, "poles" : 2, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n'},
    {"N" : 200000, "poles" : 2, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 3, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 5, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n%2'}]
    filenames,labelnames=['Lorentz2p0U','Lorentz2p3U','Lorentz3p3U','Lorentz4p3U','Lorentz5p3U','Lorentz4p3U%2N'],['$\\rho_0,\it{U=0,n=2}$','$\\rho_0,\it{U=3,n=2}$','$\\rho_0,\it{U=3,n=3}$','$\\rho_0,\it{U=3,n=4}$','$\\rho_0,\it{U=3,n=5}$','$\\rho_0,\it{U=3,n=4},N\\%$2']
    DOST=np.zeros((len(filenames),int(input[-2]["N"]*input[-2]["poles"]/200)-1),dtype = 'float')
    omegap=np.zeros((len(filenames),int(input[-2]["N"]*input[-2]["poles"]/200)-1),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        omegap[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],DOST[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],DOSsm,DOSnon=DEDlib.PolestoDOS(np.ravel(selectpcT),np.ravel(selectpT))
        DEDlib.DOSplot(DOST[i,:int(input[i]["N"]*input[i]["poles"]/200)-1], DEDlib.Lorentzian(omegap[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],0.3,4)[0], omegap[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],file,labelnames[i])
        DEDlib.textfileW(omegap[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],np.ravel(selectpT),np.ravel(selectpcT),DOST[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],file)
    DEDlib.DOSmultiplot(omega,omegap,DOST,[int(input[i]["N"]*input[i]["poles"]/200)-1 for i,_ in enumerate(filenames)],labelnames,'selection',DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])

    # Interacting impurity DOS for different Coulomb repulsion strengths characterized by different values of U
    input=[{"N" : 200000, "poles" : 4, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 7.5, "Sigma" : 3.75, "Ed" : -7.5/2, "ctype" : 'n'}]
    filenames,labelnames=['cN4p0U','cN4p1_5U','cN4p3U','cN4p4_5U','cN4p6U','cN4p7_5U'],['$\it{U=0.0}$','$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$','$\it{U=7.5}$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(DOST[i], Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file)
    DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'Utotal',DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])

    # Interacting impurity DOS for different magnitudes of correlation strength by varying the flat hybridization
    input=[{"N" : 200000, "poles" : 4, "Gamma" : 0.15, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Gamma" : 0.30, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Gamma" : 0.45, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Gamma" : 0.60, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Gamma" : 0.75, "Ed" : -3/2, "ctype" : 'n'}]
    filenames,labelnames=['cN4p_15G','cN4p_3G','cN4p_45G','cN4p_6G','cN4p_75G'],['$\Gamma\it{=0.15}$','$\\Gamma\it{=0.30}$','$\\Gamma\it{=0.45}$','$\\Gamma\it{=0.60}$','$\\Gamma\it{=0.75}$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(DOST[i], Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file)
    DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'Gtotal',DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])

    # Interacting single impurity DOS for different functions of η
    input=[{"N" : 200000, "poles" : 4, "etaco" : [0.01,1e-39], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.02,1e-39], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.04,1e-39], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.00,0.001], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.00,0.01], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.00,0.1], "Ed" : -3/2, "ctype" : 'n'}]
    filenames,labelnames=['cN4p_01+1e-39eta','cN4p_02+1e-39eta','cN4p_04+1e-39eta','cN4p_00+0.001eta','cN4p_00+0.01eta','cN4p_00+0.1eta'],['$\\eta=0.01|\\omega|$','$\\eta=0.02|\\omega|$','$\\eta=0.04|\\omega|$','$\\eta=0.001$','$\\eta=0.010$','$\\eta=0.100$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(DOST[i], Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file)
    DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'etatotal',DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])

    # Interacting DOS of asymmetric Anderson impurity model
    input=[{"N" : 200000, "poles" : 4, "Ed" : -1.5, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -1.65, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -1.8, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -2, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -2.5, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -3, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4}]
    filenames=['cN4p-1_5Ed','cN4p-1_65Ed','cN4p-1_8Ed','cN4p-2Ed','cN4p-2_5Ed','cN4p-3Ed']
    for i,file in enumerate(filenames):
        DOST,labelnames,nd=np.zeros((8,1001),dtype = 'float'),np.chararray(8, itemsize=23),np.zeros(8,dtype = 'float')
        for j,_ in enumerate(DOST):
            (nd[j], NewSigma, DOST[j], Lor, omega, selectpT, selectpcT),labelnames[j]=DEDlib.main(**input[i]),'$\\rho,\\Sigma_0=%.3f$'%input[i]['Sigma']
            DEDlib.DOSplot(DOST[j], Lor, omega,file+'%.16fSigma'%input[i]['Sigma'],'$\\rho,\\Sigma_0=%.3f$'%input[i]['Sigma'])
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[j],file+'%.16fSigma'%input[i]['Sigma'])
            if np.isclose(input[i]['Sigma'],np.real(NewSigma[500]),rtol=5e-4, atol=1e-06): break
            elif j<len(DOST)-1: input[i]['Sigma']=np.real(NewSigma[500])
        np.savetxt(file+'%.16fSigma'%input[i]['Sigma']+'nd',nd,delimiter='\t', newline='\n')
        DEDlib.DOSmultiplot(omega,np.tile(omega, (j+1,1)),DOST[~np.all(DOST == 0, axis=1)],np.tile(len(omega), j+1),labelnames.astype(str),'Asymtotal'+filenames[i],input[i]['Ed'],input[i]['Sigma'],DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])
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
    
    
    #add graphene U sim for all structures, vary t=1 to check impact nanoribbons

    # Need redos of all single impurity with Ed=Ed not sigma dependent to check if this influences symmetry U=6 for example as test
    # Problem was the index of sigmadat for Ed (now corrected)

    ######################################################################### Extra simulation to check if no constraint is correct for n=6 (does not work yet needs more testing)##########################################################################

    ######################################################################### Improve to n=7 ####################################################################

    