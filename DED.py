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
    # Comparison of DED spectra for the symmetric Anderson model for several constaints and sites
    input=[{"N" : 200000, "poles" : 2, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 3, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 20000, "poles" : 6, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : ' '},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n%2'}]
    filenames,labelnames=tqdm(['constraintN2p','constraintN3p','constraintN4p','constraintN5p','constraintN6p','noconstraintN4p','constraint%2N4p'],position=0,leave=False,desc='No. SAIM DED sims'),['$\\rho_{constr.},N,$n=2','$\\rho_{constr.},N,$n=3','$\\rho_{constr.},N,$n=4','$\\rho_{constr.},N,$n=5','$\\rho_{constr.},N,$n=6','$\\rho_{no constr.},$n=4','$\\rho_{constr.},$$N\\%$2,n=4']
    for i,file in enumerate(filenames):
        nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)
    filenames.close()

    # Series of sampled non-interacting DOS for different number of sites compared to the original Lorentzian non-interacting DOS
    input=[{"N" : 200000, "poles" : 2, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n'},
    {"N" : 200000, "poles" : 2, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 3, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 5, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n%2'}]
    filenames,labelnames=tqdm(['Lorentz2p0U','Lorentz2p3U','Lorentz3p3U','Lorentz4p3U','Lorentz5p3U','Lorentz4p3U%2N'],position=0,leave=False,desc='No. sampling Dist. SAIM DED'),['$\\rho_0,\it{U=0,n=2}$','$\\rho_0,\it{U=3,n=2}$','$\\rho_0,\it{U=3,n=3}$','$\\rho_0,\it{U=3,n=4}$','$\\rho_0,\it{U=3,n=5}$','$\\rho_0,\it{U=3,n=4},N\\%$2']
    DOST=np.zeros((len(filenames),int(input[-2]["N"]*input[-2]["poles"]/200)-1),dtype = 'float')
    omegap=np.zeros((len(filenames),int(input[-2]["N"]*input[-2]["poles"]/200)-1),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        omegap[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],DOST[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],DOSsm,DOSnon=DEDlib.PolestoDOS(np.ravel(selectpcT),np.ravel(selectpT))
        DEDlib.DOSplot(DOST[i,:int(input[i]["N"]*input[i]["poles"]/200)-1], DEDlib.Lorentzian(omegap[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],0.3,4)[0], omegap[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],file,labelnames[i])
        DEDlib.textfileW(omegap[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],np.ravel(selectpT),np.ravel(selectpcT),DOST[i,:int(input[i]["N"]*input[i]["poles"]/200)-1],file)
    filenames.close()
    DEDlib.DOSmultiplot(omega,omegap,DOST,[int(input[i]["N"]*input[i]["poles"]/200)-1 for i,_ in enumerate(filenames)],labelnames,'selection',Lor)

    # Interacting impurity DOS for different Coulomb repulsion strengths characterized by different values of U
    input=[{"N" : 200000, "poles" : 4, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 7.5, "Sigma" : 3.75, "Ed" : -7.5/2, "ctype" : 'n'}]
    filenames,labelnames=tqdm(['cN4p0U','cN4p1_5U','cN4p3U','cN4p4_5U','cN4p6U','cN4p7_5U'],position=0,leave=False,desc='No. SAIM DED U sims'),['$\it{U=0.0}$','$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$','$\it{U=7.5}$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(DOST[i], Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file)
    filenames.close()
    DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'Utotal',DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])

    # Interacting impurity DOS for different magnitudes of correlation strength by varying the flat hybridization
    input=[{"N" : 200000, "poles" : 4, "Gamma" : 0.15, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Gamma" : 0.30, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Gamma" : 0.45, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Gamma" : 0.60, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Gamma" : 0.75, "Ed" : -3/2, "ctype" : 'n'}]
    filenames,labelnames=tqdm(['cN4p_15G','cN4p_3G','cN4p_45G','cN4p_6G','cN4p_75G'],position=0,leave=False,desc='No. SAIM DED Gamma sims'),['$\Gamma\it{=0.15}$','$\\Gamma\it{=0.30}$','$\\Gamma\it{=0.45}$','$\\Gamma\it{=0.60}$','$\\Gamma\it{=0.75}$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(DOST[i], Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file)
    filenames.close()
    DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'Gtotal',DEDlib.Lorentzian(omega,0.3,4,-3/2,3/2)[0])

    # Interacting single impurity DOS for different functions of η
    input=[{"N" : 200000, "poles" : 4, "etaco" : [0.01,1e-39], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.02,1e-39], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.04,1e-39], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.00,0.001], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.00,0.01], "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "etaco" : [0.00,0.1], "Ed" : -3/2, "ctype" : 'n'}]
    filenames,labelnames=tqdm(['cN4p_01+1e-39eta','cN4p_02+1e-39eta','cN4p_04+1e-39eta','cN4p_00+0.001eta','cN4p_00+0.01eta','cN4p_00+0.1eta'],position=0,leave=False,desc='No. SAIM DED eta sims'),['$\\eta=0.01|\\omega|$','$\\eta=0.02|\\omega|$','$\\eta=0.04|\\omega|$','$\\eta=0.001$','$\\eta=0.010$','$\\eta=0.100$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(DOST[i], Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file)
    filenames.close()
    DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'etatotal',Lor)

    # Interacting DOS of asymmetric Anderson impurity model
    input=[{"N" : 200000, "poles" : 4, "Ed" : -1.5, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -1.65, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -1.8, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -2, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -2.5, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4},
    {"N" : 200000, "poles" : 4, "Ed" : -3, "Sigma" : 1.5, "ctype" : 'n', "bound" : 4}]
    filenames=tqdm(['cN4p-1_5Ed','cN4p-1_65Ed','cN4p-1_8Ed','cN4p-2Ed','cN4p-2_5Ed','cN4p-3Ed'],position=0,leave=False,desc='No. ASAIM DED sims')
    for i,file in enumerate(filenames):
        DOST,labelnames,nd,pbar=np.zeros((8,1001),dtype = 'float'),np.chararray(8, itemsize=23),np.zeros(8,dtype = 'float'),trange(8,position=1,leave=False,desc='Self-consistence iteration')
        for j in pbar:
            (nd[j], NewSigma, DOST[j], Lor, omega, selectpT, selectpcT),labelnames[j]=DEDlib.main(**input[i],posb=2),'$\\rho,\\Sigma_0=%.3f$'%input[i]['Sigma']
            DEDlib.DOSplot(DOST[j], Lor, omega,file+'%.16fSigma'%input[i]['Sigma'],'$\\rho,\\Sigma_0=%.3f$'%input[i]['Sigma'])
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[j],file+'%.16fSigma'%input[i]['Sigma'])
            if np.isclose(input[i]['Sigma'],np.real(NewSigma[500]),rtol=5e-4, atol=1e-06): break
            elif j<len(DOST)-1: input[i]['Sigma']=np.real(NewSigma[500])
        pbar.close()
        np.savetxt(file+'%.16fSigma'%input[i]['Sigma']+'nd',nd,delimiter='\t', newline='\n')
        DEDlib.DOSmultiplot(omega,np.tile(omega, (j+1,1)),DOST[~np.all(DOST == 0, axis=1)],np.tile(len(omega), j+1),labelnames.astype(str),'Asymtotal'+file,DEDlib.Lorentzian(omega,0.3,4,input[i]['Ed'],3/2)[0])
    filenames.close()
    
    #Interacting graphene impurity DOS and Interacting graphene nanoribbon center/edge DOS of Anderson impurity model
    input=tqdm([[{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8, "eigsel" : False}],
            [{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8, "eigsel" : True}]],position=0,leave=False,desc='No. selection type sims')
    for l,inp in enumerate(input):
        radius,colorbnd,ip,nd,selecm=tqdm([1.5,2.3,3.1,4.042,5.1],[7,19,37,61,91],[3,9,18,30,45],position=1,leave=False,desc='No. Graphene circular NR SAIM DED sims'),np.zeros((5,4),dtype = 'float'),['','eigval']
        for j,r in enumerate(radius):
            filenames,labelnames=tqdm(['GrapheneCirc'+str(r)+'r1_5U','GrapheneCirc'+str(r)+'r3U','GrapheneCirc'+str(r)+'r4_5U','GrapheneCirc'+str(r)+'r6U'],position=2,leave=False,desc='No. U variation sims'),['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
            DOST=np.zeros((len(filenames),4001),dtype = 'float')
            psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(ip[j],DEDlib.Graphenecirclestruct(r,1),colorbnd[j],'GrapheneCirc'+str(r)+'r')
            for i,file in enumerate(filenames):
                if j==1: inp[i]['ctype'],inp[i]['Ed']='dn','AS'
                else: inp[i]['ctype'],inp[i]['Ed']='n',-inp[i]['U']/2
                nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp[i],posb=3)
                DEDlib.DOSplot(DOST[i], nonintrho, omega,file+selecm[l],labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file+selecm[l])
            filenames.close()
            DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneCirc'+str(r)+'r'+selecm[l],nonintrho,log=True)
        radius.close()
        np.savetxt('GrapheneCirc'+selecm[l]+'nd',nd,delimiter='\t', newline='\n')
        posimp,func,args,colorbnd,structname,nd=tqdm([[85,248],[74,76]],position=1,leave=False,desc='No. Graphene A/Z NR SAIM DED sims'),[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((2,4),dtype = 'float')
        for k,pos in enumerate(posimp):
            posb=tqdm(pos,position=2,leave=False,desc='No. position variation sims')
            for j,imp in enumerate(posb):
                filenames,labelnames=tqdm(['GrapheneNR'+structname[k]+str(imp)+'pos1_5U','GrapheneNR'+structname[k]+str(imp)+'pos3U','GrapheneNR'+structname[k]+str(imp)+'pos4_5U','GrapheneNR'+structname[k]+str(imp)+'pos6U'],position=3,leave=False,desc='No. U variation sims'),['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
                DOST=np.zeros((len(filenames),4001),dtype = 'float')
                psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[k](*args[k]),colorbnd[k],'GrapheneNR'+structname[k]+str(imp)+'pos')
                for i,file in enumerate(filenames):
                    nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp[i],posb=4)
                    DEDlib.DOSplot(DOST[i], nonintrho, omega,file+selecm[l],labelnames[i],log=True)
                    DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file+selecm[l])
                filenames.close()
                DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneNR'+structname[k]+str(imp)+'pos'+selecm[l],nonintrho,log=True)
            posb.close()
            np.savetxt('GrapheneNR'+structname[k]+selecm[l]+'nd',nd,delimiter='\t', newline='\n')
        posimp.close()
    
    #Stop here###############################################################################

    #Temperature dependence interacting impurity DOS with modified constraint for the temperature
    input=[{"N" : 20000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : 'n', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]},
    {"N" : 20000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : ' ', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]},
    {"N" : 20000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : 'nb', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]}]
    filenames,labelnames,conname,pbar=['cN4pT1e-12','cN4pT1e-3','cN4pT1e-2','cN4pT1e-1','cN4pT3e-1','cN4pT1'],['$\it{k_bT= %.3f}$'%0.000,'$\it{k_bT= %.3f}$'%0.001,'$\it{k_bT= %.3f}$'%0.010,'$\it{k_bT= %.3f}$'%0.100,'$\it{k_bT= %.3f}$'%0.300,'$\it{k_bT= %.3f}$'%1.000],['','no','soft'],tqdm(input,position=0,leave=False,desc='No. T-dependent SAIM DED sims')
    DOST=np.zeros((len(input),len(input[0]["Tk"]),1001),dtype = 'complex_')
    for j,inpt in enumerate(pbar):
        nd, _, DOST[j], Lor, omega, selectpT, selectpcT=DEDlib.main(**inpt)
        for i,file in enumerate(filenames):
            DEDlib.DOSplot(DOST[j][i], Lor, omega,conname[j]+file,labelnames[i])
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[j][i],conname[j]+file)
        DEDlib.DOSmultiplot(omega,np.tile(omega, (len(input[0]["Tk"]),1)),DOST[j],np.tile(len(omega), len(input[0]["Tk"])),labelnames,conname[j]+'Ttotal',Lor)
    pbar.close()
    conlabel,fDOS=['$\it{k_bT= %.0f}$, (constr.)'%0,'$\it{k_bT= %.0f}$, (no constr.)'%0,'$\it{k_bT= %.0f}$, (constr.)'%1,'$\it{k_bT= %.0f}$, (no constr.)'%1],[DOST[0][0],DOST[1][0],DOST[0][4],DOST[1][4]]
    DEDlib.DOSmultiplot(omega,np.tile(omega, (4,1)),fDOS,np.tile(len(omega), 4),conlabel,'constrTtotal',Lor)
    
    #Check noconstraint n=5,6
    input=[{"N" : 200000, "poles" : 5, "Ed" : -3/2, "ctype" : ' '},
    {"N" : 20000, "poles" : 6, "Ed" : -3/2, "ctype" : ' '}]
    filenames,labelnames=tqdm(['noconstraintN5p','noconstraintN6p'],position=0,leave=False,desc='No. SAIM DED sims'),['$\\rho_{no constr.},$n=5','$\\rho_{no constr.},$n=6']
    for i,file in enumerate(filenames):
        nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
        DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)
    filenames.close()
    
    #add graphene U sim for all structures, vary t=1 to check impact nanoribbons

    # Need redos of all single impurity with Ed=Ed not sigma dependent to check if this influences symmetry U=6 for example as test
    # Problem was the index of sigmadat for Ed (now corrected)

    ######################################################################### Extra simulation to check if no constraint is correct for n=6 (does not work yet needs more testing)##########################################################################

    ######################################################################### Improve to n=7 ####################################################################

    