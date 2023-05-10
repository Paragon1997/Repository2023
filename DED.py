import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
from tqdm.auto import tqdm,trange
import time
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import kwant
import scipy
from scipy.optimize import fsolve

import DEDlib

if __name__=='__main__':
    # Comparison of DED spectra for the symmetric Anderson model for several constaints and sites
    input=[{"N":200000,"poles":2,"Ed":-3/2,"ctype":'n'},
    {"N" : 200000, "poles" : 3, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 20000, "poles" : 6, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : ' '},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n%2'}]
    filenames,labelnames=tqdm(['constraintN2p','constraintN3p','constraintN4p','constraintN5p','constraintN6p','noconstraintN4p','constraint%2N4p'],position=0,leave=False,desc='No. SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\\rho_{constr.},N,$n=2','$\\rho_{constr.},N,$n=3','$\\rho_{constr.},N,$n=4','$\\rho_{constr.},N,$n=5','$\\rho_{constr.},N,$n=6','$\\rho_{no constr.},$n=4','$\\rho_{constr.},$$N\\%$2,n=4']
    for i,file in enumerate(filenames):
        nd,_,fDOS,Lor,omega,selectpT,selectpcT,tsim=DEDlib.main(**input[i])
        DEDlib.DOSplot(fDOS,Lor,omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)
    filenames.close()

    # Series of sampled non-interacting DOS for different number of sites compared to the original Lorentzian non-interacting DOS
    input=[{"N" : 200000, "poles" : 2, "U" : 0, "Sigma" : 0, "Ed" : 0, "ctype" : 'n'},
    {"N" : 200000, "poles" : 2, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 3, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 5, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n'},
    {"N" : 200000, "poles" : 4, "U" : 3, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n%2'}]
    filenames,labelnames=tqdm(['Lorentz2p0U','Lorentz2p3U','Lorentz3p3U','Lorentz4p3U','Lorentz5p3U','Lorentz4p3U%2N'],position=0,leave=False,desc='No. sampling Dist. SAIM DED',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\\rho_0,\it{U=0,n=2}$','$\\rho_0,\it{U=3,n=2}$','$\\rho_0,\it{U=3,n=3}$','$\\rho_0,\it{U=3,n=4}$','$\\rho_0,\it{U=3,n=5}$','$\\rho_0,\it{U=3,n=4},N\\%$2']
    DOST=np.zeros((len(filenames),int(input[-2]["N"]*input[-2]["poles"]/200)-1),dtype = 'float')
    omegap=np.zeros((len(filenames),int(input[-2]["N"]*input[-2]["poles"]/200)-1),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, fDOS, Lor, omega, selectpT, selectpcT,tsim=DEDlib.main(**input[i])
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
    filenames,labelnames=tqdm(['cN4p0U','cN4p1_5U','cN4p3U','cN4p4_5U','cN4p6U','cN4p7_5U'],position=0,leave=False,desc='No. SAIM DED U sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\it{U=0.0}$','$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$','$\it{U=7.5}$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT,tsim=DEDlib.main(**input[i])
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
    filenames,labelnames=tqdm(['cN4p_15G','cN4p_3G','cN4p_45G','cN4p_6G','cN4p_75G'],position=0,leave=False,desc='No. SAIM DED Gamma sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\Gamma\it{=0.15}$','$\\Gamma\it{=0.30}$','$\\Gamma\it{=0.45}$','$\\Gamma\it{=0.60}$','$\\Gamma\it{=0.75}$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT,tsim=DEDlib.main(**input[i])
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
    filenames,labelnames=tqdm(['cN4p_01+1e-39eta','cN4p_02+1e-39eta','cN4p_04+1e-39eta','cN4p_00+0.001eta','cN4p_00+0.01eta','cN4p_00+0.1eta'],position=0,leave=False,desc='No. SAIM DED eta sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\\eta=0.01|\\omega|$','$\\eta=0.02|\\omega|$','$\\eta=0.04|\\omega|$','$\\eta=0.001$','$\\eta=0.010$','$\\eta=0.100$']
    DOST=np.zeros((len(filenames),1001),dtype = 'float')
    for i,file in enumerate(filenames):
        nd, _, DOST[i], Lor, omega, selectpT, selectpcT,tsim=DEDlib.main(**input[i])
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
    filenames=tqdm(['cN4p-1_5Ed','cN4p-1_65Ed','cN4p-1_8Ed','cN4p-2Ed','cN4p-2_5Ed','cN4p-3Ed'],position=0,leave=False,desc='No. ASAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i,file in enumerate(filenames):
        DOST,labelnames,nd,pbar=np.zeros((16,1001),dtype = 'float'),np.chararray(16, itemsize=23),np.zeros((16,2),dtype = 'float'),trange(16,position=1,leave=False,desc='Self-consistence iteration',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for j in pbar:
            (nd[j], NewSigma, DOST[j], Lor, omega, selectpT, selectpcT,tsim),labelnames[j]=DEDlib.main(**input[i],posb=2),'$\\rho,\\Sigma_0=%.3f$'%input[i]['Sigma']
            DEDlib.DOSplot(DOST[j], Lor, omega,file+'%.16fSigma'%input[i]['Sigma'],'$\\rho,\\Sigma_0=%.3f$'%input[i]['Sigma'])
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[j],file+'%.16fSigma'%input[i]['Sigma'],NewSigma)
            if np.isclose(input[i]['Sigma'],np.real(NewSigma[int(np.round(len(NewSigma)/2))]),rtol=6e-4, atol=1e-5): break
            input[i]['Sigma']=np.real(NewSigma[int(np.round(len(NewSigma)/2))])
        pbar.close()
        np.savetxt(file+'%.16fSigma'%np.real(NewSigma[int(np.round(len(NewSigma)/2))])+'nd.txt',nd,delimiter='\t', newline='\n')
        DEDlib.DOSmultiplot(omega,np.tile(omega, (j+1,1)),DOST[~np.all(DOST == 0, axis=1)],np.tile(len(omega), j+1),labelnames[:j+1].astype(str),'Asymtotal'+file,DEDlib.Lorentzian(omega,0.3,4,input[i]['Ed'],3/2)[0])
    filenames.close()
    
    #Interacting graphene impurity DOS and Interacting graphene nanoribbon center/edge DOS of Anderson impurity model
    input=tqdm([[{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : False},
            {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8, "eigsel" : False}],
            [{"N" : 200000, "poles" : 4, "U" : 1.5, "Sigma" : 0.75, "Ed" : -1.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 3.0, "Sigma" : 1.5, "Ed" : -3/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 4.5, "Sigma" : 2.25, "Ed" : -4.5/2, "ctype" : 'n', "bound" : 8, "eigsel" : True},
            {"N" : 200000, "poles" : 4, "U" : 6.0, "Sigma" : 3.0, "Ed" : -6/2, "ctype" : 'n', "bound" : 8, "eigsel" : True}]],position=0,leave=False,desc='No. selection type sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for l,inp in enumerate(input):
        radius,colorbnd,ip,nd,selecm=tqdm([1.5,2.3,3.1,4.042,5.1],position=1,leave=False,desc='No. Graphene circular NR SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),[7,19,37,61,91],[3,9,18,30,45],np.zeros((5,4,2),dtype = 'float'),['','eigval']
        for j,r in enumerate(radius):
            filenames,labelnames=tqdm(['GrapheneCirc'+str(r)+'r1_5U','GrapheneCirc'+str(r)+'r3U','GrapheneCirc'+str(r)+'r4_5U','GrapheneCirc'+str(r)+'r6U'],position=2,leave=False,desc='No. U variation sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
            DOST=np.zeros((len(filenames),4001),dtype = 'float')
            psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(ip[j],DEDlib.Graphenecirclestruct(r,1),colorbnd[j],'GrapheneCirc'+str(r)+'r')
            for i,file in enumerate(filenames):
                if j==1 or l==1: inp[i]['Edcalc']='AS'
                else: inp[i]['Edcalc']=''
                nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT,tsim=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp[i],posb=3)
                DEDlib.DOSplot(DOST[i], nonintrho, omega,file+selecm[l],labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file+selecm[l])
            filenames.close()
            DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneCirc'+str(r)+'r'+selecm[l],nonintrho,log=True)
        radius.close()
        np.savetxt('GrapheneCirc'+selecm[l]+'nd.txt',nd,delimiter='\t', newline='\n')
        posimp,func,args,colorbnd,structname,nd=tqdm([[85,248],[74,76]],position=1,leave=False,desc='No. Graphene A/Z NR SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((2,4,2),dtype = 'float')
        for k,pos in enumerate(posimp):
            posb=tqdm(pos,position=2,leave=False,desc='No. position variation sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for j,imp in enumerate(posb):
                filenames,labelnames=tqdm(['GrapheneNR'+structname[k]+str(imp)+'pos1_5U','GrapheneNR'+structname[k]+str(imp)+'pos3U','GrapheneNR'+structname[k]+str(imp)+'pos4_5U','GrapheneNR'+structname[k]+str(imp)+'pos6U'],position=3,leave=False,desc='No. U variation sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\it{U=1.5}$','$\it{U=3.0}$','$\it{U=4.5}$','$\it{U=6.0}$']
                DOST=np.zeros((len(filenames),4001),dtype = 'float')
                psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[k](*args[k]),colorbnd[k],'GrapheneNR'+structname[k]+str(imp)+'pos')
                for i,file in enumerate(filenames):
                    nd[j,i], AvgSigmadat, DOST[i], nonintrho, omega, selectpT, selectpcT,tsim=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp[i],posb=4)
                    DEDlib.DOSplot(DOST[i], nonintrho, omega,file+selecm[l],labelnames[i],log=True)
                    DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],file+selecm[l])
                filenames.close()
                DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneNR'+structname[k]+str(imp)+'pos'+selecm[l],nonintrho,log=True)
            posb.close()
            np.savetxt('GrapheneNR'+structname[k]+selecm[l]+'nd.txt',nd,delimiter='\t', newline='\n')
        posimp.close()
    input.close()

    #Temperature dependence interacting impurity DOS with modified constraint for the temperature
    input=[{"N" : 200000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : 'n', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : ' ', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : 'sn', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "etaco" : [0.02,1e-24], "ctype" : 'ssn', "Tk" : [0.000000000001,0.001,0.01,0.1,0.3,1]}]
    filenames,labelnames,conname,pbar=['cN4pT1e-12','cN4pT1e-3','cN4pT1e-2','cN4pT1e-1','cN4pT3e-1','cN4pT1'],['$\it{k_bT= %.3f}$'%0.000,'$\it{k_bT= %.3f}$'%0.001,'$\it{k_bT= %.3f}$'%0.010,'$\it{k_bT= %.3f}$'%0.100,'$\it{k_bT= %.3f}$'%0.300,'$\it{k_bT= %.3f}$'%1.000],['','no','soft','smartsoft'],tqdm(input,position=0,leave=False,desc='No. T-dependent SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    DOST=np.zeros((len(input),len(input[0]["Tk"]),1001),dtype = 'complex_')
    for j,inpt in enumerate(pbar):
        nd, _, DOST[j], Lor, omega, selectpT, selectpcT,tsim=DEDlib.main(**inpt)
        for i,file in enumerate(filenames):
            DEDlib.DOSplot(DOST[j][i], Lor, omega,conname[j]+file,labelnames[i])
            DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[j][i],conname[j]+file)
        DEDlib.DOSmultiplot(omega,np.tile(omega, (len(input[0]["Tk"]),1)),DOST[j],np.tile(len(omega), len(input[0]["Tk"])),labelnames,conname[j]+'Ttotal',Lor)
        np.savetxt(conname[j]+'Ttotalnd.txt',nd,delimiter='\t', newline='\n')
    pbar.close()
    conlabel,fDOS=['$\it{k_bT= %.0f}$, (constr.)'%0,'$\it{k_bT= %.0f}$, (no constr.)'%0,'$\it{k_bT= %.0f}$, (constr.)'%1,'$\it{k_bT= %.0f}$, (no constr.)'%1],[DOST[0][0],DOST[1][0],DOST[0][5],DOST[1][5]]
    DEDlib.DOSmultiplot(omega,np.tile(omega, (4,1)),fDOS,np.tile(len(omega), 4),conlabel,'constrTtotal',Lor)

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

    #The impurity entropy calculated by DED for different constraints and site quantities
    input=[{"N" : 200000, "poles" : 2},{"N" : 200000, "poles" : 4},{"N" : 20000, "poles" : 6}]
    ctypes,labelnames,S_imp,S_tot,S_bath=tqdm(['n',' ','sn'],position=0,leave=False,desc='No. constraints Entropy DED',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$N$ constr.,n=','no constr.,n=','$P(N,N_0)$ constr.,n='],np.zeros((9,801)),np.zeros((9,801)),np.zeros((9,801))
    for j,c in enumerate(ctypes):
        filenames=tqdm(['ST2p','ST4p','ST6p'],position=1,leave=False,desc='No. poles Entropy DED',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i,file in enumerate(filenames):
            S_imp[3*j+i],S_tot[3*j+i],S_bath[3*j+i],Nfin,Tk,tsim=DEDlib.Entropyimp_main(**input[i],ctype=c,posb=2)
            DEDlib.Entropyplot(Tk,S_imp[3*j+i],labelnames[j]+str(input[i]["poles"]),file+c)
            np.savetxt(file+c+'.txt',np.c_[Tk,S_imp[3*j+i],S_tot[3*j+i],S_bath[3*j+i]],delimiter='\t', newline='\n')
        filenames.close()
    ctypes.close()
    DEDlib.Entropyplot(Tk,S_imp,np.char.add(np.repeat(labelnames,len(input)),np.tile([str(inp["poles"]) for inp in input],len(labelnames))),'STtotal')
    
    #Impurity entropy for different Gamma values of the SAIM
    input={"N" : 2000, "poles" : 6, "ctype" : 'sn'}
    Gamma,filenames,labelnames,S_imp,S_tot,S_bath=[0.2,0.3,0.5,0.9],tqdm(['ST0_2G','ST0_3G','ST0_5G','ST0_9G'],position=0,leave=False,desc='No. Gamma Entropy DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\Gamma\it{=0.2}$','$\Gamma\it{=0.3}$','$\Gamma\it{=0.5}$','$\Gamma\it{=0.9}$'],np.zeros((4,801)),np.zeros((4,801)),np.zeros((4,801))
    for i,file in enumerate(filenames):
        S_imp[i],S_tot[i],S_bath[i],Nfin,Tk,tsim=DEDlib.Entropyimp_main(**input,Gamma=Gamma[i])
        DEDlib.Entropyplot(Tk,S_imp[i],labelnames[i],file)
        np.savetxt(file+'.txt',np.c_[Tk,S_imp[i],S_tot[i],S_bath[i]],delimiter='\t', newline='\n')
    filenames.close()
    DEDlib.Entropyplot(Tk,S_imp,labelnames,'STtotalG')

    #Check noconstraint n=5,6
    input=[{"N" : 200000, "poles" : 5, "Ed" : -3/2, "ctype" : ' '},
    {"N" : 20000, "poles" : 6, "Ed" : -3/2, "ctype" : ' '}]
    filenames,labelnames=tqdm(['noconstraintN5p','noconstraintN6p'],position=0,leave=False,desc='No. SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),['$\\rho_{no constr.},$n=5','$\\rho_{no constr.},$n=6']
    for i,file in enumerate(filenames):
        nd, _, fDOS, Lor, omega, selectpT, selectpcT,tsim=DEDlib.main(**input[i])
        DEDlib.DOSplot(fDOS, Lor, omega,file,labelnames[i])
        DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),fDOS,file)
    filenames.close()

    #Stdev calculator as a function of N
    filename,labelnames,Nstdev,stdev='stdevN4p',['Population $\\rho \it{n=4}$','$\\pm 3\\sigma$','DED \it{n=4}$'],np.logspace(2, 5, num=50, base=10,dtype='int'),np.zeros((50,1001))
    Npbar=tqdm(Nstdev,position=0,leave=False,desc='No. SAIM DED stdev(N) calculations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i,N in enumerate(Npbar):
        pbar,DOST=trange(20,position=1,leave=False,desc='No. SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),np.zeros((20,1001),dtype = 'complex_')
        for j in pbar:
            _, _, DOST[j], _, omega, _, _,tsim=DEDlib.main(N=N,posb=2)
        pbar.close()
        stdev[i]=np.sqrt(np.sum([(DOS-np.mean(DOST,axis=0))**2 for DOS in DOST],axis=0)/(len(DOST)-1))
    Npbar.close()
    stdavg=stdev/np.sqrt(len(DOST))
    DEDlib.stdplot(Nstdev,stdavg,filename,labelnames[2])
    np.savetxt(filename+'.txt',np.insert(stdev,0,Nstdev,axis=1),delimiter='\t', newline='\n')

    #Interacting graphene nanoribbon center DOS of Anderson impurity model for various t values
    input=tqdm([{"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'ssn', "bound" : 8, "eigsel" : False},
    {"N" : 200000, "poles" : 4, "Ed" : -3/2, "ctype" : 'ssn', "bound" : 8, "eigsel" : True}],position=0,leave=False,desc='No. selection type sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    labelnames,selecm=['$\it{t= 0.1}$','$\it{t= 0.5}$','$\it{t= 1.0}$','$\it{t= 1.5}$','$\it{t= 2.0}$'],['','eigval']
    t,func,args,colorbnd,structname,nd=[0.1,0.5,1.0,1.5,2.0],[DEDlib.GrapheneNRarmchairstruct,DEDlib.GrapheneNRzigzagstruct],[(3,12,-2.8867513459481287),(2.5,12,-11.835680518387328,0.5)],[171,147],['armchair','zigzag'],np.zeros((2,5,2),dtype = 'float')
    for l,inp in enumerate(input):
        posimp=tqdm([85,74],position=1,leave=False,desc='No. Graphene A/Z NR SAIM DED sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for j,imp in enumerate(posimp):
            filenames,DOST,nonintrho=tqdm(['cssnt0_1','cssnt0_5','cssnt1','cssnt1_5','cssnt2'],position=2,leave=False,desc='No. t variation sims',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),np.zeros((len(filenames),4001),dtype = 'float'),np.zeros((len(filenames),4001),dtype = 'float')
            for i,file in enumerate(filenames):
                psi,SPG,eig,SPrho0=DEDlib.GrapheneAnalyzer(imp,func[j](*args[j],t=t[i]),colorbnd[j],'GrapheneNR'+structname[j]+str(imp)+'pos')
                nd[j,i], AvgSigmadat, DOST[i], nonintrho[i], omega, selectpT, selectpcT,tsim=DEDlib.Graphene_main(psi,SPG,eig,SPrho0,**inp,posb=3)
                DEDlib.DOSplot(DOST[i], nonintrho[i], omega,'GrapheneNR'+file+structname[j]+selecm[l],labelnames[i],log=True)
                DEDlib.textfileW(omega,np.ravel(selectpT),np.ravel(selectpcT),DOST[i],'GrapheneNR'+file+structname[j]+selecm[l])
            filenames.close()
            DEDlib.DOSmultiplot(omega,np.tile(omega, (len(filenames),1)),DOST,np.tile(len(omega), len(filenames)),labelnames,'GrapheneNRt'+structname[l]+str(imp)+'pos'+selecm[l],nonintrho[2],log=True)
        posimp.close()
        np.savetxt('tGrapheneNR'+selecm[l]+'nd.txt',nd,delimiter='\t', newline='\n')
    input.close()

    #Stop here###############################################################################

    #run TDED with lower offset eta originally 5e-4 now testing for 1e-4

    #Temperature dep. + multi orbitals

    #figure out why u=0 does not give rho0 for r=2.3

    ######################################################################### Extra simulation to check if no constraint is correct for n=6 (does not work yet needs more testing)##########################################################################

    ######################################################################### Improve to n=7 ####################################################################

    