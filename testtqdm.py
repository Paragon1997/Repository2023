from tqdm.auto import tqdm
import DEDlib
input=[{"N" : 2000, "poles" : 2, "Ed" : -3/2, "ctype" : 'n'},
{"N" : 2000, "poles" : 3, "Ed" : -3/2, "ctype" : 'n'},
{"N" : 2000, "poles" : 4, "Ed" : -3/2, "ctype" : 'n'},
{"N" : 2000, "poles" : 5, "Ed" : -3/2, "ctype" : 'n'}]
filenamespbar,labelnames=tqdm(['constraintN2p','constraintN3p','constraintN4p','constraintN5p'],position=0,leave=False,desc='No. SAIM DED sims'),['$\\rho_{constr.},N,$n=2','$\\rho_{constr.},N,$n=3','$\\rho_{constr.},N,$n=4','$\\rho_{constr.},N,$n=5']
#for i,file in enumerate(filenamespbar):
#    nd, _, fDOS, Lor, omega, selectpT, selectpcT=DEDlib.main(**input[i])
#filenamespbar.close()
print(len(filenamespbar)==4)