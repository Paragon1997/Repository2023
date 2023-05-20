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
    input=[{"N":200000,"poles":4,"Gamma": 0.096,"U" : 2,"Sigma" : 1,"Ed":-1,"U2":2,"J":0,"ctype":'n'},
        {"N":20000,"poles":6,"Gamma": 0.096,"U" : 2,"Sigma" : 1,"Ed":-1,"U2":2,"J":0,"ctype":'n'},
        {"N":200000,"poles":4,"Gamma": 0.3,"U" : 3,"Sigma" : 3/2*3,"Ed":-3/2*3,"U2":3,"J":0,"ctype":'n',"bound":4},
        {"N":20000,"poles":6,"Gamma": 0.3,"U" : 3,"Sigma" : 3/2*3,"Ed":-3/2*3,"U2":3,"J":0,"ctype":'n',"bound":4},
        {"N":200000,"poles":4,"Gamma": 0.3,"U" : 3.5,"Sigma" : 4.0,"Ed":-4.0,"U2":2.5,"J":0.5,"ctype":'n',"bound":5},
        {"N":20000,"poles":6,"Gamma": 0.3,"U" : 3.5,"Sigma" : 4.0,"Ed":-4.0,"U2":2.5,"J":0.5,"ctype":'n',"bound":5}]