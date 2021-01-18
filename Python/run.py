# netgen
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve import grad as ngsolvegrad
from ngsolve.comp import ProxyFunction
from ngsolve.webgui import Draw
from ngsolve import ngsglobals
ngsglobals.msg_level = 0
#from ngsolve.webgui import Draw as WebGuiDraw
#import netgen.gui

# packages
import numpy as np
from scipy import linalg
import pandas as pd

# modules
from enrichment_proxy import *
from convection_diffusion import *
from utils import *


### Problem configuration
beta = (2,0.001)
eps = 0.01
p = lambda x: x + (exp(beta[0]*(x-1)/eps)-exp(-beta[0]/eps))/(exp(-beta[0]/eps)-1)
q = lambda y: y + (exp(beta[1]*(y-1)/eps)-exp(-beta[1]/eps))/(exp(-beta[1]/eps)-1)

exact = p(x) * q(y)
coeff =  beta[1] * p(x) +  beta[0] * q(y)

config = {
    'order': range(1, 5),
    'beta': (beta[0],beta[1]),
    'mesh_size': np.logspace(0,-1,num=20),
    'epsilon': 0.01,
    'exact': exact,
    'coeff': coeff,
    'alpha': [5],
    'bonus_int_order' : [20],
    'enrich_functions':[p(x), q(y)],
    'enrich_domain_ind':[lambda x,y,h: x > 1 - h, lambda x,y,h: y > 1 - h],
}


if __name__ == "__main__":

    # with enrichment
    CT = Convection_Diffusion(config)
    edg_table = CT._solveEDG()
    #ehdg_table = CT._solveEHDG()
    
    
    # # # without enrichment
    dict = {'enrich_functions':[]}
    config.update(dict)
    CT = Convection_Diffusion(config)
    dg_table = CT._solveEDG()
    #hdg_table = CT._solveEHDG()


    # # # # visualizations
    dg = pd.concat([dg_table, edg_table])
    # dg.to_csv("symmetric_results_big.csv")
    #dg.to_csv('dg.csv')
    #hdg = pd.concat([hdg_table, ehdg_table])
    plot_comparison(dg)
    #plot_comparison(hdg)