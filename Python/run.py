# netgen
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve import grad as ngsolvegrad
from ngsolve.comp import ProxyFunction
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

#[lambda x,y,h: x > 1 - h, lambda x,y,h: y > 1 - h]
config = {
    #'order': [1, 2, 3, 4],
    'order': [4],
    'beta': (beta[0],beta[1]),
    'mesh_size': np.logspace(0,-1,num=20),
    'mesh_size': [0.0625],
    #'mesh_size': [0.25, 0.125, 0.0625, 0.0333],
    'epsilon': 0.01,
    'exact': exact,
    'coeff': coeff,
    'bonus_int' : 50,
    'theta': 1e-5,
    'enrich_functions':[p(x), q(y)],
    'enrich_domain_ind':[lambda x,y,h: x > 1 - h, lambda x,y,h: y > 1- h],
}


if __name__ == "__main__":

    # with enrichment
    CT = Convection_Diffusion(config)
    # edg_table, alpha_edg  = CT._solveEDG()
    ehdg_table, alpha_ehdg = CT._solveEHDG()
    
    # without enrichment
    dict = {'enrich_functions': [], 'enrich_domain_ind': []}
    config.update(dict)
    CT = Convection_Diffusion(config)
    #dg_table, alpha_dg = CT._solveEDG()
    hdg_table, alpha_hdg = CT._solveEHDG()

    # # # # # write to files
    # alphas = pd.concat([alpha_edg, alpha_dg])
    # alphas.to_csv('alphas_dg.csv')

    # # # visualizations
    #dg = pd.concat([hdg_table, ehdg_table])
    # # #print(dg.to_latex(index=False)) 
    hdg = pd.concat([hdg_table, ehdg_table])
    plot_error_mesh(hdg) ## Plots with the mesh_size
    # plot_error_dof(hdg_table)