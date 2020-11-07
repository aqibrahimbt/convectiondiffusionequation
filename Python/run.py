## netgen
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve import grad as ngsolvegrad
from ngsolve.comp import ProxyFunction
from ngsolve.webgui import Draw
from ngsolve import ngsglobals
from ngsolve.webgui import Draw as WebGuiDraw
import netgen.gui

## packages
import numpy as np
import sympy 
from scipy import linalg
import pandas as pd
import sys

## modules
from enrichment_proxy import *
from dg import *
from hdg import *
from utils import *


### Problem configuration
beta = (2,0.001)
eps = 0.01
p = lambda x: x + (exp(beta[0]*(x-1)/eps)-exp(-beta[0]/eps))/(exp(-beta[0]/eps)-1)
q = lambda y: y + (exp(beta[1]*(y-1)/eps)-exp(-beta[1]/eps))/(exp(-beta[1]/eps)-1)

exact = p(x) * q(y)
coeff =  beta[1] * p(x) +  beta[0] * q(y)

config = {
    'order': [1, 2, 3, 4],
    'beta': (beta[0],beta[1]),
    'mesh_size': [1.0, 0.5, 0.25, 0.125, 0.0625],
    'epsilon': 0.01,
    'exact': exact,
    'coeff': coeff,
    'alpha': [100],
    'bonus_int_order' : [10],
    'enrich_functions':[p(x), q(y)],
    'enrich_domain_ind':[lambda x,y,h: x > 1 - h/2, lambda x,y,h: y > 1 - h/2]
}


if __name__ == "__main__":
    ngsglobals.msg_level = 0
    CT = Discontinous_Galerkin(config)
    dg_table = CT._solveEDG()
    plot_error(dg_table)

    # CT = Hybrid_Discontinuous_Galerkin(config)
    # hdg_table = CT._solveEHDG()


