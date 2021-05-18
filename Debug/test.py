import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *


import numpy as np

eps = 10**-3

#b = exp(-6*((x+0.5)*(x+0.5)+y*y))-exp(-6*((x-0.5)*(x-0.5)+y*y))
#b = lambda x, y : 0.5 * (2 / (1 + exp(((-1 / (2 * np.sqrt(5))) + ((4 * x) / np.sqrt(5)) - ((2 * y) / np.sqrt(5))) / np.sqrt(eps))))

#b = lambda x, y : 1/(1 + exp(((-1/(2 * np.sqrt(5)) + (4 * x)/ np.sqrt(5) - (2 * y)/np.sqrt(5))/np.sqrt(eps))))

#b = lambda x, y : 2 * exp( 2 * ((y / np.sqrt(5 * eps)) + (1 / (4 * np.sqrt(5 * eps))))) / ( (exp((4 * x) / np.sqrt(5 * eps))) + exp(2 * ((y / np.sqrt(5 * eps)) + (1 / (4 * np.sqrt(5 * eps))))))
beta = (2, 0.001)

# A = 1
# B = 1
# C = 1
# D = 1
# k = 0.6 + beta[0] / 4 

# p = lambda x : exp((4 * x) / np.sqrt(5 * eps))

# q = lambda y : exp( 2 * ((y / np.sqrt(5 * eps)) + (1 / (4 * np.sqrt(5 * eps)))))

# p = lambda x: exp((beta[0] * x / 2 * eps) * (A * (cos((np.sqrt(beta[0]**2 + 4 * k) * x ) / 2 * eps )) + (B * sin(np.sqrt(-beta[0]**2 + 4 * k) * x / 2 * eps))))

# q = lambda y: exp((beta[1] * y / 2 * eps) * C * exp(np.sqrt(beta[1]**2 + 4 * k) * y/2* eps) + D * exp(- np.sqrt(beta[1]**2 + 4 * k)  * y / 2* eps))

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
eps = 1e-4
ce = sqrt(5*eps)
p =  lambda x: (1-    (  exp(2*(x-0.5)/ce) - 1 ) / (exp(2*(x-0.5)/ce) + 1 ) )
q =  lambda y: (1-    (  exp(2*(y-0.5)/ce) - 1 ) / (exp(2*(y-0.5)/ce) + 1 ) ) 
sol = p(x) * q(y)
Draw(sol,mesh,"prod")
input('')

# orders = 1
# mesh_size = 0.0625
# mesh = Mesh(unit_square.GenerateMesh(maxh=mesh_size))
# fes = L2(mesh, order=orders, dirichlet="bottom|right|left|top")
# gfu = GridFunction(fes)
# Draw(exact, mesh, 'exact')
# input('')