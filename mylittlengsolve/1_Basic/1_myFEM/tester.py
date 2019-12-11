# Import libraries
from myngspy import *
from ngsolve import *
from netgen.geom2d import unit_square
from netgen import gui

# Generate mesh
mesh_sizes = [0.2, 0.4, 0.6, 0.8]
mesh = Mesh(unit_square.GenerateMesh(maxh=0.8))

V = L2(mesh, order=4)

# Test Functions
u, v = V.TnT()

a = BilinearForm(V)
a += u * v * dx

coeff = exp(-10 * x)

f = LinearForm(V)
f += coeff * v * dx

f.Assemble()
a.Assemble()

gu = GridFunction(V)
gu.vec.data = a.mat.Inverse() * f.vec
Draw(gu - coeff, mesh, 'test')


Q = L2(mesh, order=0)
ba = BitArray(mesh.ne)
ba.Set()
ba[2] = False
Q = Compress(Q, active_dofs=ba)
W = FESpace([V, Q])

(u, p), (v, q) = W.TnT()

p = coeff * p
q = coeff * q

u = u + p
v = v + q

a = BilinearForm(W)
a += u * v*dx

f = LinearForm(W)
f += coeff * v*dx

f.Assemble()
a.Assemble()

gu2 = GridFunction(W)
gu2.vec.data = a.mat.Inverse() * f.vec
u2 = gu2.components[0] + gu2.components[1] * coeff
Draw(u2 - coeff, mesh, 'test2')
