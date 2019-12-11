import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *
mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

order = 4
fes = L2(mesh, order=order, dgjumps=True)
u, v = fes.TnT()


jump_u = u-u.Other()
jump_v = v-v.Other()
n = specialcf.normal(2)
mean_dudn = 0.5*n * (grad(u) + grad(u.Other()))
mean_dvdn = 0.5*n * (grad(v) + grad(v.Other()))

alpha = 4
h = specialcf.mesh_size
a = BilinearForm(fes)
diffusion = grad(u)*grad(v) * dx \
    + alpha*order**2/h*jump_u*jump_v * dx(skeleton=True) \
    + (-mean_dudn*jump_v-mean_dvdn*jump_u) * dx(skeleton=True) \
    + alpha*order**2/h*u*v * ds(skeleton=True) \
    + (-n*grad(u)*v-n*grad(v)*u) * ds(skeleton=True)

a += diffusion
a.Assemble()
