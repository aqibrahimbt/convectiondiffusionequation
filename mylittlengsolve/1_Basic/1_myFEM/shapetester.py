from myngspy import *
from ngsolve import *
from netgen.geom2d import unit_square
from netgen import gui
import faulthandler
faulthandler.enable()

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
#
ba = BitArray(mesh.ne)
ba.Clear()

## Specify the element
ba[20] = True
print(ba)
fes = Cust(mesh, ba, {"secondorder": False})
print("degrees of freedom " + str(fes.ndof))
u = GridFunction(fes, "shapes")

Draw(u, mesh, "u", draw_surf=True)
input("")
# we can use the additionally exported function here
for i in range(fes.ndof):
    print("Draw basis function ", i)
    u.vec[:] = 0
    u.vec[i] = 1
    Redraw()
    input("press key to draw next shape function")
