from netgen import gui
from netgen.geom2d import unit_square
from ngsolve import *
from myngspy import *

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

ba = BitArray(mesh.ne)
ba.Clear()
ba[1]=True
fes = MyFESpace2(mesh, ba, {"secondorder" : False})
print(fes.ndof)
u = GridFunction(fes,"shapes")

Draw(u,mesh,"u",draw_surf=True)
input("")
# we can use the additionally exported function here
for i in range(fes.ndof):
    print("Draw basis function ", i)
    u.vec[:] = 0
    u.vec[i] = 1
    Redraw()
    input("press key to draw next shape function")
