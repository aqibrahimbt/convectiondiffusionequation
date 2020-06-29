# %%
import ngsolve as ng
import netgen.gui
%gui tk
# %%
from ngsolve import *
from netgen.csg import *
import math
from netgen.geom2d import SplineGeometry
from netgen.geom2d import unit_square


hcoarse = 0.08
# %% markdown
# ### Generating Geometry and Mesh
# %% markdown
# ##### Geometry 1
# %%
geom1 = SplineGeometry()
p1 = geom1.AppendPoint (0,0)
p2 = geom1.AppendPoint (1,0)
p3 = geom1.AppendPoint (1,1)
p4 = geom1.AppendPoint (0,1)
# %%
geom1.Append (["line", p1, p2])
geom1.Append (["line", p2, p3])
geom1.Append (["line", p3, p4])
geom1.Append (["line", p4, p1])
# %%
# generate mesh
mesh1 = Mesh(geom1.GenerateMesh(maxh=hcoarse))
# %% markdown
# ##### Geometry 2
# %%
def FirstDomain(geom):
    top_pnts = [ (+2.00,-0.35), (+2.00,+2.00), (-2.00,+2.00), (-2.00,-0.35)]

    top_nums = [geom.AppendPoint(*p) for p in top_pnts]
    lines  = [ (top_nums[0], top_nums[1],  10,   1,  0),
               (top_nums[1], top_nums[2],  10,   1,  0),
               (top_nums[2], top_nums[3],  10,   1,  0),
               (top_nums[3], top_nums[0],  10,   1,  2) ]

    for p0,p1,bn,ml,mr  in  lines:
        geom.Append([ "line", p0, p1 ],
                    bc=bn, leftdomain=ml, rightdomain=mr )
    return (geom, top_nums)

def SecondDomain(geom,topn):
    bot_pnts = [ (-2.00, -2.00),  (+2.00, -2.00) ]
    botn = [geom.AppendPoint(*p) for p in bot_pnts]

    lines  = [ (topn[3], botn[0],    10,   2,   0),
               (botn[0], botn[1],    10,   2,   0),
               (botn[1], topn[0],    10,   2,   0) ]

    for p0,p1,bn,ml,mr  in  lines:
        geom.Append([ "line", p0, p1 ],
                    bc=bn, leftdomain=ml, rightdomain=mr )

    return (geom, botn)

def MakeMesh(hcoarse) :
    geom2 = SplineGeometry()
    geom2, top = FirstDomain(geom2)
    geom2, bot = SecondDomain(geom2,top)
    return  Mesh(geom2.GenerateMesh(maxh=hcoarse))
# %%
# generate mesh
mesh2 = MakeMesh(hcoarse)
# %% markdown
# ##### Geometry 3
# %%
geom3 = SplineGeometry()
geom3.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
mesh3 = Mesh(geom3.GenerateMesh(maxh=hcoarse))
# %% markdown
# ### Solve Convection Diffusion Equation DG Method
# %% markdown
# ##### Configure Finte Element Spaces
# %%
def setup(mesh):
    order=4

    # define finte element space
    fes = L2(mesh, order=order, dgjumps=True)

    ## Create test and trial functions
    u,v = fes.TnT()

    # Compute jumps from one element space to the other
    jump_u = u-u.Other()
    jump_v = v-v.Other()

    n = specialcf.normal(2)

    mean_dudn = 0.5*n * (grad(u)+grad(u.Other()))
    mean_dvdn = 0.5*n * (grad(v)+grad(v.Other()))

    return fes, u, v, jump_u, jump_v, mean_dudn, mean_dvdn, n, order
# %% markdown
# #### Diffusion Equation
# %%
def diffusion(fes, u, v, n, jump_u, jump_v, mean_dudn, mean_dvdn, order):
    alpha = 4
    h = specialcf.mesh_size
    a = BilinearForm(fes)

    diffusion = grad(u)*grad(v) * dx \
    +alpha*order**2/h*jump_u*jump_v * dx(skeleton=True) \
    +(-mean_dudn*jump_v-mean_dvdn*jump_u) * dx(skeleton=True) \
    +alpha*order**2/h*u*v * ds(skeleton=True) \
    +(-n*grad(u)*v-n*grad(v)*u)* ds(skeleton=True)

    return diffusion
# %% markdown
# #### Convection Equation
# %%
def convection(fes, u, v, jump_v, n):
    b = CoefficientFunction((20,1))
    uup = IfPos(b*n, u, u.Other())

    convection = -b * u * grad(v)*dx + b*n*uup*jump_v * dx(skeleton=True)
    return convection
# %% markdown
# #### Convection-Diffusion Equation
# %%
def convection_diffusion(mesh, hcoarse):

    nrefinements = 2
    meshes = []
    Es = []

    for ref in range(nrefinements):

        meshes.append(ng.Mesh(mesh.ngmesh.Copy()))

        mesh = meshes[-1]
        mesh.ngmesh.Refine()

        fes, u, v, jump_u, jump_v, mean_dudn, mean_dvdn, n, order = setup(mesh)

        ## Convection-diffusion problem
        acd = BilinearForm(fes)

        diff = diffusion(fes, u, v, n, jump_u, jump_v, mean_dudn, mean_dvdn, order)
        convec = convection(fes, u, v, jump_v, n)

        acd += diff + convec

        acd.Assemble()


        f = LinearForm(fes)
        f += SymbolicLFI(1*v)
        f.Assemble()


        gfu = GridFunction(fes)
        gfu.vec.data = acd.mat.Inverse(freedofs=fes.FreeDofs(),inverse="umfpack") * f.vec
        Draw(gfu)
#         print(str(ref) + " " + str(acd.mat.Inverse(freedofs=fes.FreeDofs(),inverse="umfpack") * f.vec))
# %%
convection_diffusion(mesh1, hcoarse)
# %%
convection_diffusion(mesh2, hcoarse)
# %%
convection_diffusion(mesh3, hcoarse)
# %% markdown
# ### Solve Convection Diffusion Equation HDG Method
# %% markdown
# ##### Configure Finite element space & CD
# %%
def setup_hdg(mesh):
    order=4
    V = L2(mesh, order=order)
    F = FacetFESpace(mesh, order=order, dirichlet="bottom|left|right|top")
    fes = FESpace([V,F])
    u,uhat = fes.TrialFunction()
    v,vhat = fes.TestFunction()
    jump_u = u-uhat
    jump_v = v-vhat
    alpha = 4
    condense = True
    h = specialcf.mesh_size
    n = specialcf.normal(mesh.dim)

    a = BilinearForm(fes, condense=condense)
    dS = dx(element_boundary=True)

    a += grad(u)*grad(v)*dx + \
        alpha*order**2/h*jump_u*jump_v*dS + \
        (-grad(u)*n*jump_v - grad(v)*n*jump_u)*dS

    b = CoefficientFunction((20,1))

    uup = IfPos(b*n, u, uhat)

    a += -b * u * grad(v)*dx + b*n*uup*jump_v *dS
    a.Assemble()

    f = LinearForm(fes)
    f += SymbolicLFI(1*v)
    f.Assemble()

    gfu = GridFunction(fes)
    if not condense:
        inv = a.mat.Inverse(fes.FreeDofs(), "umfpack")
        gfu.vec.data = inv * f.vec
    else:
        f.vec.data += a.harmonic_extension_trans * f.vec

        inv = a.mat.Inverse(fes.FreeDofs(True), "umfpack")
        gfu.vec.data = inv * f.vec

        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * f.vec

    Draw (gfu.components[0], mesh, "u-HDG")
# %%
setup_hdg(mesh1)
# %%
setup_hdg(mesh2)
# %%
setup_hdg(mesh3)
# %%
