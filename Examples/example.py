from myngspy import *
from ngsolve import *
from netgen.geom2d import unit_square
from netgen import gui

beta = (2, 1)
mesh_size = [1.0, 0.5, 0.25, 0.1250, 0.0625, 0.0313]
mesh_size = [0.0625]

eps = 0.01

# Exact Solution


def p(x): return x + (exp(beta[0]*x/eps)-1)/(1-exp(beta[0]/eps))


def q(y): return y + (exp(beta[1]*y/eps)-1)/(1-exp(beta[1]/eps))


exact = p(x) * q(y)

# Coefifcient
coeff = beta[1] * p(x) + beta[0] * q(y)
b = CoefficientFunction((beta[0], beta[1]))

# Mesh
for size in mesh_size:
    mesh = Mesh(unit_square.GenerateMesh(maxh=size))

    V = L2(mesh, order=2)
    u, v = V.TnT()

    # a = BilinearForm(V)
    # a += SymbolicBFI((eps *  grad(u) * grad(v)) + (b * grad(u) * v))
    # #a += u * v * dx
    #
    # f = LinearForm(V)
    # f += SymbolicLFI(coeff * v)
    # #f += coeff * v * dx
    #
    # f.Assemble()
    # a.Assemble()
    #
    # gu = GridFunction(V)
    # gu.vec.data = a.mat.Inverse(V.FreeDofs(), inverse='sparsecholesky') * f.vec
    # Draw(gu, mesh, 'test')

    Q = L2(mesh, order=0)
    #ba = BitArray(mesh.ne)
    #ba.Set()
    #ba[0] = False
    #Q = Compress(Q, active_dofs=ba)
    W = FESpace([V, Q])

    (u, p), (v, q) = W.TnT()

    p = coeff * p
    q = coeff * q

    grad_u = grad(u) + CoefficientFunction((coeff.Diff(x), coeff.Diff(y))) * p
    grad_v = grad(v) + CoefficientFunction((coeff.Diff(x), coeff.Diff(y))) * q

    u = u + p
    v = v + q

    a = BilinearForm(W)

    a += SymbolicBFI((eps * grad_u * grad_v) + (b * grad_u * v))
    a += u * v * dx

    f = LinearForm(W)
    f += SymbolicLFI(coeff * v)
    # f += coeff * v * dx

    f.Assemble()
    a.Assemble()

    gu2 = GridFunction(W)
    gu2.vec.data = a.mat.Inverse(
        W.FreeDofs(), inverse='sparsecholesky') * f.vec
    u2 = gu2.components[0] + gu2.components[1] * coeff
    # Draw(u2 - coeff, mesh, 'test2')
    Draw(gu2.components[0], mesh, 'test2')
    # print('Mesh Size:', size , "L2 Error:", sqrt (Integrate ((u2 - exact)*(u2- exact), mesh)))
    input('running everything')
