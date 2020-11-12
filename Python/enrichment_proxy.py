from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve import grad as ngsolvegrad
from ngsolve.comp import ProxyFunction
from ngsolve.webgui import Draw


#### Enrichment Proxy Functions for the DG method
class EnrichmentProxy(CoefficientFunction):
    """
    Provide wrappers for grad/Other and multiplication of enrichment lists for the DG Method.
    """
    def __init__(self, func, enr_list):
        CoefficientFunction.__init__(self,0)
        self.func = func
        self.enr_list = enr_list
        self.grad_list = [CoefficientFunction((coeff.Diff(x), coeff.Diff(y))) for coeff in self.enr_list ]

    def __call__(self):
        return self.func[0] + sum([self.func[i]*self.enr_list[i-1] for i in range(1,len(self.enr_list)+1)])

    def x(self):
        return sum([self.func[i]*self.enr_list[i-1] for i in range(1,len(self.enr_list)+1)])
    
    ## Runs the default without enrichment
    def base(self):
        return self.func[0]
    
    def __mul__(self, other):
        if type(other) == EnrichmentProxy:
            return self() * other()
        else:
            return self() * other
    
    def __rmul__(self,other):
        return self.__mul__(other)    
    
    def __add__(self, other):
        if type(other) == EnrichmentProxy:
            return self() + other()
        else:
            return self() + other
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if type(other) == EnrichmentProxy:
            return self() - other()
        else:
            return self() - other
    
    def __rsub__(self,other):
        return self.__sub__(other)
    
    #gradient of newly enriched approximation spaces
    def grad(self):
        mygrad = ngsolvegrad(self.func[0])
        for i in range(1,len(self.enr_list)+1):
            mygrad += self.func[i] * self.grad_list[i-1]
            mygrad += ngsolvegrad(self.func[i])*self.enr_list[i-1]
        return mygrad 
    
    
    '''define proxy functions for Other'''
    def Other(self):
        return EnrichmentProxy([f.Other() for f in self.func],self.enr_list)



#### Enrichment Proxy Functions for the HDG method
class EnrichmentProxy_VOL(CoefficientFunction):
    """
    Provide wrappers for grad/Other and multiplication of enrichment lists for the HDG method.
    """
    def __init__(self, func, enr_list):
        CoefficientFunction.__init__(self,0)
        self.func = func
        self.enr_list = enr_list
        self.grad_list = [CoefficientFunction((coeff.Diff(x), coeff.Diff(y))) for coeff in self.enr_list ]
    
    def __call__(self):
        return self.func[0] + sum([self.func[2 + 2* i] * self.enr_list[i] for i in range(0,len(self.enr_list))])

    def x(self):
        return sum([self.func[2+2*i]*self.enr_list[i] for i in range(0,len(self.enr_list))])
    
    def base(self):
        return self.func[0]

    def __mul__(self, other):
        if type(other) == EnrichmentProxy_VOL:
            return self() * other()
        else:
            return self() * other
    
    def __rmul__(self, other):
        return self.__mul__(other)    
    
    def __add__(self, other):
        if type(other) == EnrichmentProxy_VOL:
            return self() + other()
        else:
            return self() + other
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if type(other) == EnrichmentProxy_VOL:
            return self() - other()
        else:
            return self() - other
    
    def __rsub__(self,other):
        return self.__sub__(other)
    
    
    '''gradient of newly enriched approximation spaces'''
    def grad(self):
        mygrad = ngsolvegrad(self.func[0])
        for i in range(len(self.enr_list)):
            mygrad += self.func[2+2*i] * self.grad_list[i]
            mygrad += ngsolvegrad(self.func[2+2*i])* self.enr_list[i]
        return mygrad


def grad(q):
    if type(q) == ProxyFunction:
        return ngsolvegrad(q)
    else:
        return q.grad()


#### Facet
class EnrichmentProxy_FAC(CoefficientFunction):
    """
    Provide wrappers for the facets.
    """
    def __init__(self, func, enr_list):
        CoefficientFunction.__init__(self,0)
        self.func = func
        self.enr_list = enr_list
        self.grad_list = [CoefficientFunction((coeff.Diff(x), coeff.Diff(y))) for coeff in self.enr_list ]
    
    
    def __call__(self):
        return self.func[1] + sum([self.func[3 + 2*i]*self.enr_list[i] for i in range(0,len(self.enr_list))])

    def x(self):
        return sum([self.func[3 + 2*i]*self.enr_list[i] for i in range(0,len(self.enr_list))])

    def base(self):
        return self.func[1] 
    
    def __mul__(self, other):
        if type(other) == EnrichmentProxy_FAC:
            return self() * other()
        else:
            return self() * other
    
    def __rmul__(self,other):
        return self.__mul__(other)    
    
    def __add__(self, other):
        if type(other) == EnrichmentProxy_FAC:
            return self() + other()
        else:
            return self() + other
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if type(other) == EnrichmentProxy_FAC:
            return self() - other()
        else:
            return self() - other
    
    def __rsub__(self,other):
        return self.__sub__(other)


'''Mark elements in the mesh to be enriched'''
def mark_elements(mesh, enr_indicator, mesh_size):
    ba = BitArray(mesh.ne)        
    ba.Clear()
    for el in mesh.Elements():
        for v in el.vertices:
            if (enr_indicator(mesh[v].point[0],mesh[v].point[1],mesh_size)):
                ba[el.nr] = True
    return ba

    
'''Mark elements in the mesh to be enriched'''
def mark_dofs(Q, mesh, enr_indicator, mesh_size):
    ba_el = mark_elements(mesh, enr_indicator, mesh_size)
    ba = BitArray(Q.ndof)        
    ba.Clear()
    for el in Q.Elements():
        if ba_el[el.nr]:
            for dof in el.dofs:
                ba[dof] = True
    Qx = Compress(Q, active_dofs=ba)     
    return Qx


'''Mark elements boundaries (facets) in the mesh to be enriched'''
def mark_element_bnd(Q, QF, mesh, enr_indicator, mesh_size):
    ba = BitArray(Q.ndof)        
    ba.Clear()
    for el in Q.Elements():
        mark = False
        for v in el.vertices:
            if (enr_indicator(mesh[v].point[0],mesh[v].point[1],mesh_size)):
                mark = True
        for dof in el.dofs:
            ba[dof] = mark

    gfF = GridFunction(QF)     
    gfF.vec[:] = 0
    for el in QF.Elements():
        if ba[el.nr]:
            for dof in el.dofs:
                gfF.vec[dof] += 1
    ba_F = BitArray(QF.ndof)
    ba_F.Clear()
    for i in range(QF.ndof):
        if gfF.vec[i] > 0.5:
            ba_F[i] = True
    QFx = Compress(QF, active_dofs=ba_F) 
    return QFx
