from dolfin import *

parameters["krylov_solver"]["relative_tolerance"] = 1.0e-8 
parameters["krylov_solver"]["absolute_tolerance"] = 1.0e-8 
parameters["krylov_solver"]["monitor_convergence"] = False 
parameters["krylov_solver"]["maximum_iterations"] = 10000

mesh = Mesh('../mesh/zebrafish_mesh.xml.gz')    # micrometre (um)

# Mark opening(numbered bottom to top, left to right)

class NoSlip(SubDomain):
    def inside(self,x,on_boundry):
        return on_boundry

class Opening1(SubDomain):              
    def inside(self, x, on_boundry):
        return (x[0] < 602.727) and on_boundry

class Opening2(SubDomain):                 
    def inside(self, x, on_boundry):
        return (x[0] > 710.) and on_boundry

class Opening3(SubDomain):                 
    def inside(self, x, on_boundry):
        return (x[0] < 640.) and (x[1] > 292.) and on_boundry

class Opening4(SubDomain):
    def inside(self, x, on_boundry):
        return (x[1] > 300.) and (x[0] < 648.) and on_boundry

class Opening5(SubDomain):
    def inside(self, x, on_boundry):
        return (x[1] > 300.) and (x[0] > 707.5) and on_boundry

mf = FacetFunction("size_t", mesh)
mf.set_all(6)              #      _______
NoSlip().mark(mf,0)        #     4__   __5
Opening1().mark(mf,1)      #        | | 
Opening2().mark(mf,2)      #      __| |
Opening3().mark(mf,3)      #     3__  | 
Opening4().mark(mf,4)      #        | |
Opening5().mark(mf,5)      #     ___| |____ 
#plot(mf,interactive=True) #    1__________2
                             
# Assign normal mesh function
n = FacetNormal(mesh)
ds = ds[mf]

# Define spaces and functions
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)
W = V*Q
w = Function(W)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
h = CellSize(mesh)   
beta = Constant(0.2)        # stabilization factor
mu = Constant(3.5E-9)

# Pressure
dp = 9.375E-8               # pressure gradient in uPa/um
p1 = Constant(dp*20)
p2 = Constant(0)
p3 = Constant(0)
p4 = Constant(dp*150)
p5 = Constant(0)

# Boundary condition
noslip = DirichletBC(W.sub(0), Constant((0,0,0)), mf, 0)

# Define variational problem
f = Constant((0,0,0))

a = (mu*inner(grad(u), grad(v))*dx + div(v)*p*dx \
    + div(u)*q*dx - beta*h*h*inner(grad(p), grad(q))*dx)

b = (mu*inner(grad(u), grad(v))*dx + p*q/mu*dx \
    + beta*h*h*inner(grad(p), grad(q))*dx)

L =  inner(v + beta*h*h*grad(q), f)*dx \
   + inner(v,p4*n)*ds(4) + inner(v,p5*n)*ds(5) \
   + inner(v,p3*n)*ds(3)  \
   + inner(v,p1*n)*ds(1) + inner(v,p2*n)*ds(2)

# Assemble system
(A, bb) = assemble_system(a, L, noslip)
(P, _) = assemble_system(b, L, noslip)

# Set solver and preconditioner
solver = KrylovSolver("gmres", "hypre_amg")
solver.set_operators(A, P)

U = Function(W)

# Solve
import time
start_time = time.time()

solver.solve(U.vector(), bb)

print ' \n Time used to solve system:', \
       (time.time()-start_time)/60, 'min'

# Get sub-functions
u, p = U.split()

ufile = File("velocity.pvd")
pfile = File("pressure.pvd")
ufile << u
pfile << p

plot(u)
plot(p)
interactive()