from fenics import *

mesh = Mesh('mesh/res32.xdmf')	# mm

""" Since the mesh is in mm pressure units in pascal must be scaled by alpha = (1e6)**(-1)"""
alpha = (1e6)**(-1)

# Mark boundaries
class Neumann_boundary(SubDomain):
	def inside(self, x, on_boundry):
		return on_boundry

mf = FacetFunction("size_t", mesh)
mf.set_all(0)

Neumann_boundary().mark(mf, 1)
ds = ds[mf]

# Continuum mechanics
E = 16*1e3 *alpha
nu = 0.25
mu, lambda_ = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
epsilon = lambda u: sym(grad(u))

p_outside = 133 *alpha
n = FacetNormal(mesh)
f = Constant((0, 0, 0))

V = VectorFunctionSpace(mesh, "Lagrange", 1)

# --------------- Handle Neumann-problem --------------- #
R = FunctionSpace(mesh, 'R', 0)        		 # space for one Lagrange multiplier
M = MixedFunctionSpace([R]*6)          		 # space for all multipliers
W = MixedFunctionSpace([V, M])
u, mus = TrialFunctions(W)
v, nus = TestFunctions(W)

# Establish a basis for the nullspace of RM
e0 = Constant((1, 0, 0))				# translations
e1 = Constant((0, 1, 0))
e2 = Constant((0, 0, 1))

e3 = Expression(('-x[1]', 'x[0]', '0')) # rotations
e4 = Expression(('-x[2]', '0', 'x[0]'))
e5 = Expression(('0', '-x[2]', 'x[1]'))
basis_vectors = [e0, e1, e2, e3, e4, e5]

a = 2*mu*inner(epsilon(u),epsilon(v))*dx + lambda_*inner(div(u),div(v))*dx
L = inner(f, v)*dx + p_outside*inner(n,v)*ds(1)

# Lagrange multipliers contrib to a
for i, e in enumerate(basis_vectors):
	mu = mus[i]
	nu = nus[i]
	a += mu*inner(v, e)*dx + nu*inner(u, e)*dx

# -------------------------------------------------------- #

# Assemble the system
A = PETScMatrix()
b = PETScVector()
assemble_system(a, L, A_tensor=A, b_tensor=b)

# Solve
uh = Function(W)
solver = PETScLUSolver('mumps') # NOTE: we use direct solver for simplicity
solver.set_operator(A)
solver.solve(uh.vector(), b)

# Split displacement and multipliers. Plot
u, ls = uh.split(deepcopy=True) 
plot(u, mode='displacement', title='Neumann_displacement',interactive=True)


