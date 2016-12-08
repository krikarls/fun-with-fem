from dolfin import *
import sys

class StokesProblem(object):

    def __init__(self, argdict, domain,
                 u_D_subdomains, p_N_subdomains):
        self.mu = argdict["mu"]

        self.atol = argdict["atol"]
        self.rtol = argdict["rtol"]
        self.max_its = argdict["max_its"]
        
        self.initial_guess = argdict["initial_guess"]
        self.report_convergence = argdict["report_convergence"]

        self.solver = argdict["solver"]
        self.preconditioner = argdict["preconditioner"]
        
        self.domain = domain
        self.beta = argdict["beta"]

        self.use_P1P1 = argdict["use_P1P1"]
        self.W = self.function_space()

        self.u_dirichlet_subdomains = u_D_subdomains
        self.p_neumann_subdomains = p_N_subdomains


    @staticmethod        
    def default_arg_dict():
        """Returns argdict with all default arguments."""
        
        args = {
            "mu": ("Viscosity of fluid.", 1E-3),
            "atol": ("Absolute tolerance.", 1E-9),
            "rtol": ("Relative tolerance.", 1E-9),
            "initial_guess": ("Initial iterative solver guess.", None),
            "report_convergence": ("Print Krylov solver output.", True),
            "preconditioner": ("Preconditioner.", "hypre_amg"),
            "solver": ("Solver.", "gmres"),
            "beta": ("Stabilization parameter.", 0.2),
            "max_its": ("Maximum number of iterations.", 10000),
            "use_P1P1": ("Use P1-P1 elements and stabilisation.", True),
            
        }
        return {parname: args[parname][1]
                for parname in args}



    def function_space(self):

        elt = self.domain.mesh.ufl_cell()

        if self.use_P1P1:
            print "Creating P1-P1 FunctionSpace."
            P1_u = VectorElement("Lagrange", elt, 1)
            P1_p = FiniteElement("Lagrange", elt, 1)
            P1P1 = MixedElement([P1_u, P1_p])
            return FunctionSpace(self.domain.mesh, P1P1)
        else:
            print "Creating Taylor-Hood FunctionSpace."
            P2 = VectorElement("Lagrange", elt, 2)
            P1 = FiniteElement("Lagrange", elt, 1)
            TH = MixedElement([P2, P1])
            return FunctionSpace(self.domain.mesh, TH)

    def u_dirichlet_bcs(self, bcs={}):
        # dirichlet_subdomains should hold pairs (n: val)
        # where val is the value of the solution at subdomain n
        # as a FEniCS expression

        return [DirichletBC(self.W.sub(0), bcs[subdomain_id],
                            self.domain.subdomain_data, subdomain_id)
                for subdomain_id in bcs]

    def p_neumann_terms(self, ds, dot_v_n, bcs):
        # bcs should hold pairs (n: val) where val is the value of
        # the pressure term at subdomain n as a FEniCS expression
        
        return sum([bcs[subdomain_id] * dot_v_n * ds(subdomain_id)
                    for subdomain_id in bcs])

    
    def modify_neumann_bcs(self, new_neumann):
        for subdomain_id in new_neumann:
            self.p_neumann_subdomains[subdomain_id] = new_neumann[subdomain_id]

    
    def solve(self):
        ds = self.domain.measure("ds")

        print "Creating Test/TrialFunctions."
        u, p = TrialFunctions(self.W)
        v, q = TestFunctions(self.W)
        w = Function(self.W)

        n = FacetNormal(self.domain.mesh)
        beta = self.beta
        h = CellSize(self.domain.mesh)

        mu = self.mu
        if self.use_P1P1:
            print "Using stabilized variational form with beta = {}".format(beta)
            a = (mu*inner(grad(u), grad(v))*dx + div(v)*p*dx + div(u)*q*dx
                 - beta*h*h*inner(grad(p), grad(q))*dx)
            b = (mu*inner(grad(u), grad(v))*dx + p*q/mu*dx
                 + beta*h*h*inner(grad(p), grad(q))*dx)
        else:
            print "Using unstabilized variational form"
            a = mu*inner(grad(u), grad(v))*dx + div(v)*p*dx + div(u)*q*dx
            b = mu*inner(grad(u), grad(v))*dx + p*q/mu*dx


        print "Creating Dirichlet BCs for velocity."
        dirichlet_bcs = self.u_dirichlet_bcs(self.u_dirichlet_subdomains)
       
        print "Creating Neumann BCs."
        L = Constant(0)*q*dx + self.p_neumann_terms(ds, dot(v, n),
                                                    self.p_neumann_subdomains)

        print "Assembling matrices."
        (A, bb) = assemble_system(a, L, dirichlet_bcs)
        (P, _) = assemble_system(b, L, dirichlet_bcs)

        solver = PETScKrylovSolver(self.solver, self.preconditioner)


        solver.set_operators(A, P)
            
            
            
        kpars = parameters["krylov_solver"]
        kpars["absolute_tolerance"] = self.atol # default: 1E-15
        kpars["relative_tolerance"] = self.rtol # default: 1E-9
        kpars["maximum_iterations"] = self.max_its # default: ?

        if self.report_convergence:
            kpars["report"] = True
            kpars["monitor_convergence"] = True
            
        if self.initial_guess is not None:
            print "Using nonzero initial guess."
            kpars["nonzero_initial_guess"] = True
            w2 = interpolate(self.initial_guess, self.W)
            w.vector()[:] = w2.vector()[:]
            
        solver.update_parameters(kpars)
        
        print "Starting solve."
        solver.ksp().setConvergenceHistory()
        num_its = solver.solve(w.vector(), bb)
        self.residuals = solver.ksp().getConvergenceHistory()
        print "Solving complete in {} iterations.".format(num_its)

        u, p = w.split()
        
        return u, p


    

class ZebrafishDomain(object):
   
    def __init__(self, fn):
        self.filename = fn

    def create_mesh(self):
        if self.filename[-5:] == ".xdmf":
            mesh = Mesh()
            f = XDMFFile(mpi_comm_world(), self.filename)
            f.read(mesh, True) #tries to load partition from file
            return mesh
        else:
            return Mesh(self.filename) 
    
    @property
    def mesh(self):
        try:
            return self._mesh
        except AttributeError:
            print "Starting mesh creation."
            self._mesh = self.create_mesh()
            
            return self._mesh
        
    @property
    def subdomain_data(self):
        try:
            return self._subdomain_data
        except AttributeError:
            print "Partitioning Boundary."
            self._subdomain_data = self.partition_boundary()
            
            return self._subdomain_data


        
        
    def measure(self, measure_type):
        return Measure(measure_type, domain=self.mesh,
                       subdomain_data=self.subdomain_data)

    def partition_boundary(self):
        """Returns a FacetFunction labeling the subdomains with ints."""

        ## boundary splits into:
        # a bunch of vessel openings, each parallel with a coordinate plane
        # everything else: vessel walls


        boundary_parts = FacetFunction("size_t", self.mesh)

        near_eps = 0.4
        
        class VesselWall(SubDomain): 
            def inside(self, x, on_boundary):
                # hard to specify hole boundaries, so mark everything
                # and then re-mark that which is not a hole
                return on_boundary
        VesselWall().mark(boundary_parts, 1)

        class VesselInflow1(SubDomain): 
            def inside(self, x, on_boundary):
                return (near(x[0], 602, eps=near_eps) and on_boundary)

        VesselInflow1().mark(boundary_parts, 2)

        class VesselInflow2(SubDomain): 
            def inside(self, x, on_boundary):
                return (near(x[0], 711, eps=near_eps) and on_boundary)

        VesselInflow2().mark(boundary_parts, 3)


        class VesselOutflow1(SubDomain): 
            def inside(self, x, on_boundary):
                return (near(x[1], 293, eps=near_eps)
                        and (x[0] < 657) and on_boundary)

        VesselOutflow1().mark(boundary_parts, 4)

                
        class VesselOutflow2(SubDomain): 
            def inside(self, x, on_boundary):
                return (near(x[0], 648, eps=near_eps)
                        and (x[1] > 320) and on_boundary)

        VesselOutflow2().mark(boundary_parts, 5)


        class VesselOutflow3(SubDomain): 
            def inside(self, x, on_boundary):
                return (near(x[0], 708, eps=near_eps)
                        and (x[1] > 320) and on_boundary)

        VesselOutflow3().mark(boundary_parts, 6)


        return boundary_parts

    


zero = Constant((0, 0, 0))
ppx = 9.375E-8                                # pressure gradient in uPa/um
p_top = ppx * 192
p_mid = ppx * 105
p_bot = ppx * 0
mu = 3.5E-9                                   # dynamic viscosity (uPa * s)

mesh_fn = sys.argv[1]                         # try domains/small_cutout

p_argdict = StokesProblem.default_arg_dict()

domain = ZebrafishDomain(mesh_fn)

p_argdict.update(
    {
        "solver": "gmres",
        "preconditioner": "hypre_amg",
        "use_P1P1": False,
        "mu": mu
    }
)


initial_guess = Expression(("0", "0", "0", "x[1] * dp"), dp=ppx, degree=1)
p_argdict["initial_guess"]=initial_guess      # give the solver a good start


## BCs
dirichlet_bcs = {1: zero}                     # noslip BCs on walls
neumann_bcs = {                               # set pressure on inflows/outflows
    2: Constant(p_bot), 3: Constant(p_bot),
    4: Constant(p_mid),
    5: Constant(p_top), 6: Constant(p_top)
}  


problem = StokesProblem(p_argdict, domain,
                        dirichlet_bcs, neumann_bcs)

u, p = problem.solve()

f = File("zebra_u.xdmf")
f << u

h = File("zebra_p.xdmf")
h << p
