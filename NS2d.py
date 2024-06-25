from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from tqdm import tqdm
#import matplotlib.tri as tri

from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells, compute_closest_entity, create_midpoint_tree

from dolfinx.fem import Constant, Function, functionspace, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological, Expression
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym, grad)
from dolfinx import default_scalar_type
import matplotlib.pyplot as plt

# Evaluate the basis functions at a point on the reference triangle (for bubble forcing)
# The basis functions are for quadratic Lagrange elements
def eval_basis_at_point(point):
    x = point[0]
    y = point[1]
    basis_values = np.zeros(6, dtype=default_scalar_type)
    basis_values[0] = 2*x**2 + 4*x*y - 3*x + 2*y**2 - 3*y + 1
    basis_values[1] = 2*x**2 - x
    basis_values[2] = 2*y**2 - y
    basis_values[3] = 4*x*y
    basis_values[4] = 4*y*(1-x-y)
    basis_values[5] = 4*x*(1-x-y)
    return basis_values

# Find what element a point lies on and find the basis function values at that point
def compute_cell_contributions(V, points):

    cell_candidates = compute_collisions_points(mesh_tree,points)
    midpoint_tree = create_midpoint_tree(mesh, mesh.geometry.dim, cell_candidates.array)
    cells = compute_closest_entity(mesh_tree, midpoint_tree, mesh,points) # Cell containing point
    owning_points = points

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmap
    ref_x = np.zeros((len(cells), mesh.geometry.dim),
                     dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    basis_values = np.zeros((len(ref_x), V.dofmap.dof_layout.num_dofs), dtype=default_scalar_type)
    for i in range (0,len(ref_x)):
        basis_values[i] = eval_basis_at_point(ref_x[i])

    return cells, basis_values*(eta.value/Fr.value**2)

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return (2/Re) * mu * epsilon(u) - p * Identity(len(u))

# Evaluate velocity at a point
def evaluate_velocity(x):
    cell_candidates = compute_collisions_points(mesh_tree, x)
    cells = compute_colliding_cells(mesh, cell_candidates, x)
    return u_n.eval(x, cells.array[0])

# Update bubble locations (Euler stepping)
def update_bubble_locations(bubbleLocations, dt):
    for i in range(len(bubbleLocations)):
        # Evaluate velocity at bubble location
        bubbleLocations[i][0:2] += dt * (evaluate_velocity(bubbleLocations[i]) + [0,1]) # pga^2/(3mu) = 1
    
    # Remove bubbles that are at the surface
    bubbleLocations = bubbleLocations[bubbleLocations[:,1] < 0]

    # Add new bubbles at given rate at pilot locations
    if t % (1/bubbleRate) < dt:
        bubbleLocations = np.concatenate((bubbleLocations, bubblePilotLocations), axis=0)

    return bubbleLocations

def C_Euler_Step(c_n, u_n, dt):
    
    # Variational form of Euler step for CO2 concentration
    F_c = ((c - c_n)/k) * q_c * dx - calc_C_derivative(c_n,u_n)
    a_c = form(lhs(F_c))
    L_c = form(rhs(F_c))
    A_c = assemble_matrix(a_c, bcs=bcC)
    A_c.assemble()
    b_c = create_vector(L_c)
    
    solver_c = PETSc.KSP().create(mesh.comm)
    solver_c.setOperators(A_c)
    # GMRES solver with Jacobi preconditioner
    solver_c.setType(PETSc.KSP.Type.GMRES)
    pc_c = solver_c.getPC()
    pc_c.setType(PETSc.PC.Type.JACOBI)

    with b_c.localForm() as loc:
        loc.set(0)
    assemble_vector(b_c, L_c)
    apply_lifting(b_c, [a_c], [bcC])
    b_c.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_c, bcC)
    solver_c.solve(b_c, c_.vector)
    c_.x.scatter_forward()

    return c_.x.array

# Used to calculate the time derivative of the CO2 concentration from advection-diffusion equation
def calc_C_derivative(c,u): 
    return - D*dot(grad(c),grad(q_c))*dx - dot(u,grad(c))*q_c*dx

# Test whether a point is on the walls, x = 0 or x = L or y = 0
def walls(x):
    return np.logical_or(np.isclose(x[0], 0), np.logical_or(np.isclose(x[0], L), np.isclose(x[1], -H)))
def surface(x):
    return np.isclose(x[1], 0)
def x_Surfaces(x):
    return np.logical_or(np.isclose(x[0], 0),np.isclose(x[0],L))
def y_Surfaces(x):
    return np.logical_or(np.isclose(x[1], -H),np.isclose(x[1],0))
def surfacePressure(x):
    return np.isclose(x[1], 0)
def surfaceCO2(x):
    return np.isclose(x[1], 0)

# Define the mesh
L = 1 
H = 1
#nx = 128 # points in x direction
#ny = 128 # points in y direction
ny = nx
h = L/nx

t = 0 # Starting time
dt = 10e-6 # Time step
T = 1 # Final time
num_steps = int(T/dt) # Number of time steps

# Dimensional parameters
gravity = 980.665 # Gravity (cm/s^2)
density = .1 # Density (g/cm^3)
viscosity = .01 # Viscosity (g/cm s)s
initial_c = 1.0 # Initial CO2 concentration (g/cm^3)
D_c = 1.85e-5  # diffusion coefficient for CO2 in water (cm^2/s)

# Create mesh and function space
# Make mesh with triangular elements
mesh = create_rectangle(MPI.COMM_WORLD, [np.array([0, -H]), np.array([L, 0])], [nx, ny], CellType.triangle)
mesh_tree = bb_tree(mesh, mesh.topology.dim) # Create bounding box tree for collision detection

a = .02 # Bubble radius (cm), 200 micron
bubblePilotLocations = np.array([[L/4, -H + a, 0], [L/2, -H + a, 0], [3*L/4, -H + a, 0]])
bubbleRate = 2 # Rate at which bubbles form (bubbles/second)
bubbleLocations = np.array([[.5, -.5, 0], [.25, -.5, 0], [.75, -.5, 0]], dtype=mesh.geometry.x.dtype) # Initial bubble locations
bubbleLocations = np.array([[.5, -.5, 0]], dtype=mesh.geometry.x.dtype) # Initial bubble locations

mu = Constant(mesh, PETSc.ScalarType(viscosity)) # Viscosity
rho = Constant(mesh, PETSc.ScalarType(density)) # Density
U = Constant(mesh, PETSc.ScalarType((a**2)*density*gravity/(3*viscosity))) # Characteristic velocity
Re = Constant(mesh, PETSc.ScalarType((density*U.value*L/viscosity))) # Reynolds number
Fr = Constant(mesh, PETSc.ScalarType(U.value/np.sqrt(gravity*L))) # Froude number
eta = Constant(mesh, PETSc.ScalarType((4/3)*np.pi*(a**3)/(3*L*L*H))) # Dimensionless bubble forcing coefficient (Should this be 3d or 2d?)
g = Constant(mesh, PETSc.ScalarType((0, 1/(Fr.value**2)))) # Body force
k = Constant(mesh, PETSc.ScalarType(dt/(L/U))) # Time step
D = Constant(mesh, PETSc.ScalarType(D_c/(L*U))) # Dimensionless diffusion coefficient

print("# Characteristic Velocity: ", U.value) # Characteristic velocity
print("# Reynolds Number: ", Re.value) # Reynolds number
print("# Froude Number: " , Fr.value) # Froude number
print("# Eta: ", eta.value) # Dimensionless bubble forcing coefficient
print("# Characteristic Time: ", L/U.value) # Characteristic time
print("# Dimensionless Time Step: ", k.value) # Dimensionless time step
print("# Boundary layer thickness: ", np.sqrt(D_c*H/(2*U.value))) # Boundary layer thickness

# Making function space
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2) # Velocity space
Q = functionspace(mesh, s_cg1) # Pressure space
C = functionspace(mesh, s_cg1) # CO2 concentration space

# Define trial and test functions
# Trial functions are solutions, test functions are used to multiply the weak form
u = TrialFunction(V) # Velocity
v = TestFunction(V)
p = TrialFunction(Q) # Pressure
q = TestFunction(Q)
c = TrialFunction(C) # C02 concentration
q_c = TestFunction(C)

# Create functions

# Velocity functions
u_n = Function(V) # Velocity at previous time step
u_n1 = Function(V) # Velocity at previous time step (n-1)
u_n.name = "u_n" # Name for visualization
U_middle = 0.5 * (u_n + u) # Velocity at intermediate time step
n = FacetNormal(mesh) # Normal vector

# C02 concentration functions
c_n = Function(C) # CO2 concentration at current time step
c_n.interpolate(lambda x: np.full((x.shape[1],), initial_c)) # Initialize to uniform concentration
c_n1 = Function(C) # CO2 concentration at previous time step (n-1)

# Bubble forcing functions
bubbleForceFunction = Function(V)
bubbleForceFunction.name = "bubbleForceFunction"

# Pressure functions
p_n = Function(Q) # Pressure at previous time step
p_n.name = "p_n"

# Boundary Conditions

# No slip boundary conditions for side walls and bottom wall
wall_dofs = locate_dofs_geometrical(V, walls)
u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bc_noslip = dirichletbc(u_noslip, wall_dofs, V)

# No normal flow at the surface, y = 0
surface_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, surface)
surface_dofs_y = locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, surface_facets)
bc_surfaceNormal = dirichletbc(PETSc.ScalarType(0), surface_dofs_y, V.sub(1))

# Surface pressure = 0
surfacePressure_dofs = locate_dofs_geometrical(Q, surfacePressure)
bc_surfacePressure = dirichletbc(PETSc.ScalarType(0), surfacePressure_dofs, Q)

# Surface CO2 concentration = 0
surfaceCO2_dofs = locate_dofs_geometrical(C, surfaceCO2)
bc_surfaceCO2 = dirichletbc(PETSc.ScalarType(0), surfaceCO2_dofs, C)

# no normal flow on any side and surface (y=0)
x_surface_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, x_Surfaces)
surface_dofs_x = locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, x_surface_facets)
bc_x_surfaceNormal = dirichletbc(PETSc.ScalarType(0), surface_dofs_x, V.sub(0))

y_surface_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, y_Surfaces)
surface_dofs_y = locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, y_surface_facets)
bc_y_surfaceNormal = dirichletbc(PETSc.ScalarType(0), surface_dofs_y, V.sub(1))

# Combine boundary conditions
bcu = [bc_noslip, bc_surfaceNormal]
bcFinalu = [bc_x_surfaceNormal, bc_y_surfaceNormal]
bcp = [bc_surfacePressure]
bcC = [bc_surfaceCO2]

# Variational problem fo step 1 (Velocity prediction)
F1 = rho * dot((u - u_n) / k, v) * dx # Left-hand side of the first equation
F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx # Convective term
F1 += inner(sigma(U_middle, p_n), epsilon(v)) * dx # Stress term
F1 -= dot(sigma(U_middle, p_n) * n, v) * ds # Boundary term
L1 = form(-dot(g, v) * dx) # Right-hand side of the first equation
a1 = form(lhs(F1))
A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = create_vector(L1)

# Define variational problem for step 2 (Pressure correction)
u_ = Function(V)
a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / k) * div(u_) * q * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

# Define variational problem for step 3 (Velocity correction)
p_ = Function(Q)
a3 = form(dot(u, v) * dx)
L3 = form(dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
A3 = assemble_matrix(a3, bcs=bcFinalu)
A3.assemble()
b3 = create_vector(L3)

# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
# Biconjugate gradient stabilized method with BoomerAMG preconditioner
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
# Biconjugate gradient stabilized method with BoomerAMG preconditioner
solver2.setType(PETSc.KSP.Type.BCGS)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
# Conjugate gradient method with SOR preconditioner
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)  

# Adams-Bashforth 2nd order method for CO2 concentration
c_ = Function(C)
F_c =((c - c_n)/k) * q_c * dx  - (3/2)*calc_C_derivative(c_n,u_n) + (1/2)*calc_C_derivative(c_n1,u_n1)
a_c = form(lhs(F_c))
L_c = form(rhs(F_c))
A_c = assemble_matrix(a_c, bcs=bcC)
A_c.assemble()
b_c = create_vector(L_c)

# Solver for CO2 concentration
solver_c = PETSc.KSP().create(mesh.comm)
solver_c.setOperators(A_c)
# GMRES solver with Jacobi preconditioner
solver_c.setType(PETSc.KSP.Type.GMRES)
pc_c = solver_c.getPC()
pc_c.setType(PETSc.PC.Type.JACOBI)


# Time-stepping
for i in tqdm(range(num_steps)):
#for i in range(num_steps):
    # Update current time step
    t += dt
    
    #  Step 1: Velocity prediction step
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    b1.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.REVERSE)
    
    # Applying bubble forces to RHS of velocity prediction step
    if mesh.comm.rank != 0:
        bubbleLocations = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
    cells, basis_values = compute_cell_contributions(V, bubbleLocations)        
    for i in range(len(cells)):
        b1.setValues(V.dofmap.cell_dofs(cells[i])*2 + 1, basis_values[i], addv=PETSc.InsertMode.ADD_VALUES)
    b1.assemblyBegin()
    b1.assemblyEnd()
    
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward() 

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    apply_lifting(b3, [a3], [bcFinalu]) # added
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b3, bcFinalu)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()
    
    # Update variable with solution form this time step
    u_n1.x.array[:] = u_n.x.array[:]
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]
    
    # First step is Euler for CO2 concentration
    if (t == dt):
        c_n1.x.array[:] = c_n.x.array[:]
        c_n.x.array[:] = C_Euler_Step(c_n, u_n, dt)
    else:
        # Solve for CO2 concentration with AB-2
        with b_c.localForm() as loc:
            loc.set(0)
        assemble_vector(b_c, L_c)
        apply_lifting(b_c, [a_c], [bcC]) # Same boundary conditions as the pressure field, zero at the surface
        b_c.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_c, bcC)
        solver_c.solve(b_c, c_.vector)
        c_.x.scatter_forward()

        c_n1.x.array[:] = c_n.x.array[:]
        c_n.x.array[:] = c_.x.array[:]

    # Update bubble locations
    bubbleLocations = update_bubble_locations(bubbleLocations, dt)

# Plot the resulting velocity field
plt.figure()
dof_coordinates = V.tabulate_dof_coordinates()
vectors = u_n.vector.array.reshape(-1, 2)
order = np.argsort(dof_coordinates[:,0])
xSorted = np.array(dof_coordinates[:,0])[order]
ySorted = np.array(dof_coordinates[:,1])[order]
uSorted = np.array(vectors[:,0])[order]
vSorted = np.array(vectors[:,1])[order]
plt.quiver(xSorted, ySorted, uSorted, vSorted)
plt.show()
    

b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()

