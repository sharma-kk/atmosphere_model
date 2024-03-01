import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
# import numpy as np
from firedrake import *
import math
import time
from firedrake.petsc import PETSc

Ny = 256
Nx = 7*256
mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x") #resolution ~ 15 km
mesh.name = "mesh_256"

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

x, y = SpatialCoordinate(mesh)

# define dimensionless parameters
Ro = 0.3 ; Re = 3*10**5 ; B = 0 ; C = 0.02 ; Pe = 3*10**5

# define initial condtions
y0 = 1/14 ; y1 = 13/14
alpha = 1.64
u0_1 = conditional(Or(y <= y0, y >= y1), 0.0, exp(alpha**2/((y - y0)*(y - y1)))*exp(4*alpha**2/(y1 - y0)**2))
u0_2 = 0.0

u0 = project(as_vector([u0_1, u0_2]), V1)
g = project(as_vector([u0_2, -(C/Ro)*(1 + B*y)*u0_1]), V1)

f = interpolate(div(g), V0)

theta0 = TrialFunction(V2)
q = TestFunction(V2)

a = -inner(grad(theta0), grad(q))*dx
L = f*q*dx

theta0 = Function(V2) # potential temperature
nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD) # this is required with Neumann bcs
solve(a == L, theta0, nullspace=nullspace)

# temperature perturbation
theta0_c = 1.0
c0 = 0.01 ; c1 = 4 ;  c2 = 81 ; x_0 = 3.5; y_2 = 0.5
theta0_p = interpolate(c0*cos(math.pi*y/2)*exp(-c1*(x  - x_0)**2)*exp(-c2*(y - y_2)**2), V2)

theta0_f = interpolate(theta0_c + theta0 + theta0_p, V2) # perturbed initial temperature

# Variational formulation
Z = V1*V2

utheta = Function(Z)
u, theta = split(utheta)
v, phi = TestFunctions(Z)
u_ = Function(V1)
theta_ = Function(V2)

u_.assign(u0)
theta_.assign(theta0_f)

perp = lambda arg: as_vector((-arg[1], arg[0]))

Dt =0.005 # 4 minutes

F = ( inner(u-u_,v)
    + Dt*0.5*(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
    + Dt*0.5*(1/Ro)*inner((1 + B*y)*(perp(u) + perp(u_)), v)
    - Dt*0.5*(1/C)*(theta + theta_)* div(v)
    + Dt *0.5 *(1/Re)*inner((nabla_grad(u)+nabla_grad(u_)), nabla_grad(v))
    + (theta - theta_)*phi - Dt*0.5*inner(theta_*u_ + theta*u, grad(phi))
    + Dt*0.5*(1/Pe)*inner((grad(theta)+grad(theta_)),grad(phi)) )*dx

bound_cond = [DirichletBC(Z.sub(0).sub(1), Constant(0.0), (1,2))]

# visulization at t=0
vort_ = interpolate(u_[1].dx(0) - u_[0].dx(1), V0)
vort_.rename("vorticity")
theta_.rename("temperature")
u_.rename("velocity")
energy_ = 0.5*(norm(u_)**2)
KE = []
KE.append(energy_)
PETSc.Sys.Print(f'KE at time t=0: {round(energy_,6)}')

outfile = File("./results/test1.pvd")
outfile.write(u_, theta_, vort_)

# time stepping and visualization at other time steps
t_start = Dt
t_end = Dt*10800 # 30 days

t = Dt
iter_n = 1
freq = 1800 
t_step = freq*Dt # 5 days
current_time = time.strftime("%H:%M:%S", time.localtime())
PETSc.Sys.Print("Local time at the start of simulation:",current_time)
start_time = time.time()
data_file = "./KE_details/test1_data.txt"

while (round(t,4) <= t_end):
    solve(F == 0, utheta, bcs = bound_cond)
    u, theta = utheta.subfunctions
    energy = 0.5*(norm(u)**2)
    KE.append(energy)
    if iter_n%freq == 0:
        if iter_n == freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            PETSc.Sys.Print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/t_step)*execution_time
            PETSc.Sys.Print("Approx. total running time: %.2f minutes:" %total_execution_time)

        PETSc.Sys.Print("t=", round(t,4))
        PETSc.Sys.Print("kinetic energy:", round(KE[-1],6))
        with open(data_file, 'w') as ff:
            print(f'KE_over_time = {KE}', file = ff)
        vort = interpolate(u[1].dx(0) - u[0].dx(1), V0)
        vort.rename("vorticity")
        theta.rename("temperature")
        u.rename("velocity")
        h5_file = "./h5_files/grid_256_fields_at_time_t="+ str(round(t,4)) + ".h5"
        PETSc.Sys.Print(f'Saving the fields at t={round(t,4)} into the .h5 file')
        with CheckpointFile(h5_file, 'w') as afile:
               afile.save_mesh(mesh)
               afile.save_function(u)
               afile.save_function(theta)
        outfile.write(u, theta, vort)
    u_.assign(u)
    theta_.assign(theta)

    t += Dt
    iter_n +=1

PETSc.Sys.Print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))