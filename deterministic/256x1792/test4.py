import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time
from firedrake.petsc import PETSc

PETSc.Sys.Print("loading high resolution mesh, velocity and temp......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("./h5_files/grid_256_fields_at_time_t=27.0.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh_256")
    u0 = afile.load_function(mesh, "velocity") 
    theta0 = afile.load_function(mesh, "temperature")

PETSc.Sys.Print("finished loading!",
    time.strftime("%H:%M:%S", time.localtime()))

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)

# define dimensionless parameters
Ro = 0.3 ; Re = 3*10**5 ; B = 0 ; C = 0.02 ; Pe = 3*10**5

# Variational formulation
Z = V1*V2

utheta = Function(Z)
u, theta = split(utheta)
v, phi = TestFunctions(Z)
u_ = Function(V1)
theta_ = Function(V2)

u_.assign(u0)
theta_.assign(theta0)

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

# time stepping and visualization at other time steps
t_start = 27.0 + Dt # 15 days
t_end = Dt*10800 # 30 days

t = 27.0 + Dt
iter_n = 1
freq = 120
t_step = freq*Dt # 8 hours
current_time = time.strftime("%H:%M:%S", time.localtime())
PETSc.Sys.Print("Local time at the start of simulation:",current_time)
start_time = time.time()

while (round(t,4) <= t_end):
    solve(F == 0, utheta, bcs = bound_cond)
    u, theta = utheta.subfunctions
    if iter_n%freq == 0:
        if iter_n == freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            PETSc.Sys.Print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/t_step)*execution_time
            PETSc.Sys.Print("Approx. total running time: %.2f minutes:" %total_execution_time)

        PETSc.Sys.Print("t=", round(t,4))
        theta.rename("temperature")
        u.rename("velocity")
        h5_file = "./h5_files/grid_256_fields_at_time_t="+ str(round(t,4)) + ".h5"
        PETSc.Sys.Print(f'Saving the fields at t={round(t,4)} into the .h5 file')
        with CheckpointFile(h5_file, 'w') as afile:
               afile.save_mesh(mesh)
               afile.save_function(u)
               afile.save_function(theta)
    u_.assign(u)
    theta_.assign(theta)

    t += Dt
    iter_n +=1
    
PETSc.Sys.Print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))