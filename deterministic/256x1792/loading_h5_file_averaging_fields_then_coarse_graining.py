import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

print("loading high resolution mesh, velocity and temp......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("./h5_files/grid_256_fields_at_time_t=27.0.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh_256")
    u_ = afile.load_function(mesh, "velocity") 
    theta_ = afile.load_function(mesh, "temperature")

print("finished loading! Now getting value of loaded functions on coord.... ",
    time.strftime("%H:%M:%S", time.localtime()))

V1f = VectorFunctionSpace(mesh, "CG", 1)
V2f = FunctionSpace(mesh, "CG", 1)
V0f = FunctionSpace(mesh, "DG", 0)

# vort_= interpolate(u_[1].dx(0) - u_[0].dx(1), V0f)
# vort_.rename("vorticity")

print("Coarse graining.........",
    time.strftime("%H:%M:%S", time.localtime()))

#####Averaging and Coarse graining#########
u_trial = TrialFunction(V1f)
theta_trial = TrialFunction(V2f)

u_test = TestFunction(V1f)
theta_test = TestFunction(V2f)

u_avg = Function(V1f)
theta_avg = Function(V2f)

c_sqr = Constant(1/(64**2)) # averaging solution within box of size 1/64x1/64

a_vel = (c_sqr * inner(nabla_grad(u_trial), nabla_grad(u_test)) + inner(u_trial, u_test)) * dx
l_vel = inner(u_, u_test) * dx

a_temp = (c_sqr * inner(grad(theta_trial), grad(theta_test)) + theta_trial*theta_test) * dx
l_temp = theta_*theta_test* dx

bound_cond = [DirichletBC(V1f.sub(1), Constant(0.0), (1,2))] # making sure that n.v is zero after coarse graining

# step 1: spatial averaging using Helmholtz operator
solve(a_vel==l_vel, u_avg, bcs = bound_cond)
solve(a_temp==l_temp, theta_avg)

print("solved the PDEs (alpha-regularization)",
    time.strftime("%H:%M:%S", time.localtime()))

u_avg.rename("avg_vel")
theta_avg.rename("avg_temp")

vort_avg= interpolate(u_avg[1].dx(0) - u_avg[0].dx(1), V0f)
vort_avg.rename("avg_vort")

outfile1 = File("./results/averaged_fields.pvd")
outfile1.write(u_, theta_, u_avg, theta_avg, vort_avg)

# projecting on coarse grid
Nx = 7*64
Ny = 64
c_mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")
c_mesh.name = "coarse_mesh"

V1c = VectorFunctionSpace(c_mesh, "CG", 1)
V2c = FunctionSpace(c_mesh, "CG", 1)
V0c = FunctionSpace(c_mesh, "DG", 0)

coords_func_coarse = Function(V1c).interpolate(SpatialCoordinate(c_mesh))
coords_coarse = coords_func_coarse.dat.data

uc = Function(V1c)
thetac = Function(V2c)

print("retrieving velocity data........",
    time.strftime("%H:%M:%S", time.localtime()))
uc.assign(0)
u_avg_vals = np.array(u_avg.at(coords_coarse, tolerance=1e-10))
uc.dat.data[:] += u_avg_vals

print("retrieving temperature data........",
    time.strftime("%H:%M:%S", time.localtime()))
thetac.assign(0)
theta_avg_vals = np.array(theta_avg.at(coords_coarse, tolerance=1e-10))
thetac.dat.data[:] += theta_avg_vals

print("calculating vorticity........",
    time.strftime("%H:%M:%S", time.localtime()))

vortc= interpolate(uc[1].dx(0) - uc[0].dx(1), V0c)
vortc.rename("coarse_vort")
uc.rename("coarse_vel")
thetac.rename("coarse_temp")

outfile2 = File("./results/coarse_grained_fields.pvd")
outfile2.write(uc, thetac, vortc)

# saving coarse grained solution into a .h5 file
print("saving coarse grained solution into a .h5 file.....",
    time.strftime("%H:%M:%S", time.localtime()))

# uncomment to save results into .h5 file

# h5_file = "./h5_files/coarse_grained_vel_temp_at_t=27.0_mesh_64_c_1_by_64.h5"

# with CheckpointFile(h5_file, 'w') as afile:
#     afile.save_mesh(c_mesh)
#     afile.save_function(uc)
#     afile.save_function(thetac)

print("Simulation completed !",
    time.strftime("%H:%M:%S", time.localtime()))
