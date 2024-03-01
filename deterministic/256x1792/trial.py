import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

###################################
# mesh = PeriodicRectangleMesh(10, 5, 2, 1, direction="x")
# V = VectorFunctionSpace(mesh, "CG", 1)
# W = FunctionSpace(mesh, "CG", 1)
# x, y = SpatialCoordinate(mesh)
# coord_func = Function(V).interpolate(SpatialCoordinate(mesh))
# coords = coord_func.dat.data
# print(np.shape(coords ))
# print(coords)
# u = project(as_vector([x,y]), V)
# theta = interpolate(x, W)

# # print("u data",np.shape(u.dat.data), u.dat.data)
# # print("temp data",np.shape(theta.dat.data), theta.dat.data)
# print("u_data", np.array(u.at(coords)))

# outfile = File("./results/trial.pvd")
# outfile.write(u)
##################################
print("loading high resolution mesh, velocity and temp......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("./h5_files/grid_256_fields_at_time_t=27.0.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh_256")
    u_ = afile.load_function(mesh, "velocity") 
    theta_ = afile.load_function(mesh, "temperature")

print("finished loading! calculating vorticity from velocity",
    time.strftime("%H:%M:%S", time.localtime()))

h5_file = "./h5_files/trial.h5"

with CheckpointFile(h5_file, 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(u_)

print("Simulation completed !",
    time.strftime("%H:%M:%S", time.localtime()))

####################################
# V1f = VectorFunctionSpace(f_mesh, "CG", 1)
# V2f = FunctionSpace(f_mesh, "CG", 1)
# V0f = FunctionSpace(f_mesh, "DG", 0)

# V1c = VectorFunctionSpace(c_mesh, "CG", 1)
# V2c = FunctionSpace(c_mesh, "CG", 1)
# V0c = FunctionSpace(c_mesh, "DG", 0)

# coords_func_256 = Function(V1f).interpolate(SpatialCoordinate(f_mesh))
# coords_256 = coords_func_256.dat.data

# print(coords_256.shape)
# print(coords_256)

# uf = Function(V1f)
# thetaf = Function(V2f)

# print("loading high resolution mesh, velocity and temp......",
#     "current time:",time.strftime("%H:%M:%S", time.localtime()))

# with CheckpointFile("./h5_files/grid_256_fields_at_time_t=27.0.h5", 'r') as afile:
#     mesh = afile.load_mesh("mesh_256")
#     u_ = afile.load_function(mesh, "velocity") 
#     theta_ = afile.load_function(mesh, "temperature")

# print("finished loading! Now getting value of loaded functions on coord.... ",
#     time.strftime("%H:%M:%S", time.localtime()))


# uf.assign(0)
# thetaf.assign(0)
# u_vals = np.asarray(u_.at(coords_256, tolerance=1e-10))
# theta_vals = np.asarray(theta_.at(coords_256, tolerance=1e-10))

# uf.dat.data[:] += u_vals
# thetaf.dat.data[:] += theta_vals

# print("values obtained. Now saving them as pvd file.... !","current time:",
#     time.strftime("%H:%M:%S", time.localtime()))

# vortf = interpolate(uf[1].dx(0) - uf[0].dx(1), V0f)
# uf.rename("velocity")
# thetaf.rename("temperature")
# vortf.rename("vorticity")

# outfile1 = File("./results/opening_mulitple_core_result_in_single_core_test2.pvd")
# outfile1.write(uf, thetaf, vortf)

# #####Coarse graining#########
# u_trial = TrialFunction(V1f)
# theta_trial = TrialFunction(V2f)

# u_test = TestFunction(V1f)
# theta_test = TrialFunction(V2f)

# u_avg = Function(V1f)
# theta_avg = Function(V2f)

# c_sqr = Constant(1/(Nx**2))

# a_vel = (c_sqr * inner(nabla_grad(u_trial), nabla_grad(u_test)) + inner(u_trial, u_test)) * dx
# l_vel = inner(uf, u_test) * dx

# a_temp = (c_sqr * inner(grad(theta_trial), grad(theta_test)) + theta_trial*theta_test) * dx
# l_temp = thetaf*theta_test* dx

# # step 1: spatial averaging using Helmholtz operator
# solve(a_vel==l_vel, u_avg)
# solve(a_temp==l_temp, theta_avg)

# print("solved the PDEs (alpha-regularization)",
#     time.strftime("%H:%M:%S", time.localtime()))

# # step 2: projecting on coarse mesh
# uc = Function(V1c)
# thetac = Function(V2c)

# inject(u_avg, uc)
# inject(theta_avg, thetac)

# uc.rename("coarse_vel")
# thetac.rename("coarse_temp")

# h5_file = "./h5_files/coarse_grained_vel_temp_at_t=27.0_mesh_64_c_1_by_64.h5"

# with CheckpointFile(h5_file, 'w') as afile:
#     afile.save_mesh(c_mesh)
#     afile.save_function(uc)
#     afile.save_function(thetac)

# vortc = interpolate(uc[1].dx(0) - uc[0].dx(1), V0c)
# vortc.rename("coarse_vort")

# outfile2 = File("./results/coarse_grained_fields_test2.pvd")
# outfile2.write(uf, thetaf, vortc)
##############################################
