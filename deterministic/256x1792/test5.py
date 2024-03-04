import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

dX = []
delta_x = 7/4
delta_y = 1/4
n = 3
vel_data_truth, temp_data_truth, vort_data_truth = [], [], []

gridpoints = np.array([[delta_x + i * delta_x, delta_y + j * delta_y] for j in range(n) for i in range(n)])

Dt_uc = 0.6 # assumed decorrelated time interval
t_start = 27.0
t_end = 54.1
time_array = np.arange(t_start, t_end, Dt_uc)
t_stamps = np.round(time_array, 2)

print("time_stamps:", time_array)

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

Dt = 0.02 # coarse grid time step

for i in t_stamps:
    
    print('time:', i)
    print("loading fine resolution mesh and velocity.....",
      "current time:",time.strftime("%H:%M:%S", time.localtime()))
    
    with CheckpointFile("./h5_files/grid_256_fields_at_time_t="+str(i)+".h5", 'r') as afile:
        mesh = afile.load_mesh("mesh_256")
        u_ = afile.load_function(mesh, "velocity") 
        theta_ = afile.load_function(mesh, "temperature")
    
    print("finished loading! Now getting value of loaded function on coord.... ",
      time.strftime("%H:%M:%S", time.localtime()))
    
    V1f = VectorFunctionSpace(mesh, "CG", 1)
    V2f = FunctionSpace(mesh, "CG", 1)
    V0f = FunctionSpace(mesh, "DG", 0)

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

    # step 1: spatial averaging using Helmholtz operator
    bound_cond = [DirichletBC(V1f.sub(1), Constant(0.0), (1,2))] # making sure that n.v is zero after coarse graining

    solve(a_vel==l_vel, u_avg, bcs = bound_cond)
    solve(a_temp==l_temp, theta_avg)

    print("solved the PDEs (alpha-regularization)",
        time.strftime("%H:%M:%S", time.localtime()))

    # projecting on coarse grid

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

    vel_data_truth.append(np.array(uc.at(gridpoints, tolerance=1e-10)))
    temp_data_truth.append(np.array(thetac.at(gridpoints, tolerance=1e-10)))
    vort_data_truth.append(np.array(vortc.at(gridpoints, tolerance=1e-10)))
   
    print("calculating (u - u_avg)Dt........",
        time.strftime("%H:%M:%S", time.localtime()))

    dX.append(Dt*(np.array(u_.at(coords_coarse, tolerance=1e-10)) 
                - np.array(uc.at(coords_coarse, tolerance=1e-10))))
    
    print("Calculation done, saving the data into a separate file",
        time.strftime("%H:%M:%S", time.localtime()))
    
    dX_x = np.array(dX)[:,:,0]
    print("shape of dX1_x:", dX_x.shape)

    dX_y = np.array(dX)[:,:,1]
    print("shape of dX1_y:", dX_y.shape)

    data_file_1 = './data_for_xi_calculation/dX_data_t='+str(t_start)+'_to_t=54.0_c=1by64.npz'
    np.savez(data_file_1, dX_x = dX_x, dX_y = dX_y)

    data_file_2 = './data_from_deterministic_run/coarse_grained_vel_temp_vort_data_t='+str(t_start)+'_to_t=54.0_c=1by64.npz'
    np.savez(data_file_2, gridpoints = gridpoints, vel_data_truth = np.array(vel_data_truth), temp_data_truth= np.array(temp_data_truth), vort_data_truth = np.array(vort_data_truth))

print("simulation completed !!!", time.strftime("%H:%M:%S", time.localtime()))