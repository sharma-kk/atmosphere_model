import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
import math
import time

# earlier it was not possible to open files created from parallel run on a single core. Now it is possible !

print("loading high resolution mesh, velocity and temp......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("./h5_files/grid_256_fields_at_time_t=27.0.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh_256")
    u_ = afile.load_function(mesh, "velocity") 
    theta_ = afile.load_function(mesh, "temperature")

print("finished loading! calculating vorticity from velocity",
    time.strftime("%H:%M:%S", time.localtime()))

V0 = FunctionSpace(mesh, "DG", 0)
vort_= interpolate(u_[1].dx(0) - u_[0].dx(1), V0)
vort_.rename("vorticity")

outfile = File("./results/visualizing_fields_from_h5_file.pvd")
outfile.write(u_, theta_, vort_)

print("Simulation completed !",
    time.strftime("%H:%M:%S", time.localtime()))