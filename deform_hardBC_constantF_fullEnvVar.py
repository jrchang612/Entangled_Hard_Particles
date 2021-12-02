from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import odespy
import os
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
from datetime import datetime
import time
from tqdm import tqdm
import h5py
import pylab
import math
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay
import sys
cwd = os.getcwd()
sys.path.insert(1, cwd)
from rs_helper import rock_string_helpers_fullc2 as rs
from rs_helper.rock_string_analysis_helper import *

env_var = os.environ
Nc = int(env_var['Nc'])
Nf = int(env_var['Nf'])
seed = int(env_var['SEED'])
try:
    F = float(env_var['FORCE'])
except:
    F = 15
try:
    random_init = bool(env_var['RANDOMINIT'])
except:
    random_init = True
try:
    mLf = float(env_var['mLf'])
except:
    mLf = 6
try:
    Lx = float(env_var['LXHALF'])
    Ly = float(env_var['LYHALF'])
except:
    Lx = 15
    Ly = 15

UE_ave = 1
KE_ave = 1
note = '_mLf'+str(mLf)+'_deform_hardBC_constantF_'+str(F)+'_seed'+str(seed)
canvas_initial_xy = np.array([[-Lx, -Ly], [Lx, -Ly], [Lx, Ly], [-Lx, Ly]])
CF_sys = rs.Colloid_and_Filament(Nc = Nc, Nf = Nf, 
	canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
    periodic_bc = False, full_repulsion = True, seed = seed)

print('volume fraction = %s' %CF_sys.vol_frac)
counter = 0
# fully relax in initialization phase
while ((KE_ave > 1E-10) or (UE_ave > 1E-8)) and ((counter < 50) or (percent_changes > 0.01)):
    try:
        t0 = CF_sys.Time[-1]
        CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
    except:
        t0 = 0
    tic()
    CF_sys.simulate(Tf = 2000, t0 = t0, method = 'bdf', save = True, Npts = 40, note = note, use_odespy = True, 
        path = '/scratch/users/jrc612')
    toc()

    U_tot_arr = []
    KE_arr = []
    for i, t in enumerate(CF_sys.Time):
        CF_sys.change_r_expanded(CF_sys.R[:, i].flatten())
        result = CF_sys.compute_potential_energy(CF_sys.R[:, i])
        U_tot_arr.append(result['U_colloid_average'])
        KE_arr.append(CF_sys._KE(CF_sys.R[:, i], CF_sys.Time[i])['KE_colloid_ave'])
    percent_changes = (UE_ave - np.min(U_tot_arr))/UE_ave + (KE_ave - np.min(KE_arr))/KE_ave

    UE_ave = np.min(U_tot_arr)
    KE_ave = np.min(KE_arr)
    print('time = %s' %t0)
    counter += 1
    print('counter = %s' %counter)

# shrinkage
shrinkage = 1
while shrinkage > 0.500:
    CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
    Fx, Fy = CF_sys.canvas.find_force_on_walls(CF_sys.r[0:CF_sys.Nc].reshape((CF_sys.Nc,1)), 
                                                    CF_sys.r[CF_sys.Nc:].reshape((CF_sys.Nc,1)), 
                                                    CF_sys.radius.reshape((CF_sys.Nc,1)))
    Fx = (np.sum(np.abs(Fx)))/2
    if F-Fx > 0:
        dx = (F - Fx)/(CF_sys.eta*6*np.pi*Ly)
        shrinkage -= dx/Lx
        canvas_new = np.copy(canvas_initial_xy).astype(float)
        canvas_center = np.mean(CF_sys.canvas.xy, axis = 0)
        canvas_new[:,0] = (canvas_new[:,0] - canvas_center[0])*shrinkage + canvas_center[0]
        canvas_new[:,1] = (canvas_new[:,1] - canvas_center[1])/shrinkage + canvas_center[1]
        CF_sys.change_canvas(canvas_new)
    else:
        pass
    t0 = CF_sys.Time[-1]
    tic()
    CF_sys.simulate(Tf = 200, t0 = t0, method = 'bdf', save = True, Npts = 4, note = note, use_odespy = True, 
        path = '/scratch/users/jrc612')
    toc()
    print('shrinkage = %s' %shrinkage)

# held in contracted state for awhile
print('holding in contracted state')
t0 = CF_sys.Time[-1]
CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
tic()
CF_sys.simulate(Tf = 2000, t0 = t0, method = 'bdf', save = True, Npts = 4, note = note, use_odespy = True, 
    path = '/scratch/users/jrc612')
toc()

# relaxation
for shrinkage in np.arange(0.505, 1.005, 0.005):
    canvas_new = np.copy(canvas_initial_xy).astype(float)
    canvas_center = np.mean(CF_sys.canvas.xy, axis = 0)
    canvas_new[:,0] = (canvas_new[:,0] - canvas_center[0])*shrinkage + canvas_center[0]
    canvas_new[:,1] = (canvas_new[:,1] - canvas_center[1])/shrinkage + canvas_center[1]
    CF_sys.change_canvas(canvas_new)
    t0 = CF_sys.Time[-1]
    CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
    tic()
    CF_sys.simulate(Tf = 2000, t0 = t0, method = 'bdf', save = True, Npts = 4, note = note, use_odespy = True, 
        path = '/scratch/users/jrc612')
    toc()
    print('shrinkage = %s' %shrinkage)

# held in final state for awhile
print('holding in relaxed state')
for j in range(5):
    t0 = CF_sys.Time[-1]
    CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
    tic()
    CF_sys.simulate(Tf = 2000, t0 = t0, method = 'bdf', save = True, Npts = 4, note = note, use_odespy = True, 
        path = '/scratch/users/jrc612')
    toc()

"""### If Continue simulation ###
folder_name = '2021-04-26/SimResults_Nc_120_Np_10_Nf_80_volfrac_0.62_filfrac_0.23_solver_bdf_jamming_periodic_fullRep_odespy_pid00/'
copy_number = 193
CF_sys.load_data(file = folder_name + 'SimResults_{0:03d}.hdf5'.format(copy_number))
CF_sys.r_expanded = CF_sys.R[:, -1].flatten()
### If Continue simulation ###"""

os.system('say "your program has finished"')
