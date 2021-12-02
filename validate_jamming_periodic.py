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

UE_ave = 1
KE_ave = 1
note = '_mLf6_jamming_periodic_fullRep_odespy_pid00'
CF_sys = rs.Colloid_and_Filament(Nc = 120, Nf = 180, 
	canvas_initial_xy = np.array([[-15, -15], [15, -15], [15, 15], [-15, 15]]), random_init = False, 
    periodic_bc = True, full_repulsion = True, mLf = 6)

### If Continue simulation ###
'''
folder_name = '2021-05-10/SimResults_Nc_120_Np_10_Nf_180_volfrac_0.62_filfrac_0.57_solver_bdf_mLf6_jamming_periodic_fullRep_odespy_pid00/'
copy_number = 58
CF_sys.load_data(file = folder_name + 'SimResults_{0:03d}.hdf5'.format(copy_number))
CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
'''
### If Continue simulation ###
print('volume fraction = %s' %CF_sys.vol_frac)

tic()
try:
    t0 = CF_sys.Time[-1]
except:
    t0 = 0
CF_sys.simulate(Tf = 2000, t0 = t0, method = 'bdf', save = True, Npts = 40, note = note, use_odespy = True, pid = 0)
toc()
CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
counter = 0
while (KE_ave > 1E-15)  and ((counter < 50) or (percent_changes > 0.01)):
    counter = 0
    while (KE_ave > 1E-15)  and ((counter < 50) or (percent_changes > 0.01)):
        try:
            t0 = CF_sys.Time[-1]
        except:
            t0 = 0
        tic()
        CF_sys.simulate(Tf = 2000, t0 = t0, method = 'bdf', save = True, Npts = 40, note = note, use_odespy = True)
        toc()

        U_tot_arr = []
        KE_arr = []
        for i, t in enumerate(CF_sys.Time):
            CF_sys.r_expanded = CF_sys.R[:, i].flatten()
            result = CF_sys.compute_potential_energy(CF_sys.R[:, i])
            U_tot_arr.append(result['U_colloid_average'])
            KE_arr.append(CF_sys._KE(CF_sys.R[:, i], CF_sys.Time[i])['KE_colloid_ave'])
        percent_changes = (UE_ave - np.min(U_tot_arr))/UE_ave + (KE_ave - np.min(KE_arr))/KE_ave

        UE_ave = np.min(U_tot_arr)
        KE_ave = np.min(KE_arr)
        print('time = %s' %t0)
        counter += 1
        print('counter = %s' %counter)
        print('volume fraction = %s' %CF_sys.vol_frac)
    
    while (KE_ave < 1E-10) :
        canvas_center = np.mean(CF_sys.canvas.xy, axis = 0)
        CF_sys.change_canvas((CF_sys.canvas.xy - canvas_center)*0.995 + canvas_center)
        U_tot_arr = []
        KE_arr = []
        for i, t in enumerate(CF_sys.Time):
            CF_sys.r_expanded = CF_sys.R[:, i].flatten()
            result = CF_sys.compute_potential_energy(CF_sys.R[:, i])
            U_tot_arr.append(result['U_colloid_average'])
            KE_arr.append(CF_sys._KE(CF_sys.R[:, i], CF_sys.Time[i])['KE_colloid_ave'])

        UE_ave = np.min(U_tot_arr)
        KE_ave = np.min(KE_arr)
        print('volume fraction = %s' %CF_sys.vol_frac)

os.system('say "your program has finished"')

