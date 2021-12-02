from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import odespy
import os
from datetime import datetime
import time
from tqdm import tqdm
import h5py
import pylab
import math
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.sparse import coo_matrix
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import sys
cwd = os.getcwd()
sys.path.insert(1, cwd)
from rs_helper import rock_string_helpers_fullc2 as rs
from rs_helper.rock_string_analysis_helper import *
from matplotlib import animation, rc
from IPython.display import HTML
from moviepy.editor import VideoFileClip, concatenate_videoclips

env_var = os.environ
folder_name = env_var['FOLDERNAME']
try:
	copy_number = env_var['CPNUMBER']
except:
	copy_number = 0

exist = True
while exist:
	file_name = folder_name + 'Anim_{0:03d}.mp4'.format(copy_number)
	if os.path.exists(file_name):
		copy_number += 1
	else:
		exist = False

animate_folder_result(
    folder_name, 
    copy_number = copy_number)
