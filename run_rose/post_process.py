import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from rose.utils.plot_utils_tmp import create_animation

# Note that this script works together with the output as generated from the batch_calculation.py script
# ______________________________________________________________________________________________________

# open result file from batch calculation

with open(os.path.join('../rose/batch_results', r"res_Kdyn_Segment 1001_scenario 1.pickle"), 'rb') as f:
    res_numerical = pickle.load(f)


# define first output time index of calculation phase (Note that this might not be the same as the time index which is
# used during the calculation).

first_calc_t_idx = 500

# Get displacement of each node of the train in the calculation phase
disp_train = np.array(res_numerical['vertical_displacements_train'])[:,first_calc_t_idx:]

# Get displacement of each node of the train in the calculation phase
vert_force_train = np.array(res_numerical['vertical_force_train'])[:,first_calc_t_idx:]


# get calculation time
time = np.array(res_numerical['time'])[first_calc_t_idx:]

# define delta time
dt = np.diff(time)

# get velocity in the calculation phase
velocity = np.array(res_numerical['velocity'])[first_calc_t_idx:]

# determine traveled distance
distance = np.insert(np.cumsum(dt*velocity[1:]),0,0)

# get vertical displacement in the soil during calculation phase
vertical_displacements_soil = np.array(res_numerical['vertical_displacements_soil'])[:,first_calc_t_idx:]

# get vertical force in the soil during calculation phase
vertical_force_soil = np.array(res_numerical['vertical_force_soil'])[:,first_calc_t_idx:]

# get vertical force in the rail during calculation phase
vertical_force_rail = np.array(res_numerical['vert_force_rail'])[:,first_calc_t_idx:]

# get vertical displacements of the rail during calculation phase
vertical_displacements_rail = np.array(res_numerical['vert_disp_rail'])[:,first_calc_t_idx:]

# plot vertical force and displacement of the rail at one node
fig, (ax1, ax2) = plt.subplots(2)

rail_node_nbr = 100
ax1.plot(time[:], vertical_force_rail[rail_node_nbr, :])
ax2.plot(time[:], vertical_displacements_rail[rail_node_nbr, :])

ax1.set_xlabel('time [s]')
ax1.set_ylabel('Force [N]')

ax2.set_xlabel('time [s]')
ax2.set_ylabel('Displacement [m]')

fig.set_size_inches(7.5,4.2)

plt.show()

# create animation of the vertical force in the soil over time
animation_file_name = r"force_soil_animation.html"
create_animation(animation_file_name, (np.array(res_numerical['coords_soil'][0::2])),
                 (vertical_force_soil),fps=60)

