import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from rose.utils.plot_utils_tmp import create_animation,create_train_animation, create_train_animation_new

# with open('genieten_zonder_damping.json') as f:
#     res_numerical = json.load(f)
with open(os.path.join('../rose/batch_results', r"res_Kdyn_Segment 1090_scenario 1_no_damping_140.pickle"), 'rb') as f:
    res_numerical = pickle.load(f)
with open(os.path.join('../rose/batch_results', r"res_Kdyn_Segment 1001_scenario 1_no_damping_140.pickle"), 'rb') as f:
    res_numerical2 = pickle.load(f)

disp_train = np.array(res_numerical['vertical_displacements_train'])[:,500:]
rot_train = np.array(res_numerical['rot_train'])[:,500:]
vert_force_train = np.array(res_numerical['vertical_force_train'])[:,500:]


time = np.array(res_numerical['time'])[500:]
dt = np.diff(time)

velocity = np.array(res_numerical['velocity'])[500:]
distance = np.insert(np.cumsum(dt*velocity[1:]),0,0)

wheel_dist = 2.5

bogie_dist = 20

x_wheel1 = distance
x_wheel2 = distance + wheel_dist
x_wheel3 = distance + bogie_dist
x_wheel4 = distance + bogie_dist + wheel_dist

y_wheel1 = disp_train[2,:]
y_wheel2 = disp_train[3,:]
y_wheel3 = disp_train[5,:]
y_wheel4 = disp_train[6,:]

x_bogie_w1 = x_wheel1
x_bogie_w2 = x_wheel2
x_bogie_w3 = x_wheel3
x_bogie_w4 = x_wheel4

x_bogie_1 = np.vstack((x_bogie_w1,x_bogie_w2))
x_bogie_2 = np.vstack((x_bogie_w3,x_bogie_w4))

y_bogie_w1 = disp_train[1,:] +0.5 * wheel_dist * rot_train[1,:] -1.1*disp_train[1,0]
y_bogie_w2 = disp_train[1,:] -0.5 * wheel_dist * rot_train[1,:] -1.1*disp_train[1,0]
y_bogie_w3 = disp_train[4,:] +0.5 * wheel_dist * rot_train[4,:] -1.1*disp_train[4,0]
y_bogie_w4 = disp_train[4,:] -0.5 * wheel_dist * rot_train[4,:] -1.1*disp_train[4,0]

# y_bogie_1 = np.vstack((y_bogie_w1, y_bogie_w2))
# y_bogie_2 = np.vstack((y_bogie_w3, y_bogie_w4))
#
#
# x_cart_b1 = (x_bogie_w1 + x_bogie_w2) / 2
# x_cart_b2 = (x_bogie_w3 + x_bogie_w4) / 2
#
# x_cart_1 = np.vstack((x_cart_b1, x_cart_b2))
#
# y_cart_b1 = disp_train[0,:] +0.5 * bogie_dist * rot_train[0,:] -1.1*disp_train[0,0]
# y_cart_b2 = disp_train[0,:] -0.5 * bogie_dist * rot_train[0,:] -1.1*disp_train[0,0]
#
# y_cart_1 = np.vstack((y_cart_b1, y_cart_b2))


# plt.plot(x_wheel1[500:600], y_wheel1[500:600], color='r')
# plt.plot(x_wheel2[500:600], y_wheel2[500:600], color='r')
# plt.plot(x_wheel3[500:600], y_wheel3[500:600], color='r')
# plt.plot(x_wheel4[500:600], y_wheel4[500:600], color='r')
#
# plt.plot(x_bogie_w1[500:600], y_bogie_w1[500:600], color='b')
# plt.plot(x_bogie_w2[500:600], y_bogie_w2[500:600], color='b')
# plt.plot(x_bogie_w3[500:600], y_bogie_w3[500:600], color='b')
# plt.plot(x_bogie_w4[500:600], y_bogie_w4[500:600], color='b')
#
# plt.plot(x_cart_b1[500:600], y_cart_b1[500:600], color='g')
# plt.plot(x_cart_b2[500:600], y_cart_b2[500:600], color='g')
#
#
# plt.plot(x_wheel1[600], y_wheel1[600], color='r', marker='o')
# plt.plot(x_wheel2[600], y_wheel2[600], color='r', marker='o')
# plt.plot(x_wheel3[600], y_wheel3[600], color='r', marker='o')
# plt.plot(x_wheel4[600], y_wheel4[600], color='r', marker='o')
# plt.plot(x_bogie_w1[600], y_bogie_w1[600], color='b', marker='o')
# plt.plot(x_bogie_w2[600], y_bogie_w2[600], color='b', marker='o')
# plt.plot(x_bogie_w3[600], y_bogie_w3[600], color='b', marker='o')
# plt.plot(x_bogie_w4[600], y_bogie_w4[600], color='b', marker='o')
# plt.plot(x_cart_b1[600], y_cart_b1[600], color='g', marker='o')
# plt.plot(x_cart_b2[600], y_cart_b2[600], color='g', marker='o')
#
#
#
# plt.show()

cart_fn = r"D:\software_development\ROSE\static\only_locomotive.png"
bogie_fn = r"D:\software_development\ROSE\static\bogie.png"
wheel_fn = r"D:\software_development\ROSE\static\wheel.png"

# create_train_animation_new(cart_fn, bogie_fn, wheel_fn, r"move_wheels.html", (x_wheel1, x_wheel2, x_wheel3, x_wheel4),
#                            (y_wheel1, y_wheel2, y_wheel3, y_wheel4), (x_bogie_1, x_bogie_2),
#                            (y_bogie_1, y_bogie_2), (x_cart_1), (y_cart_1), fps=60)


create_train_animation(r"move_wheels.html", (x_wheel1, x_wheel2, x_wheel3, x_wheel4),
                       (y_wheel1, y_wheel2, y_wheel3, y_wheel4), (x_bogie_1, x_bogie_2, x_cart_1),
                       (y_bogie_1, y_bogie_2, y_cart_1), fps=60)

# with open(os.path.join('batch_results',r"res_Kdyn_Segment 1001_scenario 1_temp_load.pickle"), 'rb') as f:
#     res_numerical3 = pickle.load(f)
# with open(os.path.join('batch_results',r"res_Kdyn_Segment 1001_scenario 2_temp_load.pickle"), 'rb') as f:
#     res_numerical4 = pickle.load(f)

vertical_displacements_soil = np.array(res_numerical['vertical_displacements_soil'])[:,int(500/1):]
vertical_displacements_soil2 = np.array(res_numerical2['vertical_displacements_soil'])[:,int(500/1):]

vertical_force_soil = np.array(res_numerical['vertical_force_soil'])[:,int(500/1):]
vertical_force_soil2 = np.array(res_numerical2['vertical_force_soil'])[:,int(500/1):]

# vertical_force_soil = np.array(res_numerical['vertical_force_soil'])[:,int(0/1):]
# vertical_force_soil2 = np.array(res_numerical2['vertical_force_soil'])[:,int(0/1):]

vertical_force_rail = np.array(res_numerical['vert_force_rail'])[:,int(500/1):]
vertical_force_rail2 = np.array(res_numerical2['vert_force_rail'])[:,int(500/1):]

vertical_displacements_rail = np.array(res_numerical['vert_disp_rail'])[:,500:]
vertical_displacements_rail2 = np.array(res_numerical2['vert_disp_rail'])[:,500:]
# # vertical_displacements_rail3 = np.array(res_numerical3['vert_disp_rail'])[:,500:]
# # vertical_displacements_rail4 = np.array(res_numerical4['vert_disp_rail'])[:,500:]
# vertical_force_rail = np.array(res_numerical['vert_force_rail'])[:,int(500/1):]
# vertical_force_wheel1 = np.array(res_numerical['vertical_force_train'])[-1,int(500/1):]


# tot_calc_time = 1.2       # total time during calculation phase   [s]
# n_t_calc = int(800/1)
#
# time = np.linspace(0, tot_calc_time, n_t_calc)
#
# import rose.tests.utils.signal_proc as sp
# freq, amplitude_num,_,_ = sp.fft_sig(vertical_displacements_rail[110, :], int(1 / time[1]),
#                                      nb_points=2**14)


fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
# ax1.plot(x, y)
# ax2.plot(x, -y)
ax1.plot(time[:], vertical_force_soil[100, :]/1000, 'blue')
ax1.plot(time, vertical_force_soil2[100, :]/1000,'red')

ax1.set_xlabel('time [s]')
ax1.set_ylabel('Force [kN]')

ax2.plot(time[:], vertical_displacements_soil[200, :] * 1000, 'blue')
ax2.plot(time, vertical_displacements_soil2[200, :] * 1000,'red')

ax2.set_xlabel('time [s]')
ax2.set_ylabel('Displacement [mm]')
ax2.set_ylim([-1.25, 0.135])


# ax1.plot(time[:], vertical_force_rail[100, :])
# ax1.plot(time, vertical_force_rail2[100, :])
#
# ax2.plot(time[:], vertical_displacements_rail[100, :])
# ax2.plot(time, vertical_displacements_rail2[100, :])

plt.legend(["Stiffness = 11e7", "Stiffness = 11e8"])
fig.set_size_inches(7.5,4.2)
# plt.plot(freq,amplitude_num)
plt.show()
# vertical_displacements_soil_2 = np.array(res_numerical_2['vertical_displacements_soil'])

# plt.plot(vertical_force_rail[110,:] )
# plt.plot(vertical_displacements_soil[200,:])
# plt.plot(vertical_displacements_soil2[200,:])

# plt.plot(vertical_force_soil[200,:])
# plt.plot(vertical_force_soil2[200,:])

# plt.plot(vertical_force_soil[0::2,0])
# plt.plot(vertical_force_soil2[0::2,0])


# plt.plot(vertical_force_rail[100,:])
# plt.plot(vertical_force_rail2[100,:])

# plt.plot(vertical_displacements_rail[100,:])
# plt.plot(vertical_displacements_rail2[90,:])
# plt.plot(vertical_displacements_rail2[100,:])
# plt.plot(vertical_displacements_rail2[110,:])
# plt.plot(vertical_displacements_rail2[120,:])
# plt.plot(disp_train[-6,:])
# plt.plot(vert_force_train[-4,:])
# vert_force_train
# plt.show()

# plt.legend([str(2.4e8), str(1.8e8)])
# plt.plot(vertical_displacements_soil_2[170,:] )
# plt.show()
# result = {"vert_disp_rail": vertical_displacements_rail.tolist(),
#           "vertical_displacements_rail_pad": vertical_displacements_rail_pad.tolist(),
#           "vertical_displacements_sleeper": vertical_displacements_sleeper.tolist(),
#           "vertical_displacements_soil": vertical_displacements_soil.tolist(),

# plt.plot(np.array(res_numerical['vert_disp_rail'])[75,:])
# vert_dips_rail = np.array(res_numerical['vert_disp_rail'])
# for i in range(80,81):
#     plt.plot(vert_dips_rail[i,:],marker='x')
# plt.show()
# plt.plot(np.array(res_numerical['vert_disp_rail'])[75,:],marker='x')


# plt.plot(np.array(res_numerical['vertical_displacements_sleeper'])[70,:])
# plt.plot(np.array(res_numerical['vertical_displacements_rail_pad'])[150,:])


# min_train, min_load = vertical_displacements_rail.min(), vertical_displacements_rail2.min()
# create_animation(r"res_Kdyn_Segment 1090_scenario 1_compare.html", (np.array(res_numerical['coords_rail'][:]), np.array(res_numerical2['coords_rail'][:])),
#                  (vertical_displacements_rail, vertical_displacements_rail2),fps=60)

# create_animation(r"res_Kdyn_Segment 1001_scenario 2_rail.html", (np.array(res_numerical['coords_rail'][:])),
#                  (np.array(res_numerical['vert_disp_rail'])[:,1000:]),fps=60)

# create_animation("moving_train.html", (np.array(res_numerical['coords_rail'][:])),
#                  (np.array(res_numerical['vert_disp_rail'])[:,100:]),fps=60)

# np.array(res_numerical['coords_rail'][0:175])

pass



# plt.legend(['rail', 'sleeper', 'rail_pad'])
# plt.plot(np.array(res_numerical['vertical_displacements_soil'])[5,:])
# plt.show()

# plt.plot(np.array(res_numerical['vertical_displacements_sleeper'])[:,5])
# plt.show()