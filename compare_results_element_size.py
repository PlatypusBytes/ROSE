import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("./results_TZ_soft_stiff/DOUBLEDEKKER.pickle", "rb") as f:
    data1 = pickle.load(f)

with open("./results_TZ_soft_stiff_4/DOUBLEDEKKER.pickle", "rb") as f:
    data2 = pickle.load(f)

fig, ax = plt.subplots(2, 1)
ax[0].plot(data1["vert_disp_rail"][50], marker="o", color="b")
ax[1].plot(data1["vert_disp_rail"][100], marker="o", color="b")

ax[0].plot(data2["vert_disp_rail"][50*4], marker="x", color="r")
ax[1].plot(data2["vert_disp_rail"][100*4], marker="x", color="r")

ax[0].grid()
ax[1].grid()

ax[0].set_title("Vertical displacements of the rail at 50 m")
ax[1].set_title("Vertical displacements of the rail at 100 m")


# verification
# np.testing.assert_almost_equal(data1["vert_disp_rail"][50], data2["vert_disp_rail"][50*4], decimal=12)
# np.testing.assert_almost_equal(data1["vert_disp_rail"][100], data2["vert_disp_rail"][100*4], decimal=12)

plt.show()




fig, ax = plt.subplots(2, 1)
ax[0].plot(data1["coords_rail"], np.max(np.abs(np.array(data1["vert_disp_rail"])), axis=1), marker="o", color="b")
ax[0].plot(data2["coords_rail"], np.max(np.abs(np.array(data2["vert_disp_rail"])), axis=1), marker="x", color="r")


ax[1].plot(np.max(np.abs(np.array(data1["vertical_displacements_soil"]))[::2], axis=1), marker="o", color="b")
ax[1].plot(np.max(np.abs(np.array(data2["vertical_displacements_soil"]))[::2], axis=1), marker="x", color="r")

ax[0].grid()
ax[1].grid()

plt.show()



plt.plot(data1["vertical_displacements_soil"][50])
plt.plot(data1["vertical_displacements_soil"][100])
plt.plot(data2["vertical_displacements_soil"][50])
plt.plot(data2["vertical_displacements_soil"][100])
plt.show()