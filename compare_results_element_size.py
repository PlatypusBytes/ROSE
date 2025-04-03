import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("./results_TZ_soft_stiff/DOUBLEDEKKER.pickle", "rb") as f:
    data1 = pickle.load(f)

plt.plot(data1["vert_disp_rail"][50])
plt.plot(data1["vert_disp_rail"][100])



with open("./results_TZ_soft_stiff_4/DOUBLEDEKKER.pickle", "rb") as f:
    data2 = pickle.load(f)

plt.plot(data2["vert_disp_rail"][50*4])
plt.plot(data2["vert_disp_rail"][100*4])

# verification
np.testing.assert_almost_equal(data1["vert_disp_rail"][50], data2["vert_disp_rail"][50*4], decimal=3)
np.testing.assert_almost_equal(data1["vert_disp_rail"][100], data2["vert_disp_rail"][100*4], decimal=3)

plt.show()


plt.plot(data1["vertical_displacements_soil"][50])
plt.plot(data1["vertical_displacements_soil"][100])
plt.plot(data2["vertical_displacements_soil"][50])
plt.plot(data2["vertical_displacements_soil"][100])
plt.show()