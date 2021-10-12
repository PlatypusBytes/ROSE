import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class KalmanFilter:
    def __init__(self, initial_conditions, control_variable, process_variance, delta_t, independent=False):# error_estimate, ini_estimate):
        """
        Kalman filter: Two dimensional (e.g. displacement, velocity)

        @param initial_conditions:
        @param control_variable:
        @param process_variance:
        @param delta_t:
        @param independent:
        """
        self.x = initial_conditions  # state matrix
        self.P = []  # process covariance matrix
        self.u = control_variable  # control variable matrix
        self.w = 0  # predicted state noise matrix
        self.Qr = 0  # process noise covariance matrix
        self.Y = []  # measurement of state
        self.zk = 0  # measurement noise
        self.R = []  # sensor noise covariance matrix (measurement error)
        self.K = []  # kalman gain
        self.observation = []  # measured value
        self.time_step = delta_t  # time step

        # dependency between variables
        self.independent = independent

        # A: prediction matrix: relation between equation of motion and velocity
        self.A = np.array([[1, self.time_step], [0, 1]])
        self._A = np.copy(self.A)

        # B: control matrix: relation between equation of motion and acceleration
        self.B = np.array([1 / 2 * self.time_step ** 2, self.time_step])
        self._B = np.copy(self.B)

        # H: matrix to change format
        self.H = self.A * np.identity(2)
        self._H = np.copy(self.H)

        # C: matrix to change format for measurements
        self.C = self.A * np.identity(2)
        self._C = np.copy(self.C)

        # define initial covariance matrix
        self.initial_cov_matrix(process_variance[0], process_variance[1])

        self.updated_x = [initial_conditions]
        self.updated_covariance = [self.P]

        return

    def initialise_control_matrices(self, timesteps: np.ndarray):
        """
        Initialise control matrices when timestep size is varying
        :param timesteps:
        :return:
        """
        self.A = np.ones((2,2,timesteps.size))
        self.A[0, 1, :] = timesteps
        self.A[1, 0, :] = 0

        self.B = np.zeros((2, timesteps.size))

        self.B[0, :] = 1 / 2 * timesteps ** 2
        self.B[1, :] = timesteps

        self.H = np.zeros((2,2,timesteps.size))
        self.H[0, 0, :] = self.A[0, 0, :]
        self.H[1, 1, :] = self.A[1, 1, :]

        self.C = np.copy(self.H)

    def update_control_matrices_by_index(self, t_idx):

        # A: prediction matrix: relation between equation of motion and velocity
        self._A = self.A[:,:,t_idx]

        # B: control matrix: relation between equation of motion and acceleration
        self._B = self.B[:,t_idx]

        # H: matrix to change format
        self._H = self.H[:,:,t_idx]

        # C: matrix to change format for measurements
        self._C = self.C[:,:,t_idx]

    def update_control_matrices(self, timestep):

        # A: prediction matrix: relation between equation of motion and velocity
        self._A = np.array([[1, timestep], [0, 1]])

        # B: control matrix: relation between equation of motion and acceleration
        self._B = np.array([1 / 2 * timestep ** 2, timestep])

        # H: matrix to change format
        # self.H = np.diag(np.diag(self.A))
        self._H = np.array([[1, 0], [0, 1]])

        # C: matrix to change format for measurements
        # self.C = np.diag(np.diag(self.A))
        self._C = np.array([[1, 0], [0, 1]])

    def state_matrix(self):

        self.x = np.dot(self._A, self.x) + np.dot(self._B, self.u) + self.w

        # print(self.x)

        return

    def initial_cov_matrix(self, sigma_xx, sigma_yy):

        # Process covariance matrix
        if self.independent:
            self.P = np.array([[sigma_xx ** 2, 0],
                               [0, sigma_yy ** 2]])
        else:
            self.P = np.array([[sigma_xx ** 2, sigma_xx * sigma_yy],
                               [sigma_xx * sigma_yy, sigma_yy ** 2]])

        # print(self.P)
        return

    def predict_process_cov_matrix(self):

        # predicted process covariance matrix
        self.P = np.dot(np.dot(self._A, self.P), self._A.T) + self.Qr

        if self.independent:
            self.P = self.P * np.identity(2)
            # self.P = np.diag(np.diag(self.P) )
        # print(self.P)
        return

    def error_covariance_measures(self, sigma_xx, sigma_yy):

        # Process covariance matrix
        if self.independent:
            self.R = np.array([[sigma_xx ** 2, 0],
                               [0, sigma_yy ** 2]])
        else:
            self.R = np.array([[sigma_xx ** 2, sigma_xx * sigma_yy],
                               [sigma_xx * sigma_yy, sigma_yy ** 2]])

        # print(f"R: {self.R}")
        return

    def kalman_gain(self):
        # kalman gain
        self.K = np.dot(self.P, self._H.T) / (np.dot(np.dot(self._H, self.P), self._H.T) + self.R)
        self.K[np.isnan(self.K)] = 0
        # print(f"Kalman gain: {self.K}")
        return

    def new_observation(self, y):

        # observation
        self.observation = y
        # new observation
        self.Y = np.dot(self._C, y) + self.zk
        # print(f"new Y: {self.Y}")
        return

    def predicted_state(self):

        self.x = self.x + np.dot(self.K, (self.Y - np.dot(self._H, self.x)))

        # update to results
        self.updated_x.append(self.x)
        # print(f"new x: {self.x}")
        return

    def update_process_covariance_matrix(self):

        self.P = np.dot((np.identity(len(self.K)) - np.dot(self.K, self._H)), self.P)
        # update to results
        self.updated_covariance.append(self.P)
        # print(f"new P: {self.P}")
        return


if __name__ == "__main__":

    delta_t = 1
    X0 = np.array([4000, 280])  # initial conditions [displacement, velocity]
    control_vector = 2
    P0 = [20, 5]  # initial process errors
    observation_errors = [25, 6]

    observations = np.array([[4000, 280],
                             [4260, 282],
                             [4550, 285],
                             [4860, 286],
                             [5110, 290],
                             ])

    # initialise Kalman
    kf = KalmanFilter(observations[0], control_vector, P0, delta_t, independent=True)

    for i in range(1, len(observations)):
        print(f"######################\n{i}\n######################")
        kf.update_control_matrices(delta_t)
        kf.state_matrix()
        # kf.process_cov_matrix(P0[0], P0[1])
        kf.predict_process_cov_matrix()
        kf.error_covariance_measures(observation_errors[0], observation_errors[1])
        kf.kalman_gain()
        kf.new_observation(observations[i])
        kf.predicted_state()
        kf.update_process_covariance_matrix()


    import matplotlib.pylab as plt
    plt.plot(observations[:, 0], marker="x", label="Observation")
    plt.plot(np.array(kf.updated_x)[:, 0], label="Kalman")
    plt.grid()
    plt.ylabel("Displacement")
    plt.legend()
    plt.show()
    plt.plot(observations[:, 1], marker="x", label="Observation")
    plt.plot(np.array(kf.updated_x)[:, 1], label="Kalman")
    plt.grid()
    plt.ylabel("Velocity")
    plt.legend()
    plt.show()
