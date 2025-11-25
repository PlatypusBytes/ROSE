import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence


class RailDefect:
    """
    Class containing a rail defect following the trajectory of the centre of the wheel rolling over the rail defect.

    :Attributes:
        - :self.irregularities:     np array of the irregularities at each position
    """

    def __init__(self, x_track: np.ndarray, wheel_diameter: float,
                 local_defect_geometry_coordinates: Sequence[Sequence[float]], start_position: float):
        """
        Creates an array with a rail defect. The irregularity is following the trajectory of the centre of the wheel
        rolling over the rail defect defined by local_defect_geometry_coordinates.

        :param x_track: global x coordinates of the track where the irregularity should be calculated [m]
        :param wheel_diameter: diameter of the wheel [m]
        :param local_defect_geometry_coordinates:  the local geometry coordinates of the defect, where the first column
                is the local position [m] and the second column is the defect height [m]
        :param start_position: global starting position of the defect [m]
        """

        # wheel radius
        R = wheel_diameter / 2.0

        global_x_coordinates_defect = np.array([pt[0] for pt in local_defect_geometry_coordinates]) + start_position
        global_y_coordinates_defect = np.array([pt[1] for pt in local_defect_geometry_coordinates])

        # add the first and last point of the track at zero height
        global_x_coordinates_defect = np.concatenate([[x_track[0]], global_x_coordinates_defect, [x_track[-1]]])
        global_y_coordinates_defect = np.concatenate([[0.0], global_y_coordinates_defect, [0.0]])

        self.irregularities = np.zeros_like(x_track)

        # The candidate heights from vertices and segments are calculated, then the max is taken. As any point can be the
        # contact point.

        # Broadcast: (N_track_points, 1) - (1, N_vertices)
        dx_v = x_track[:, None] - global_x_coordinates_defect[None, :]

        # Valid only where the wheel center is horizontally within R of the vertex
        valid_v = np.abs(dx_v) <= R

        # Circle Equation: y = y_vert + sqrt(R^2 - dx^2) - R
        # Initialize with -inf so invalid points don't affect the max
        h_vertices = np.full(dx_v.shape, -np.inf)

        # Compute circle arcs
        y_broadcasted = np.broadcast_to(global_y_coordinates_defect[None, :], dx_v.shape)
        h_vertices[valid_v] = y_broadcasted[valid_v] + np.sqrt(np.maximum(0, R ** 2 - dx_v[valid_v] ** 2)) - R

        # Calculate geometric properties of segments
        d_seg_x = np.diff(global_x_coordinates_defect)
        d_seg_y = np.diff(global_y_coordinates_defect)
        lengths = np.hypot(d_seg_x, d_seg_y)

        # Unit normal vectors (pointing "up" relative to the line direction)
        # If moving Left->Right, Normal is (-dy, dx) / L
        nx = -d_seg_y / lengths
        ny = d_seg_x / lengths

        # Valid X range for the *Wheel Center* to touch this segment
        # The contact shifts by R * normal_x
        seg_start_x = global_x_coordinates_defect[:-1] + nx * R
        seg_end_x = global_x_coordinates_defect[1:] + nx * R

        # Broadcast: (N_track_points, 1) vs (1, N_segments)
        # Check which track points fall within the valid rolling range of each segment
        in_segment_range = (x_track[:, None] >= seg_start_x[None, :]) & (x_track[:, None] <= seg_end_x[None, :])

        # Line equation for the wheel center trajectory parallel to the segment
        # The trajectory is offset from the segment by R in the normal direction.
        # y_traj(x) = y_seg_start + R*ny + slope * (x - x_seg_start_adjusted) - R

        # Avoid division by zero for vertical walls (d_seg_x = 0)
        # For vertical walls, the wheel cannot "roll" on top, so we can ignore/mask them.
        slope = np.divide(d_seg_y, d_seg_x, out=np.zeros_like(d_seg_x), where=d_seg_x != 0)

        # Calculate heights for segments
        h_segments = np.full((len(x_track), len(d_seg_x)), -np.inf)

        if np.any(in_segment_range):
            # Base height at the start of the trajectory segment
            base_y = global_y_coordinates_defect[:-1] + ny * R

            # X distance from the start of the valid trajectory segment
            dx_traj = x_track[:, None] - seg_start_x[None, :]

            # Linear projection
            # Result = (Base Height) + (Slope * dist) - R
            seg_heights = base_y[None, :] + slope[None, :] * dx_traj - R

            h_segments[in_segment_range] = seg_heights[in_segment_range]

        # Combine the results from vertices and segments
        all_candidates = np.hstack([h_vertices, h_segments])

        # The wheel rides on the highest point of contact, negative irregularity means upward deviation
        self.irregularities = -np.max(all_candidates, axis=1)

class WheelFlat:
    """
   Class containing a wheel flat.

   :Attributes:
       - :self.irregularities:     np array of the irregularities at each position
   """
    def __init__(self, x: np.ndarray, wheel_diameter: float, flatness_length: float, start_position=None, seed=14):
        """
        Creates an array with a wheel flat. where every circumference of the wheel, the wheel is indented.
        Adapted from :cite:`Uzzal_2013`.

        :param x: position of the node [m]
        :param wheel_diameter: diameter of the wheel [m]
        :param flatness_length: total flat length of the wheel [m]
        :param start_position: optional starting position of where a wheel flat is known
        :param seed: (default 14) seed for random generator
        """

        # random generator
        random_generator = np.random.default_rng(seed)

        # calculate wheel circumference
        wheel_circumference = np.pi * wheel_diameter

        # calculate half of the angle between the flat part and the new wheel radius at wheel flat
        half_angle_wheel_flat = np.arcsin(flatness_length / wheel_diameter)

        # calculate new wheel radius at wheel flat
        new_radius = (wheel_diameter / 2) * np.cos(half_angle_wheel_flat)

        # calculate height of the wheel flat, with respect to the bottom
        irregularity_height = wheel_diameter / 2 - new_radius

        # Apply a random start distance if no start position is chosen, else the wheel flat starts at the start
        # position, note that also prior wheel rotations are taken into account
        if start_position is None:
            random_start = random_generator.random(1)[0] * wheel_circumference
            x = x + random_start
        else:
            x = x - start_position

        # an array is made which contains a value if the wheel has made a round, else the array contains zeros
        n_circumferences = x / wheel_circumference

        # create array which indicates the amount of rotations of the wheel as an integer
        tmp_array = np.ceil(n_circumferences).astype(int)

        # wheel flat index is located at every new wheel rotation
        wheel_flat_indices = []
        for i in range(1, len(tmp_array) - 1):
            if (tmp_array[i] > tmp_array[i - 1]):
                wheel_flat_indices.append(i)
        wheel_flat_indices = np.array(wheel_flat_indices)

        # generate irregularities array
        self.irregularities = np.zeros_like(x)
        self.irregularities[wheel_flat_indices] = -irregularity_height


class RailIrregularities:
    """
    Class containing rail unevenness following :cite:`zhang_2001`.

    :Attributes:
        - :self.Av:                 Vertical track irregularity parameter
        - :self.omega_c:            critical wave angular frequency
        - :self.omega:              wave angular frequency
        - :self.irregularities:     np array of the irregularities at each position
    """
    def __init__(self, x: np.ndarray,
                 f_min: float = 2, f_max: float = 500, N: int = 2000, Av: float = 0.00002095, omega_c: float = 0.8242, seed=99):
        """
        Creates rail unevenness following :cite:`zhang_2001`.

        A summary of default values can be found in :cite:`Podworna_2015`.

        Parameters
        ----------
        :param x: position of the node
        :param f_min: (default 2 Hz) minimum frequency for the PSD of the unevenness
        :param f_max: (default 500 Hz) maximum frequency for the PSD of the unevenness
        :param N: (default 2000) number of frequency increments
        :param Av: (default 0.00002095 m2 rad/m) vertical track irregularity parameters
        :param omega_c: (default 0.8242 rad/m) critical wave number
        :param seed: (default 99) seed for random generator
        """

        # default parameters
        self.Av = Av
        self.omega_c = omega_c
        # random generator
        random_generator = np.random.default_rng(seed)

        # define omega range
        omega_max = 2 * np.pi * f_max
        omega_min = 2 * np.pi * f_min
        delta_omega = (omega_max - omega_min) / N
        self.omega = np.linspace(omega_min, omega_max, N)

        # irregularities
        self.irregularities = np.zeros(len(x))

        # for each frequency increment
        for n in range(N):
            omega_n = omega_min + delta_omega * n
            phi = random_generator.uniform(0, 2 * np.pi)
            self.irregularities += np.sqrt(4 * self.spectral(omega_n) * delta_omega) * np.cos(omega_n * x - phi)

        # # compute spectrum
        # sig = signal_tools.Signal(x, self.irregularities)
        # sig.psd()
        # self.frequency_spectrum = sig.frequency_Pxx
        # self.spectrum = sig.Pxx

        return

    def spectral(self, omega):
        """
        Computes spectral unevenness

        :param omega: angular frequency rad/s
        :return:
        """
        spectral_unevenness = 2 * np.pi * self.Av * self.omega_c ** 2 / ((omega ** 2 + self.omega_c ** 2) * omega ** 2)
        return spectral_unevenness


if __name__ == "__main__":
    distance = np.linspace(0, 100, 10001)

    r = RailIrregularities(distance, seed=14)
    wheel_diameter = 0.92
    r_defect = RailDefect(distance,wheel_diameter, local_defect_geometry_coordinates=np.array([[0,0], [2, 0.002], [5,0]]), start_position=14)

    irr = r.irregularities + r_defect.irregularities
    plt.plot(distance, irr)
    plt.grid()
    plt.show()
