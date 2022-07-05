import pytest

import numpy as np

from rose.model.train_model import TrainModel, Cart, Bogie, Wheel
from solvers.zhai_solver import ZhaiSolver

class TestTrainModel:

    def test_calculate_total_static_load_bogie_and_wheel(self):
        """
        Tests the calculation of the total static load of a train which consists of 1 bogie and 1 wheel
        :return:
        """
        # Setup parameters train
        mass_wheel = 5750
        mass_bogie = 3000

        velocity = 100 / 3.6

        # setup geometry train
        wheel = Wheel()
        wheel.mass = mass_wheel

        bogie = Bogie()
        bogie.wheels = [wheel]
        bogie.wheel_distances = [0]
        bogie.mass = mass_bogie

        cart = Cart()
        cart.bogies = [bogie]
        cart.bogie_distances = [0]

        train = TrainModel()
        train.carts = [cart]
        train.time = np.array([0,1])

        train.velocities = np.ones(len(train.time)) * velocity
        train.cart_distances = [0]

        # initialise train
        train.calculate_distances()
        train.set_mesh()
        train.get_train_parts()
        train.initialise_ndof()

        # set static force vector
        train.static_force_vector = np.array([0, 0, -mass_bogie*9.81, 0, -mass_wheel*9.81])[:, None]
        train.global_force_vector = np.array([0, 0, -mass_bogie*9.81, 0, -mass_wheel*9.81])[:, None]

        # calculate total static load
        train.calculate_total_static_load(0)

        # get static load results
        calculated_bogie_load = train.carts[0].bogies[0].total_static_load
        calculated_wheel_load = train.carts[0].bogies[0].wheels[0].total_static_load

        # calculate expected static loads
        expected_bogie_load = -mass_bogie * 9.81
        expected_wheel_load = expected_bogie_load - mass_wheel * 9.81

        assert pytest.approx(calculated_bogie_load) == expected_bogie_load
        assert pytest.approx(calculated_wheel_load) == expected_wheel_load

    def test_calculate_total_static_load_shared_bogie(self):
        """
        Tests the calculation of the total static load of a train which consists of 1 bogie and 1 wheel
        :return:
        """
        # Setup parameters train
        mass_wheel = 5750
        mass_bogie = 3000
        mass_cart = 2000
        velocity = 100 / 3.6

        # create carts and bogies
        carts = [Cart(), Cart()]
        bogies = [Bogie(), Bogie(), Bogie()]

        train = TrainModel()
        train.time = np.array([0,1])
        train.velocities = np.ones(len(train.time)) * velocity

        # set up carts
        train.cart_distances = [0,10]
        train.carts = carts

        # set up bogies
        train.carts[0].bogies = [bogies[0], bogies[1]]
        train.carts[1].bogies = [bogies[1], bogies[2]]

        for cart in train.carts:
            cart.bogie_distances = [-5,5]
            cart.mass = mass_cart
            cart.length = 10

            # setup bogies per cart
            for bogie in cart.bogies:
                bogie.wheel_distances = [-1.5,1.5]
                bogie.mass = mass_bogie
                bogie.length = 3

                # setup wheels per bogie
                bogie.wheels = [Wheel(), Wheel()]
                for wheel in bogie.wheels:
                    wheel.mass = mass_wheel

        # initialise train
        train.calculate_distances()
        train.set_mesh()
        train.get_train_parts()
        train.initialise_ndof()

        # set static force vector
        train.initialize_force_vector()

        # calculate total static load
        train.calculate_total_static_load(0)

        # calculate expected static loads in the carts, bogies and wheels
        expected_static_load_cart = - mass_cart * 9.81
        expected_static_load_end_bogies = 0.5*expected_static_load_cart - mass_bogie*9.81
        expected_static_load_mid_bogie = expected_static_load_cart - mass_bogie*9.81

        expected_static_load_end_wheels = expected_static_load_end_bogies / 2 - mass_wheel * 9.81
        expected_static_load_mid_wheels = expected_static_load_mid_bogie / 2 - mass_wheel * 9.81

        # get the end wheels and the middle wheels
        end_wheels = np.array(train.wheels)[[0, 1, 4, 5]]
        mid_wheels = np.array(train.wheels)[[2, 3]]

        # assert static load on end wheels
        for wheel in end_wheels:
            assert pytest.approx(wheel.total_static_load) == expected_static_load_end_wheels

        # assert static load on mid wheels
        for wheel in mid_wheels:
            assert pytest.approx(wheel.total_static_load) == expected_static_load_mid_wheels

    def test_set_global_mass_matrix_cart(self, expected_cart_mass_matrix, set_up_cart):
        """
        Checks of mass matrix of cart is as expected
        :param expected_cart_mass_matrix:
        :param set_up_cart:
        :return:
        """
        train = set_up_cart
        train.set_global_mass_matrix()

        calculated_mass_matrix = train.global_mass_matrix

        for i in range(len(expected_cart_mass_matrix)):
            for j in range(len(expected_cart_mass_matrix[i])):
                assert expected_cart_mass_matrix[i][j] == pytest.approx(calculated_mass_matrix[i, j])

    def test_set_global_stiffness_matrix_cart(self, expected_cart_stiffness_matrix, set_up_cart):
        """
        Checks if stiffness matrix of cart is as expected

        :param expected_cart_stiffness_matrix:
        :param set_up_cart:
        :return:
        """
        train = set_up_cart
        train.set_global_stiffness_matrix()

        calculated_stiffness_matrix = train.global_stiffness_matrix

        for i in range(len(expected_cart_stiffness_matrix)):
            for j in range(len(expected_cart_stiffness_matrix[i])):
                assert expected_cart_stiffness_matrix[i][j] == pytest.approx(calculated_stiffness_matrix[i,j])

    def test_set_global_damping_matrix_cart(self, expected_cart_damping_matrix, set_up_cart):
        """
        Checks if stiffness matrix of cart is as expected

        :param expected_cart_damping_matrix:
        :param set_up_cart:
        :return:
        """

        train = set_up_cart
        train.set_global_damping_matrix()

        calculated_damping_matrix = train.global_damping_matrix

        for i in range(len(expected_cart_damping_matrix)):
            for j in range(len(expected_cart_damping_matrix[i])):
                assert expected_cart_damping_matrix[i][j] == pytest.approx(calculated_damping_matrix[i, j])


@pytest.fixture
def expected_cart_stiffness_matrix():
   k1 = 2.14e6
   k2 = 5.32e6
   lt = 3
   lw = 1

   return [[2*k2,	0.,	        -k2,	    0.,	        0.,	    0.,	    -k2,	    0.,	        0.,	        0.],
        [0.,	    2*k2*lt**2,	-k2*lt,	    0.,	        0.,	    0.,	    k2*lt,	    0.,	        0.,	        0.],
        [-k2,	-k2*lt,	        k2+2*k1,	0.,	        -k1,    -k1,	0.,	        0.,	        0.,	        0.],
        [0.,	    0.,	        0.,	        2*k1*lw**2,	-k1*lw,	k1*lw,	0.,	        0.,	        0.,	        0.],
        [0.,	    0.,	        -k1,	    -k1*lw,	    k1,	    0.,	    0.,	        0.,	        0.,	        0.],
        [0.,	    0.,	        -k1,	    k1*lw,	    0.,	    k1,	    0.,	        0.,	        0.,	        0.],
        [-k2,	k2*lt,	        0.,	        0.,	        0.,	    0.,	    k2+2*k1,	0.,	        -k1,	    -k1],
        [0.,	    0.,	        0.,	        0.,	        0.,	    0.,	    0.,	        2*k1*lw**2,	-k1*lw,	    k1*lw],
        [0.,	    0.,	        0.,	        0.,	        0.,	    0.,	    -k1,	    -k1*lw,	    k1,	        0.],
        [0.,	    0.,	        0.,	        0.,	        0.,	    0.,	    -k1,	    k1*lw,	    0.,	        k1]]


@pytest.fixture
def expected_cart_damping_matrix():
   c1 = 4.9e4
   c2 = 7e4
   lt = 3
   lw = 1

   return [[2*c2,	0.,	        -c2,	    0.,	        0.,	    0.,	    -c2,	    0.,	        0.,	        0.],
        [0.,	    2*c2*lt**2,	-c2*lt,	    0.,	        0.,	    0.,	    c2*lt,	    0.,	        0.,	        0.],
        [-c2,	-c2*lt,	        c2+2*c1,	0.,	        -c1,    -c1,	0.,	        0.,	        0.,	        0.],
        [0.,	    0.,	        0.,	        2*c1*lw**2,	-c1*lw,	c1*lw,	0.,	        0.,	        0.,	        0.],
        [0.,	    0.,	        -c1,	    -c1*lw,	    c1,	    0.,	    0.,	        0.,	        0.,	        0.],
        [0.,	    0.,	        -c1,	    c1*lw,	    0.,	    c1,	    0.,	        0.,	        0.,	        0.],
        [-c2,	c2*lt,	        0.,	        0.,	        0.,	    0.,	    c2+2*c1,	0.,	        -c1,	    -c1],
        [0.,	    0.,	        0.,	        0.,	        0.,	    0.,	    0.,	        2*c1*lw**2,	-c1*lw,	    c1*lw],
        [0.,	    0.,	        0.,	        0.,	        0.,	    0.,	    -c1,	    -c1*lw,	    c1,	        0.],
        [0.,	    0.,	        0.,	        0.,	        0.,	    0.,	    -c1,	    c1*lw,	    0.,	        c1]]


@pytest.fixture
def expected_cart_mass_matrix():
    mc = 77000
    ic = 1.2e6
    mb = 1100
    ib = 760
    mw = 1200

    return [[mc,	0.,	        0.,	    0.,	        0.,	    0.,	    0.,	    0.,	    0.,	    0.],
            [0.,    ic,	        0.,	    0.,	        0.,	    0.,	    0.,	    0.,	    0.,	    0.],
            [0.,	0.,	        mb,	    0.,	        0.,     0.,	    0.,	    0.,	    0.,	    0.],
            [0.,	0.,	        0.,	    ib,	        0.,	    0.,	    0.,	    0.,	    0.,	    0.],
            [0.,	0.,	        0.,	    0.,	        mw,	    0.,	    0.,	    0.,	    0.,	    0.],
            [0.,	0.,	        0.,	    0.,	        0.,	    mw,	    0.,	    0.,	    0.,	    0.],
            [0.,	0.,	        0.,	    0.,	        0.,	    0.,	    mb,	    0.,	    0.,	    0.],
            [0.,	0.,	        0.,	    0.,	        0.,	    0.,	    0.,	    ib,	    0.,     0.],
            [0.,	0.,	        0.,	    0.,	        0.,	    0.,	    0.,	    0.,	    mw,	    0.],
            [0.,	0.,	        0.,	    0.,	        0.,	    0.,	    0.,	    0.,	    0.,	    mw]]


@pytest.fixture
def set_up_cart():
    """
    Set up cart with 2 bogies and 2 wheel sets per bogie
    :return:
    """

    mass_wheel = 1200
    mass_bogie = 1100
    mass_cart = 77000
    inertia_cart = 1.2e6
    inertia_bogie = 760
    prim_stiffness = 2.14e6
    sec_stiffness = 5.32e6
    prim_damping = 4.9e4
    sec_damping = 7e4


    length_cart = 3
    length_bogie = 1

    wheels = [Wheel() for _ in range(4)]

    for wheel in wheels:
        wheel.mass = mass_wheel

    bogies = [Bogie() for _ in range(2)]
    bogies[0].wheels = wheels[:2]
    bogies[1].wheels = wheels[2:]
    for bogie in bogies:
        bogie.wheel_distances=[-1, 1]
        bogie.mass = mass_bogie
        bogie.inertia = inertia_bogie
        bogie.stiffness = prim_stiffness  # stiffness between bogie and wheels
        bogie.damping = prim_damping
        bogie.length = length_bogie
        bogie.calculate_total_n_dof()

    cart = Cart()
    cart.bogies = bogies
    cart.bogie_distances = [-3, 3]
    cart.inertia = inertia_cart
    cart.mass = mass_cart
    cart.stiffness = sec_stiffness
    cart.damping = sec_damping
    cart.length = length_cart


    train = TrainModel()
    train.time = np.array([0])
    train.velocities = np.array([0])
    train.carts = [cart]
    train.cart_distances = [0]

    # initialise train model steps
    train.calculate_distances()
    train.set_mesh()

    # Get bogies and wheels
    train.get_train_parts()

    # check distribution of bogie and wheel stiffnesses
    train.check_distribution_factors()
    train.initialise_ndof()

    return train

