import pytest

import numpy as np

from rose.train_model.train_model import TrainModel, Cart, Bogie, Wheel
from rose.solver.solver import NewmarkSolver, StaticSolver, ZhaiSolver

class TestTrainModel:

    @pytest.mark.workinprogress
    def test_set_aux_mass_matrix_cart(self, expected_cart_mass_matrix, set_up_cart):
        """
        Checks of mass matrix of cart is as expected
        :param expected_cart_mass_matrix:
        :param set_up_cart:
        :return:
        """
        cart = set_up_cart
        cart.set_aux_mass_matrix()

        calculated_mass_matrix = cart.aux_mass_matrix

        for i in range(len(expected_cart_mass_matrix)):
            for j in range(len(expected_cart_mass_matrix[i])):
                assert expected_cart_mass_matrix[i][j] == pytest.approx(calculated_mass_matrix[i, j])

    @pytest.mark.workinprogress
    def test_set_aux_stiffness_matrix_cart(self, expected_cart_stiffness_matrix, set_up_cart):
        """
        Checks if stiffness matrix of cart is as expected

        :param expected_cart_stiffness_matrix:
        :param set_up_cart:
        :return:
        """
        cart = set_up_cart
        cart.set_aux_stiffness_matrix()

        calculated_stiffness_matrix = cart.aux_stiffness_matrix

        for i in range(len(expected_cart_stiffness_matrix)):
            for j in range(len(expected_cart_stiffness_matrix[i])):
                assert expected_cart_stiffness_matrix[i][j] == pytest.approx(calculated_stiffness_matrix[i,j])

    @pytest.mark.workinprogress
    def test_set_aux_damping_matrix_cart(self, expected_cart_damping_matrix, set_up_cart):
        """
        Checks if stiffness matrix of cart is as expected

        :param expected_cart_stiffness_matrix:
        :param set_up_cart:
        :return:
        """
        cart = set_up_cart
        cart.set_aux_damping_matrix()

        calculated_damping_matrix = cart.aux_damping_matrix

        for i in range(len(expected_cart_damping_matrix)):
            for j in range(len(expected_cart_damping_matrix[i])):
                assert expected_cart_damping_matrix[i][j] == pytest.approx(calculated_damping_matrix[i, j])

    @pytest.mark.workinprogress
    def test_train(self, set_up_cart):
        train = TrainModel()
        train.carts = [set_up_cart]
        time = np.linspace(0,1,10000)


        train.time = time
        train.velocities = np.ones(len(train.time)) * 3.6
        train.cart_distances = [0]

        train.herzian_contact_cof = 9.1e-7

        train.calculate_distances()

        train.get_train_parts()
        train.get_irregularities_track_at_wheels()

        # train.set_mesh()
        train.calculate_active_n_dof()

        train.get_contact_dofs()

        train.get_irregularities_track_at_wheels()
        train.get_deformation_track_at_wheels()

        train.get_deformation_wheels()
        train.set_static_force_vector()
        train.initialize_force_vector()
        train.calculate_static_wheel_deformation()
        # train.calculate_static_wheel_deformation()

        train.solver = ZhaiSolver()
        # train.solver.initialise(train.active_n_dof, train.time)
        # train.solver.load_func = train.update_force_vector
        # train.set_aux_mass_matrix()
        # train.set_aux_damping_matrix()
        # train.set_aux_stiffness_matrix()

        # train.solver = NewmarkSolver()
        train.solver.initialise(train.active_n_dof, train.time)
        train.solver.load_func = train.update_force_vector
        train.set_aux_mass_matrix()
        train.set_aux_damping_matrix()
        train.set_aux_stiffness_matrix()

        # for t in range(len(train.time)-2):
        #     train.set_dynamic_force_vector(t)
        #     train.set_force_vector()
        #     train.update_stage(t, t+1)
        #     train.calculate_stage(t, t+1)
        #     a = train.force_vector

        # for t in range(len(train.time) - 2):
        # train.set_dynamic_force_vector(0)
        # train.set_force_vector()
        train.calculate_initial_displacement([0, 0, 0, 0])
        train.update_stage(0, len(time)-1)
        train.calculate_stage(0, len(time)-1)

        import matplotlib.pyplot as plt
        plt.plot(train.solver.u[:, 0])
        # plt.plot(train.solver.u[:, 8])
        # plt.plot(train.solver.u[:, 9])
        # plt.plot(train.solver.u[:, 4])
        # plt.plot(train.solver.u[:, 5])

        # plt.plot(train.solver.u[:, 4])
        # plt.plot(train.solver.u[:, 5])
        # plt.plot(train.solver.u[:, 8])
        # plt.plot(train.solver.u[:, 9])
        # plt.plot(-train.irregularities_at_wheels[-1,:])

        # plt.plot(train.solver.u[:, 8])
        # plt.plot(train.irregularities_at_wheels[-2,:])
        plt.show()


        # import matplotlib.pyplot as plt
        #
        # plt.plot(train.solver.u[:,0])
        # plt.show()

        pass



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

    # mass_wheel = 1e-6
    # mass_bogie = 1e-6
    # mass_cart = 1e-6
    # inertia_cart = 1.2e6
    # inertia_bogie = 760
    # prim_stiffness = 2.14e6
    # sec_stiffness = 5.32e6
    # prim_damping = 1e-6
    # sec_damping = 1e-6


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

    cart.calculate_total_n_dof()
    # cart.calculate_active_n_dof(0)

    return cart

