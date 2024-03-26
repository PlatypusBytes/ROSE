from enum import Enum
import numpy as np

from rose.model.train_model import TrainModel, Cart, Bogie, Wheel



class TrainType(Enum):
    DOUBLEDEKKER = 0
    SPRINTER_SLT = 1
    SPRINTER_SGM = 2
    ICM = 3
    CARGO_FALNS5 = 4
    CARGO_SGNS = 5
    CARGO_TAPPS = 6
    TRAXX = 7
    BR189 = 8
    # RSMV = 9


def set_train(time: np.ndarray, velocities: np.ndarray, start_coord: float, train_type: TrainType, nb_carts=1):
    """
    Sets a default train according to the TrainType

    :param time: all time steps
    :param velocities: velocities of the train per time step
    :param start_coord: initial coordinate of the middle of the cart
    :param train_type: type of train
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """
    if train_type == TrainType.DOUBLEDEKKER:
        return set_double_dekker_train(time, velocities, start_coord, nb_carts=nb_carts)
    elif train_type == TrainType.SPRINTER_SLT:
        return set_sprinter_slt_train(time, velocities, start_coord, nb_carts=nb_carts)
    elif train_type == TrainType.SPRINTER_SGM:
        return set_sprinter_sgm_train(time, velocities, start_coord, nb_carts=nb_carts)
    elif train_type == TrainType.ICM:
        return set_icm_train(time, velocities, start_coord, nb_carts=nb_carts)
    elif train_type == TrainType.CARGO_FALNS5:
        return set_cargo_FALNS5_train(time, velocities, start_coord, nb_carts=nb_carts)
    elif train_type == TrainType.CARGO_SGNS:
        return set_cargo_SGNS_train(time, velocities, start_coord, nb_carts=nb_carts)
    elif train_type == TrainType.CARGO_TAPPS:
        return set_cargo_TAPPS_train(time, velocities, start_coord, nb_carts=nb_carts)
    elif train_type == TrainType.TRAXX:
        return set_traxx_locomotive(time, velocities, start_coord, nb_carts=nb_carts)
    elif train_type == TrainType.BR189:
        return set_br189_locomotive(time, velocities, start_coord, nb_carts=nb_carts)
    # elif train_type == TrainType.RSMV:
    #     return set_rsmv_train(time, velocities, start_coord)
    else:
        return None

def set_traxx_locomotive(time, velocities, start_coord, nb_carts=1):
    """
    Sets a train model with the default parameters for the Traxx locomotive

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """

    traxx_locomotive = TrainModel()
    traxx_length = 18.95
    traxx_locomotive.time = time
    traxx_locomotive.velocities = velocities
    traxx_locomotive.carts = [Cart() for _ in range(nb_carts)]
    traxx_locomotive.cart_distances = [start_coord + i * traxx_length for i in range(nb_carts)]

    for cart in traxx_locomotive.carts:
        cart.bogie_distances = [-5.175, 5.175]
        cart.inertia = 1558125/2
        cart.mass = 54000/2
        cart.stiffness = 24e6
        cart.damping = 71e3
        cart.length = traxx_length
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = [-1.3, 1.3]
            bogie.mass = 7e3/2
            bogie.inertia = 9.333e3/2
            bogie.stiffness = 2.232e6
            bogie.damping = 36.7e3
            bogie.length = 2.6
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = 4.e3

    return traxx_locomotive


def set_br189_locomotive(time, velocities, start_coord, nb_carts=1):
    """
    Sets a train model with the default parameters for the Traxx locomotive

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart

    :return: TrainModel
    """

    br189_locomotive = TrainModel()
    br189_length = 19.6
    br189_locomotive.time = time
    br189_locomotive.velocities = velocities
    br189_locomotive.carts = [Cart() for _ in range(nb_carts)]
    br189_locomotive.cart_distances = [start_coord + i * br189_length for i in range(nb_carts)]

    for cart in br189_locomotive.carts:
        cart.bogie_distances = [-4.95, 4.95]
        cart.inertia = 1592750/2
        cart.mass = 55200/2
        cart.stiffness = 24e6
        cart.damping = 71e3
        cart.length = br189_length
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = [-1.45, 1.45]
            bogie.mass = 7000/2
            bogie.inertia = 9333.3/2
            bogie.stiffness = 2.232e6
            bogie.damping = 36.7e3
            bogie.length = 2.9
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = 4.e3

    return br189_locomotive


def set_double_dekker_train(time, velocities, start_coord, nb_carts=1):
    """
    Sets a train model with the default parameters for the dutch double dekker

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """

    intercity_train = TrainModel()
    train_length = 24.1

    intercity_train.time = time
    intercity_train.velocities = velocities
    intercity_train.carts = [Cart() for _ in range(nb_carts)]
    intercity_train.cart_distances = [start_coord + i * train_length for i in range(nb_carts)]

    for cart in intercity_train.carts:
        cart.bogie_distances = [-9.95, 9.95]
        cart.inertia = 128.8e3/2
        cart.mass = 50e3/2
        cart.stiffness = 2708e3
        cart.damping = 64e3
        cart.length = train_length
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = [-1.25, 1.25]
            bogie.mass = 6e3/2
            bogie.inertia = 0.31e3/2
            bogie.stiffness = 4800e3
            bogie.damping = 0.25e3
            bogie.length = 2.5
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = 1.5e3

    return intercity_train


def set_sprinter_slt_train(time, velocities, start_coord, nb_carts=2):
    """
    Sets a train model with the default parameters for the Dutch local sprinter train

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """

    if nb_carts < 2:
        raise ValueError("Number of carts must be at least 2 for the sprinter SLT train")

    cart_length = 2 * 7.785

    sprinter_train = TrainModel()

    sprinter_train.time = time
    sprinter_train.velocities = velocities
    sprinter_train.carts = [Cart() for _ in range(nb_carts)]
    sprinter_train.cart_distances =  [start_coord + i * cart_length for i in range(nb_carts)]

    # create all bogies
    bogies = [Bogie() for _ in range(len(sprinter_train.carts) + 1)]

    for idx, cart in enumerate(sprinter_train.carts):

        cart.inertia = 73.4e3 / 2
        cart.mass = 46e3 / 2
        cart.stiffness = 2468e3
        cart.damping = 50.2e3
        cart.length = cart_length

        # connect bogies to carts
        cart.bogies =[bogies[idx], bogies[idx+1]]
        cart.bogie_distances = [-cart_length / 2, cart_length / 2]

        # right bogie is shared between 2 carts
        if idx == 0:
            cart.distribution_factor = [1, 0.5]
        # left bogie is shared between 2 carts
        elif idx == len(sprinter_train.carts) - 1:
            cart.distribution_factor = [0.5, 1]
        # both bogies are shared between 2 carts
        else:
            cart.distribution_factor = [0.5, 0.5]

    # fill in bogie parameters
    for bogie in bogies:
        bogie.wheel_distances = [-1.4, 1.4]
        bogie.mass = 3.2e3/2
        bogie.inertia = 0.17e3/2
        bogie.stiffness = 4400e3
        bogie.damping = 0.59e3
        bogie.length = 2.8
        bogie.calculate_total_n_dof()

        # setup wheels per bogie
        bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
        for wheel in bogie.wheels:
            wheel.mass = 1.5e3

    return sprinter_train


def set_sprinter_sgm_train(time, velocities, start_coord, nb_carts=1):
    """
    Sets a train model with the default parameters for the dutch local sprinter train

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """

    sprinter_train = TrainModel()
    train_length = 25.15

    sprinter_train.time = time
    sprinter_train.velocities = velocities
    sprinter_train.carts = [Cart() for _ in range(nb_carts)]
    sprinter_train.cart_distances = [start_coord + i * train_length for i in range(nb_carts)]

    for cart in sprinter_train.carts:
        cart.bogie_distances = [-9, 9]
        cart.inertia = 73.4e3/2
        cart.mass = 46e3/2
        cart.stiffness = 2468e3
        cart.damping = 50.2e3
        cart.length = 25.15
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = [-1.25, 1.25]
            bogie.mass = 3.2e3/2
            bogie.inertia = 0.17e3/2
            bogie.stiffness = 4400e3
            bogie.damping = 0.59e3
            bogie.length = 2.5
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = 1.5e3

    return sprinter_train


def set_icm_train(time, velocities, start_coord, nb_carts=1):
    """
    Sets a train model with the default parameters for the ICM train

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """

    icm_train = TrainModel()

    icm_length = 14.3

    icm_train.time = time
    icm_train.velocities = velocities
    icm_train.carts = [Cart() for _ in range(nb_carts)]
    icm_train.cart_distances = [start_coord + i * icm_length for i in range(nb_carts)]

    for cart in icm_train.carts:
        cart.bogie_distances = [-7.7, 7.7]
        cart.inertia = 91.5e3
        cart.mass = 34.3e3
        cart.stiffness = 10000e3
        cart.damping = 42.5e3
        cart.length = 14.3
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = [-0.9, 0.9]
            bogie.mass = 1.495e3
            bogie.inertia = 1.9e3
            bogie.stiffness = 2612e3
            bogie.damping = 52.2e3
            bogie.length = 1.8
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = 0.6775e3

    return icm_train

def set_cargo_SGNS_train(time, velocities, start_coord, nb_carts=1):
    """
    Sets a train model with the default parameters for the Dutch cargo train

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """

    cargo_train = TrainModel()
    cargo_length = 19.74

    cargo_train.time = time
    cargo_train.velocities = velocities
    cargo_train.carts = [Cart() for _ in range(nb_carts)]
    cargo_train.cart_distances = [start_coord + i * cargo_length for i in range(nb_carts)]

    for cart in cargo_train.carts:
        cart.bogie_distances = [-7.1, 7.1]
        cart.inertia = 784200/2
        cart.mass = 29.48e3/2
        cart.stiffness = 3.27e7
        cart.damping = 8.02e5
        cart.length = cargo_length
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = [-0.91, 0.91]
            bogie.mass = 5e3/2
            bogie.inertia = 2.1e3/2
            bogie.stiffness = 3.27e6
            bogie.damping = 8.02e4
            bogie.length = 1.82
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = 1.5e3

    return cargo_train


def set_cargo_FALNS5_train(time, velocities, start_coord, nb_carts=1):
    """
    Sets a train model with the default parameters for the Dutch cargo train

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """

    cargo_train = TrainModel()
    cargo_length = 15.79

    cargo_train.time = time
    cargo_train.velocities = velocities
    cargo_train.carts = [Cart() for _ in range(nb_carts)]
    cargo_train.cart_distances = [start_coord + i * cargo_length for i in range(nb_carts)]

    for cart in cargo_train.carts:
        cart.bogie_distances = [-5.335, 5.335]
        cart.inertia = 248278/2
        cart.mass = 60e3/2
        cart.stiffness = 2.37e8
        cart.damping = 8.02e5
        cart.length = cargo_length
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = [-0.91, 0.91]
            bogie.mass = 5.2e3/2
            bogie.inertia = 2100/2
            bogie.stiffness = 2.37e6
            bogie.damping = 8.02e4
            bogie.length = 1.82
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = 3e3

    return cargo_train


def set_cargo_TAPPS_train(time, velocities, start_coord, nb_carts=1):
    """
    Sets a train model with the default parameters for the Dutch cargo train

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param nb_carts: number of carts (default 1)

    :return: TrainModel
    """

    cargo_train = TrainModel()
    cargo_length = 12.55

    cargo_train.time = time
    cargo_train.velocities = velocities
    cargo_train.carts = [Cart() for _ in range(nb_carts)]
    cargo_train.cart_distances = [start_coord + i * cargo_length for i in range(nb_carts)]

    for cart in cargo_train.carts:
        cart.bogie_distances = [-3.745, 3.745]
        cart.inertia = 1558125/2
        cart.mass = 65e3/2
        cart.stiffness = 24e6
        cart.damping = 71e3
        cart.length = cargo_length
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = [-0.91, 0.91]
            bogie.mass = 9333.3/2
            bogie.inertia = 9.3e3/2
            bogie.stiffness = 2.232e6
            bogie.damping = 36.7e3
            bogie.length = 1.82
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = 2.5e3

    return cargo_train

def build_cargo_train(time: np.ndarray, velocities: np.ndarray, start_coord: float, locomotive: TrainType,
                      wagon: TrainType, nb_locomotives=1, nb_wagons=1):
    """
    Builds a cargo train model with locomotives and wagons

    :param time: all time steps
    :param velocities: velocity of the train
    :param start_coord: initial coordinate of the cart
    :param locomotive: type of locomotive
    :param wagon: type of wagon
    :param nb_locomotives: number of locomotives (default 1)
    :param nb_wagons: number of wagons (default 1)

    :return: TrainModel
    """

    # define locomotive
    if locomotive == TrainType.TRAXX:
        cargo_train = set_traxx_locomotive(time, velocities, start_coord, nb_carts=nb_locomotives)
    elif locomotive == TrainType.BR189:
        cargo_train = set_br189_locomotive(time, velocities, start_coord, nb_carts=nb_locomotives)
    else:
        raise ValueError(f"Locomotive {locomotive} not defined")

    # define wagon
    if wagon == TrainType.CARGO_TAPPS:
        wagon_length = 12.55
        wagon_train = set_cargo_TAPPS_train(time, velocities, start_coord, nb_carts=nb_wagons)
    elif wagon == TrainType.FALNS5:
        wagon_length = 15.79
        wagon_train = set_cargo_FALNS5_train(time, velocities, start_coord, nb_carts=nb_wagons)
    elif wagon == TrainType.SGNS:
        wagon_length = 19.74
        wagon_train = set_cargo_SGNS_train(time, velocities, start_coord, nb_carts=nb_wagons)
    else:
        raise ValueError(f"Wagon {wagon} not defined")

    cargo_train.cart_distances.extend([start_coord + wagon_length * (i + 1) for i in range(nb_wagons)])
    cargo_train.carts.extend([Cart() for _ in range(nb_wagons)])

    for i, cart in enumerate(cargo_train.carts):
        if i < nb_locomotives:
            # is the locomotive => skip
            continue

        # wagon props
        cart.bogie_distances = wagon_train.carts[0].bogie_distances
        cart.inertia = wagon_train.carts[0].inertia
        cart.mass = wagon_train.carts[0].mass
        cart.stiffness = wagon_train.carts[0].stiffness
        cart.damping = wagon_train.carts[0].damping
        cart.length = wagon_length
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for _ in range(len(cart.bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = wagon_train.carts[0].bogies[0].wheel_distances
            bogie.mass = wagon_train.carts[0].bogies[0].mass
            bogie.inertia = wagon_train.carts[0].bogies[0].inertia
            bogie.stiffness = wagon_train.carts[0].bogies[0].stiffness
            bogie.damping = wagon_train.carts[0].bogies[0].damping
            bogie.length = wagon_train.carts[0].bogies[0].length
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for _ in range(len(bogie.wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = wagon_train.carts[0].bogies[0].wheels[0].mass

    return cargo_train

# def set_rsmv_train(time, velocities, start_coord):
#     """
#     Sets a train model with the default parameters for the dutch intercity train
#     :return:
#     """
#
#     intercity_train = TrainModel()
#
#     intercity_train.time = time
#     intercity_train.velocities = velocities
#     intercity_train.cart_distances = [start_coord]
#     intercity_train.carts = [Cart()]
#
#     cart = intercity_train.carts[0]
#     cart.bogie_distances = [-10, 10]
#     cart.inertia = 128.8e3
#     cart.mass = 50e3
#     cart.stiffness = 2708e3
#     cart.damping = 64e3
#     cart.length = 28
#     cart.calculate_total_n_dof()
#
#     # setup bogies per cart
#     cart.bogies = [Bogie() for idx in range(len(cart.bogie_distances))]
#     for bogie in cart.bogies:
#         bogie.wheel_distances = [0]
#         bogie.mass = 6e3
#         bogie.inertia = 0.31e3
#         bogie.stiffness = 4800e3
#         bogie.damping = 0.25e3
#         bogie.length = 2.5
#         bogie.calculate_total_n_dof()
#
#         # setup wheels per bogie
#         bogie.wheels = [Wheel() for idx in range(len(bogie.wheel_distances))]
#         for wheel in bogie.wheels:
#             wheel.mass = 1.5e3
#
#     return intercity_train

# def train_class_to_dict(train_class:TrainModel):
#
#     train_dict = {}
#     train_dict["wheel_distances"] = train_class.carts[0].bogies[0].wheel_distances   # wheel distances from the centre of the bogie [m]
#     train_dict["bogie_length"] = train_class.carts[0].bogies[0].length  # length of the bogie [m]
#
#     # set up cart configuration
#     train_dict["bogie_distances"] = train_class.carts[0].bogie_distances  # bogie distances from the centre of the cart [m]
#     train_dict["cart_length"] = train_class.carts[0].length  # length of the cart [m]
#
#     # set up train configuration
#     train_dict["cart_distances"] = train_class.cart_distances  # cart distances from the start of the track [m]
#
#     # set train parameters
#     train_dict["mass_wheel"] = train_class.carts[0].bogies[0].wheels[0].mass  # mass of one wheel [kg]
#     train_dict["mass_bogie"] = train_class.carts[0].bogies[0].mass  # mass of one bogie [kg]
#     train_dict["mass_cart"] = train_class.carts[0].mass  # mass of one cart  [kg]
#
#     train_dict["inertia_bogie"] = train_class.carts[0].bogies[0].inertia  # mass inertia of one bogie   [kg.m2]
#     train_dict["inertia_cart"] = train_class.carts[0].inertia  # mass inertia of one cart   [kg.m2]
#
#     train_dict["prim_stiffness"] = train_class.carts[0].bogies[0].stiffness  # primary suspension: stiffness between wheels and bogie  [N/m]
#     train_dict["sec_stiffness"] = train_class.carts[0].stiffness  # secondary suspension: stiffness between bogies and cart  [N/m]
#
#     train_dict["prim_damping"] = train_class.carts[0].bogies[0].damping  # primary suspension: damping between wheels and bogie  [N.s/m]
#     train_dict["sec_damping"] = train_class.carts[0].damping
#
#     return train_dict