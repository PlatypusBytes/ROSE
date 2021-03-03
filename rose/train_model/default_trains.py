from enum import Enum
from rose.train_model.train_model import TrainModel, Cart, Bogie, Wheel

import numpy as np


class TrainType(Enum):
    INTERCITY = 0
    SPRINTER = 1
    CARGO = 2

def set_train(time: np.ndarray, velocities: np.ndarray, start_coord: float, train_type: TrainType):
    """
    Sets a default train accordint to the TrainType

    :param time: all time steps
    :param velocities: velocities of the train per time step
    :param start_coord: initial coordinate of the middle of the cart
    :param train_type: type of train
    :return:
    """
    if train_type == TrainType.INTERCITY:
        return set_intercity_train(time, velocities, start_coord)
    elif train_type == TrainType.SPRINTER:
        return set_sprinter_train(time, velocities, start_coord)
    elif train_type == TrainType.CARGO:
        return set_cargo_train(time, velocities, start_coord)
    else:
        return None



def set_intercity_train(time, velocities, start_coord):
    """
    Sets a train model with the default parameters for the dutch intercity train
    :return:
    """

    intercity_train = TrainModel()

    intercity_train.time = time
    intercity_train.velocities = velocities
    intercity_train.cart_distances = [start_coord]
    intercity_train.carts = [Cart()]

    cart = intercity_train.carts[0]
    cart.bogie_distances = [-10, 10]
    cart.inertia = 128.8e3
    cart.mass = 50e3
    cart.stiffness = 2708e3
    cart.damping = 64e3
    cart.length = 28
    cart.calculate_total_n_dof()

    # setup bogies per cart
    cart.bogies = [Bogie() for idx in range(len(cart.bogie_distances))]
    for bogie in cart.bogies:
        bogie.wheel_distances = [-1.25, 1.25]
        bogie.mass = 6e3
        bogie.inertia = 0.31e3
        bogie.stiffness = 4800e3
        bogie.damping = 0.25e3
        bogie.length = 2.5
        bogie.calculate_total_n_dof()

        # setup wheels per bogie
        bogie.wheels = [Wheel() for idx in range(len(bogie.wheel_distances))]
        for wheel in bogie.wheels:
            wheel.mass = 1.5e3

    return intercity_train


def set_sprinter_train(time, velocities, start_coord):
    """
    Sets a train model with the default parameters for the dutch local sprinter train
    :return:
    """

    sprinter_train = TrainModel()

    sprinter_train.time = time
    sprinter_train.velocities = velocities
    sprinter_train.carts = [Cart()]
    sprinter_train.cart_distances = [start_coord]

    cart = sprinter_train.carts[0]
    cart.bogie_distances = [-10.3, 10.3]
    cart.inertia = 73.4e3
    cart.mass = 48.8e3
    cart.stiffness = 2468e3
    cart.damping = 50.2e3
    cart.length = 26.1
    cart.calculate_total_n_dof()


    # setup bogies per cart
    cart.bogies = [Bogie() for idx in range(len(cart.bogie_distances))]
    for bogie in cart.bogies:
        bogie.wheel_distances = [-1.25, 1.25]
        bogie.mass = 3.2e3
        bogie.inertia = 0.17e3
        bogie.stiffness = 4400e3
        bogie.damping = 0.59e3
        bogie.length = 2.5
        bogie.calculate_total_n_dof()

        # setup wheels per bogie
        bogie.wheels = [Wheel() for idx in range(len(bogie.wheel_distances))]
        for wheel in bogie.wheels:
            wheel.mass = 1.5e3

    return sprinter_train


def set_cargo_train(time, velocities, start_coord):
    """
    Sets a train model with the default parameters for the dutch cargo train
    :return:
    """

    cargo_train = TrainModel()

    cargo_train.time = time
    cargo_train.velocities = velocities
    cargo_train.carts = [Cart()]
    cargo_train.cart_distances = [start_coord]

    cart = cargo_train.carts[0]
    cart.bogie_distances = [-8.33, 8.33]
    cart.inertia = 784.2e3
    cart.mass = 65e3
    cart.stiffness = 3270e3
    cart.damping = 80.2e3
    cart.length = 21.7
    cart.calculate_total_n_dof()


    # setup bogies per cart
    cart.bogies = [Bogie() for idx in range(len(cart.bogie_distances))]
    for bogie in cart.bogies:
        bogie.wheel_distances = [-0.9, 0.9]
        bogie.mass = 5.2e3
        bogie.inertia = 2.1e3
        bogie.stiffness = 32700e3
        bogie.damping = 0.8e3
        bogie.length = 1.8
        bogie.calculate_total_n_dof()

        # setup wheels per bogie
        bogie.wheels = [Wheel() for idx in range(len(bogie.wheel_distances))]
        for wheel in bogie.wheels:
            wheel.mass = 1.5e3

    return cargo_train