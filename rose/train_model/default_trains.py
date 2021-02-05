
from rose.train_model.train_model import TrainModel, Cart, Bogie, Wheel

def set_intercity_train(time, velocities):
    """
    Sets a train model with the default parameters for the dutch intercity train
    :return:
    """

    intercity_train = TrainModel()

    intercity_train.time = time
    intercity_train.velocities = velocities
    intercity_train.carts = [Cart()]

    cart = intercity_train.carts[0]
    cart.bogie_distances = [-10, 10]
    cart.inertia = 128.8e3
    cart.mass = 75.5e3
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
            wheel.mass = 3.5e3

    return intercity_train


def set_sprinter_train(time, velocities):
    """
    Sets a train model with the default parameters for the dutch local sprinter train
    :return:
    """

    intercity_train = TrainModel()

    intercity_train.time = time
    intercity_train.velocities = velocities
    intercity_train.carts = [Cart()]

    cart = intercity_train.carts[0]
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

    return intercity_train