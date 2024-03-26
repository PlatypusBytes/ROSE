from copy import deepcopy
import pytest
from rose.pre_process.default_trains import TrainType, set_train, build_cargo_train


TOL = 1e-6
# Define train configuration data as a dictionary for easier management
train_configs = {TrainType.DOUBLEDEKKER: {'cart_length': 24.1,
                                          'bogie_distances': [-9.95, 9.95],
                                          'inertia': 64.4e3,
                                          'mass': 25e3,
                                          'stiffness': 2708e3,
                                          'damping': 64e3,
                                          'bogie_properties': {'wheel_distances': [-1.25, 1.25],
                                                               'mass': 3e3,
                                                               'inertia': 0.155e3,
                                                               'stiffness': 4800e3,
                                                               'damping': 0.25e3,
                                                               'length': 2.5,
                                                               },
                                          'wheel_mass': 1.5e3,
                                          },
                TrainType.SPRINTER_SGM: {'cart_length': 25.15,
                                         'bogie_distances': [-9, 9],
                                         'inertia': 36.7e3,
                                         'mass': 23e3,
                                         'stiffness': 2468e3,
                                         'damping': 50.2e3,
                                         'bogie_properties': {'wheel_distances': [-1.25, 1.25],
                                                              'mass': 1.6e3,
                                                              'inertia': 0.085e3,
                                                              'stiffness': 4400e3,
                                                              'damping': 0.59e3,
                                                              'length': 2.5,
                                                              },
                                          'wheel_mass': 1.5e3
                                          },
                TrainType.SPRINTER_SLT: {'cart_length': 15.57,
                                         'bogie_distances': [-7.785, 7.785],
                                         'inertia': 36.7e3,
                                         'mass': 23e3,
                                         'stiffness': 2468e3,
                                         'damping': 50.2e3,
                                         'bogie_properties': {'wheel_distances': [-1.4, 1.4],
                                                              'mass': 1.6e3,
                                                              'inertia': 85,
                                                              'stiffness': 4400e3,
                                                              'damping': 0.59e3,
                                                              'length': 2.8,
                                                        },
                                'wheel_mass': 1.5e3
                                },
                TrainType.ICM: {'cart_length': 14.3,
                                'bogie_distances': [-7.7, 7.7],
                                'inertia': 91.5e3,
                                'mass': 34.3e3,
                                'stiffness': 10000e3,
                                'damping': 42.5e3,
                                'bogie_properties': {'wheel_distances': [-0.9, 0.9],
                                                     'mass': 1.495e3,
                                                     'inertia': 1.9e3,
                                                     'stiffness': 2612e3,
                                                     'damping': 52.2e3,
                                                     'length': 1.8,
                                               },
                                'wheel_mass': 0.6775e3
                                },
                TrainType.CARGO_FALNS5: {'cart_length': 15.79,
                                         'bogie_distances': [-5.335, 5.335],
                                         'inertia': 124139,
                                         'mass': 30e3,
                                         'stiffness': 2.37e8,
                                         'damping': 8.02e5,
                                         'bogie_properties': {'wheel_distances': [-0.91, 0.91],
                                                              'mass': 2.6e3,
                                                              'inertia': 1050,
                                                              'stiffness': 2.37e6,
                                                              'damping': 8.02e4,
                                                              'length': 1.82,
                                                              },
                                         'wheel_mass': 3e3,
                                         },
                TrainType.CARGO_SGNS: {'cart_length': 19.74,
                                          'bogie_distances': [-7.1, 7.1],
                                          'inertia': 392100,
                                          'mass': 14.74e3,
                                          'stiffness': 3.27e7,
                                          'damping': 8.02e5,
                                          'bogie_properties': {'wheel_distances': [-0.91, 0.91],
                                                               'mass': 2.5e3,
                                                               'inertia': 1.05e3,
                                                               'stiffness': 3.27e6,
                                                               'damping': 8.02e4,
                                                               'length': 1.82,
                                                               },
                                          'wheel_mass': 1.5e3,
                                          },
                TrainType.CARGO_TAPPS: {'cart_length': 12.55,
                                          'bogie_distances': [-3.745, 3.745],
                                          'inertia': 779062.5,
                                          'mass': 32.5e3,
                                          'stiffness': 24e6,
                                          'damping': 71e3,
                                          'bogie_properties': {'wheel_distances': [-0.91, 0.91],
                                                               'mass': 4666.65,
                                                               'inertia': 4.65e3,
                                                               'stiffness': 2.232e6,
                                                               'damping': 36.7e3,
                                                               'length': 1.82,
                                                               },
                                          'wheel_mass': 2.5e3,
                                          },
                TrainType.TRAXX: {'cart_length': 18.95,
                                          'bogie_distances': [-5.175, 5.175],
                                          'inertia': 779062.5,
                                          'mass': 27000,
                                          'stiffness': 24e6,
                                          'damping': 71e3,
                                          'bogie_properties': {'wheel_distances': [-1.3, 1.3],
                                                               'mass': 3.5e3,
                                                               'inertia': 4.6665e3,
                                                               'stiffness': 2.232e6,
                                                               'damping': 36.7e3,
                                                               'length': 2.6,
                                                               },
                                          'wheel_mass': 4e3,
                                          },
                TrainType.BR189: {'cart_length': 19.6,
                                          'bogie_distances': [-4.95, 4.95],
                                          'inertia': 796375,
                                          'mass': 27600,
                                          'stiffness': 24e6,
                                          'damping': 71e3,
                                          'bogie_properties': {'wheel_distances': [-1.45, 1.45],
                                                               'mass': 3.5e3,
                                                               'inertia': 4666.65,
                                                               'stiffness': 2.232e6,
                                                               'damping': 36.7e3,
                                                               'length': 2.9,
                                                               },
                                          'wheel_mass': 4e3,
                                          },
                }


def test_trains():
    """
    Test the passenger trains
    """
    time = 1
    velocity = 2
    start_coordinate = 10
    nb_carts = [1, 3]

    for n in nb_carts:
        for train_type, config in train_configs.items():
            if train_type == TrainType.SPRINTER_SLT and n == 1:
                # check for error message
                with pytest.raises(ValueError) as excinfo:
                    set_train(time, velocity, start_coordinate, train_type, nb_carts=n)
                assert str(excinfo.value) == "Number of carts must be at least 2 for the sprinter SLT train"
                continue
            # set the train
            train = set_train(time, velocity, start_coordinate, train_type, nb_carts=n)
            # validate the train
            _validate_carts(train, config, start_coordinate, n)


def test_build_cargo_trains():
    """
    Test the custom cargo trains
    """
    time = 1
    velocity = 2
    start_coordinate = 10

    # test only three locomotives
    cargo_train = build_cargo_train(time, velocity, start_coordinate,
                                    locomotive=TrainType.TRAXX, wagon=TrainType.CARGO_TAPPS,
                                    nb_locomotives=3, nb_wagons=0)

    _validate_carts(cargo_train, train_configs[TrainType.TRAXX], start_coordinate, 3)

    # test Traxx + three wagons
    cargo_train = build_cargo_train(time, velocity, start_coordinate,
                                    locomotive=TrainType.TRAXX, wagon=TrainType.CARGO_TAPPS,
                                    nb_locomotives=1, nb_wagons=3)

    assert(len(cargo_train.carts) == 4)
    loc = deepcopy(cargo_train)
    loc.carts = [loc.carts[0]]
    loc.cart_distances = [loc.cart_distances[0]]
    _validate_carts(loc, train_configs[TrainType.TRAXX], start_coordinate, 1)
    wag = deepcopy(cargo_train)
    wag.carts =wag.carts[1:]
    # correct distances
    aux = wag.cart_distances[1] - wag.cart_distances[0]
    wag.cart_distances = [i - aux for i in wag.cart_distances[1:]]
    _validate_carts(wag, train_configs[TrainType.CARGO_TAPPS], start_coordinate, 3)


def _validate_carts(train, config, start_coordinate, nb_carts):
    """
    Validate the train
    """
    assert (len(train.carts) == nb_carts)
    cart_distances = [start_coordinate + i * config['cart_length'] for i in range(len(train.carts))]

    for _, cart in enumerate(train.carts):
        assert cart.bogie_distances == config['bogie_distances']
        assert cart.length == config['cart_length']
        assert cart.inertia == config["inertia"]
        assert cart.mass == config["mass"]
        assert cart.stiffness == config["stiffness"]
        assert cart.damping == config["damping"]
        assert all(abs(a - b) < TOL for a, b in zip(train.cart_distances, cart_distances))

        for _, bogie in enumerate(cart.bogies):
            for prop, value in config['bogie_properties'].items():
                assert getattr(bogie, prop) == value

            for _, wheel in enumerate(bogie.wheels):
                assert wheel.mass == config['wheel_mass']

