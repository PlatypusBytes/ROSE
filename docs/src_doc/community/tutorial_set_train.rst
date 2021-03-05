
Tutorial set train
======================
An import part of ROSE is the train model. It is possible to set up your own custom train
otherwise, it is also possible to choose from a selection of default trains

Set up default train model
__________________________

1. In order to set up a default train, it is required to specify the type. Below an example is given on how to select
an intercity train.

.. code-block:: python

    from rose.train_model.train_model.default_trains import TrainType

    train_type = TrainType.INTERCITY

2. Step two is to set the train. The only required additional input is the time and velocity.

.. code-block:: python

    from rose.train_model.train_model.default_trains import set_train
    import numpy as np

    time= np.linspace(0,1,10000) # [s]
    velocity = 40 # [m/s]
    velocities = np.ones(len(time)) * velocity

    train_model = set_train(time, velocities, train_type)


Set up custom train model
_________________________
An other possibility is to set a custom train, where each part of the train is customisable.

1. The first step is to import the train model.

.. code-block:: python

    from rose.train_model.train_model.train_model import TrainModel

    train_model = TrainModel()

2. The next step is to define the attributes of the train. The train consists of carts, bogies, wheels and dampers.
an example of how to define a cart is shown below:

.. code-block:: python

    from rose.train_model.train_model.train_model import Cart

    cart = Cart()
    cart.inertia = 128.8e3 # inertia of the cart [@@]
    cart.mass = 50e3 # mass of the cart [kg]
    cart.stiffness = 2708e3 # stiffness between cart and bogies [N/m]
    cart.damping = 64e3 # damping between cart and bogies [@@]
    cart.length = 28  # length of the cart
    cart.bogie_distances = [-10, 10] # list of horizontal distances from the mid of the cart to the mid of the bogies [m]

3. Each cart needs to be connected to 1 or more bogies. In the code below it is shown how the bogies are defined
and how they are connected to the cart. In this example 2 equal bogies are generated and connected to the cart

.. code-block:: python

    from rose.train_model.train_model.train_model import Bogie
    cart.bogies = [Bogie() for idx in range(len(cart.bogie_distances))]

    for bogie in cart.bogies:
        bogie.wheel_distances = [-1.25, 1.25] # list of horizontal distances from the mid of the bogie to the wheel connections [m]
        bogie.mass = 3.2e3 # mass of the bogie [kg]
        bogie.inertia = 0.17e3 # intertia of the bogie [@@]
        bogie.stiffness = 4400e3 # stiffness between bogie and wheels [N/m]
        bogie.damping = 0.59e3 # damping between bogie and wheels [@@]
        bogie.length = 2.5 # length of the bogie

4. When the bogies are defined, the wheels can be defined and attached to the bogies. Below an example is shown on how equal
wheels can be defined.

.. code-block:: python

    from rose.train_model.train_model.train_model import Wheel

    for bogie in cart.bogies:
        # Create a new wheels for the bogie
        bogie.wheels = [Wheel() for idx in range(len(bogie.wheel_distances))]

        # apply the same parameters to each wheel of the bogie
        for wheel in bogie.wheels:
            wheel.mass = 1.5e3 # mass of the wheel [kg]

5. Lastly the carts need to be connected to the train, the calculation time and the train velocity needs to be defined and
the starting point need to be defined.


.. code-block:: python

    import numpy as np

    # define time and velocity
    time= np.linspace(0,1,10000) # [s]
    velocity = 40 # [m/s]
    velocities = np.ones(len(time)) * velocity

    # apply parameters to the train
    train_model.carts = [cart] # list of all carts of the train
    train_model.cart_distances = [20.0] # list of starting distances (relative to the left side of the track model) of the middle of each cart
    train_model.time = time
    train_model.velocities = velocities
