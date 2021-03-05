
Tutorial set track
======================

In ROSE it is possible to create a custom schematisation for the track and subsoil. In this tutorial it is shown
how a simple track schematisation can be generated. ROSE does however not include a visual geometry editor. The user can
choose to use one of the default geometry creators. Or the user is required to create a script to create a custom geometry.

Set up default geometry
_______________________

A default geometry is not completely default, it is still possible to alter some input values.

1. Therefore the first step is to define the variables. It is required to set the number of sleepers, the distance
between sleepers, and the depth of the soil. Below an example is shown on how to define
the variables and create a horizontal track geometry.

.. code-block:: python

    from rose.utils.mesh_utils import create_horizontal_track

    n_sleepers = 100 # number of sleepers [-]
    sleeper_distance = 0.6 # distance between sleepers [m]
    depth_soil = 1 # depth of the soil [m]

    # create horizontal track
    element_model_parts, mesh = create_horizontal_track(
    n_sleepers, sleeper_distance, depth_soil)

2. Below it is shown what the resulting element_model_parts dictionary represents:

.. code-block:: python

    # collect results from dictionary
    rail_model_part = element_model_parts["rail"]
    rail_pad_model_part = element_model_parts["rail_pad"]
    sleeper_model_part = element_model_parts["sleeper"]
    soil_model_part = element_model_parts["soil"]

3. The geometry is now generated, however the parameters are still to be assigned. Below it is shown which
parameters need to be assigned to the element model parts.

.. code-block:: python

    from rose.base.model_part import Material, Section

    # set up rail material
    material = Material()
    material.youngs_modulus = 210e9 # youngs modulus of the rail [N/m2]
    material.poisson_ratio = 0.0 # poisson ratio of the beam [-]
    material.density = 7860 # density of the beam [kg/m3]

    # set up rail section
    section = Section()
    section.area = 69.6e-2 # section area of the rail [m2]
    section.sec_moment_of_inertia = # second moment of inertia [m4]
    section.shear_factor = 0 # 0 for euler beam [-]

    # set up rail
    rail_model_part.material = material
    rail_model_part.section = section

    # set up rail pad
    rail_pad_model_part.mass = 5 # mass of the rail pad[kg]
    rail_pad_model_part.stiffness = 750e6 # stiffness of the rail pad [N/m2]
    rail_pad_model_part.damping = 750e3 #damping of the rail pad [Ns/m2]

    # set up sleeper
    sleeper_model_part.mass = 140 # mass of the sleeper [kg]

    # set up soil
    soil_model_part.stiffness = 180e7 # stiffness of the soil [N/m2]
    soil_model_part.damping = 180e6 # damping of the soil [Ns/m2]
