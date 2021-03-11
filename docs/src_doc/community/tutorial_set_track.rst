
Tutorial set track
======================

In ROSE it is possible to create a custom schematisation for the track and subsoil. In this tutorial it is shown
how a simple track schematisation can be generated. ROSE does however not include a visual geometry editor. The user can
choose to use one of the default geometry creators. Or the user is required to create a script to create a custom geometry.

Set up default geometry
_______________________

A default geometry is not completely default, it is still possible to alter some input values.

1. The first step is to initialise the track system and define the time and solver options. In this example, a dynamic load
is simulated for 1 second in 5000 time steps. Below, it is shown how the track system is
initialised.

.. code-block:: python

    from rose.base.global_system import GlobalSystem
    from rose.solver.solver import NewmarkSolver
    import numpy as np

    # set calculation time
    time = np.linspace(0, 1, 5000)

    #choose solver
    solver = NewmarkSolver()

    # initialise global system
    track = GlobalSystem()
    track.time = time
    track.solver = solver

2. The next step is to define the variables. It is required to set the number of sleepers, the distance
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

    # add mesh to global system
    track.mesh = mesh
3. Below it is shown what the resulting element_model_parts dictionary represents:

.. code-block:: python

    # collect results from dictionary
    rail_model_part = element_model_parts["rail"]
    rail_pad_model_part = element_model_parts["rail_pad"]
    sleeper_model_part = element_model_parts["sleeper"]
    soil_model_part = element_model_parts["soil"]

4. The geometry is now generated, however the parameters are still to be assigned. Below it is shown which
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

5. Now that the geometry and parameters are set, it is required to set the boundary conditions. In this example, the
bottom boundary is set to a fixed boundary condition. The two end points of the rail are fixed in the x-direction. Below
it is shown how the boundary conditions can be set:

.. code-block:: python

    from rose.base.model_part import ConstraintModelPart
    from rose.utils.mesh_utils import add_no_displacement_boundary_to_bottom

    # constraint x displacement at the first and last rail model part node
    side_boundaries = ConstraintModelPart(x_disp_dof=False, y_disp_dof=True, z_rot_dof=True)
    side_boundaries.nodes = [rail_model_part.nodes[0], rail_model_part.nodes[-1]]

    # Fixate the bottom boundary
    bottom_boundary = add_no_displacement_boundary_to_bottom(soil_model_part)["bottom_boundary"]

6. The next step is to apply a load to the geometry. In this example, a single point load is added to the middle node of the
rail.

.. code-block:: python

    from rose.base.boundary_conditions import LoadCondition

    # initialise pointload and indicate which degrees of freedom are used
    point_load = LoadCondition(x_disp_dof=False, y_disp_dof=True, z_rot_dof=False)

    # Set the same value for the point load at each time step
    F = -10000
    point_load.y_force_matrix = np.ones((1, len(time))) * F

    # Define the time in the point load
    load.time = time

    # Indicate on which note the pointload should be applied, in this case the middle
    # node of the rail is selected
    load.nodes = [rail_model_part.nodes[50]]

7. Now that the geometry and boundary conditions are set, all the components can be combined in the global
system.

.. code-block:: python
    model_parts = [rail_model_part,rail_pad_model_part,sleeper_model_part, soil_model_part,side_boundaries,
                    bottom_boundary]

    track.model_parts = model_parts

8. Calculating the model can be done with the following command:
.. code-block:: python

    track.main()

9. Results of the model are projected on the nodes. In this example it is shown how to get the vertical displacement
on all the rail nodes for each time step.

.. code-block:: python

    # select the index of the desired degree of freedom (0 for horizontal displacement, 1 for vertical displacement, 2
    # for rotation around the z axis)
    dof_idx = 1

    # get all vertical displacements
    vertical_displacements = np.array([node.displacements[:, dof_idx] for node in rail_model_part.nodes])


