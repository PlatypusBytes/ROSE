
class TrainModel:
    def __init__(self):
        pass

    def optimisation_function_motion_eq(self):

        acceleration_component = mass_matrix*acceleration_c
        velocity_component = (1-alpha)*damping_matrix*velocity_c - alpha*damping_matrix*velocity_p
        displacement_component = (1+alpha)*stiffness_matrix*displacement_c - alpha*stiffness_matrix*displacement_p
        load_component = (1+alpha)*load_c - alpha*load_p

        temp = acceleration_component + velocity_component + displacement_component - load_component