
import numpy as np

class ModelPart:
    """
    The model part has to consist of the same element types
    """
    def __init__(self):
        self.name = ""
        self.nodes = np.array([])
        self.elements = np.array([])
        self.rotation_dof = False
        self.x_disp_dof = False
        self.y_disp_dof = False

    def initialize(self):
        pass

    def set_geometry(self):
        pass


class ElementModelPart(ModelPart):
    def __init__(self):
        super(ElementModelPart, self).__init__()
        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None
        self.aux_mass_matrix = None

    def initialize(self):
        self.set_aux_stiffness_matrix()
        self.set_aux_mass_matrix()

        # import that damping matrix is set last, as rayleigh damping needs mass and stiffness
        self.set_aux_damping_matrix()


    def set_aux_stiffness_matrix(self):
        pass

    def set_aux_damping_matrix(self):
        pass

    def set_aux_mass_matrix(self):
        pass


class ConditionModelPart(ModelPart):
    def __init__(self):
        super(ConditionModelPart, self).__init__()


class ConstraintModelPart(ConditionModelPart):
    def __init__(self, rotation_dof=False, y_disp_dof=False, x_disp_dof=False):
        super(ConstraintModelPart, self).__init__()
        self.rotation_dof = rotation_dof
        self.x_disp_dof = x_disp_dof
        self.y_disp_dof = y_disp_dof


    def set_scalar_condition(self):
        pass

    def set_constraint_condition(self):
        for node in self.nodes:
            node.rotation_dof = self.rotation_dof
            node.x_disp_dof = self.x_disp_dof
            node.y_disp_dof = self.y_disp_dof
