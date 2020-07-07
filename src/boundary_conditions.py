from src.model_part import ConditionModelPart

class NoDispRotCondition(ConditionModelPart):
    def __init__(self):
        super().__init__()
        self.rotation_dof = False
        self.x_disp_dof = False
        self.y_disp_dof = False

class CauchyCondition(ConditionModelPart):
    def __init__(self, rotation_dof=False, y_disp_dof=False, x_disp_dof=False):
        super(ConditionModelPart).__init__()
        self.rotation_dof = rotation_dof
        self.x_disp_dof = x_disp_dof
        self.y_disp_dof = y_disp_dof

        self.moment = []
        self.x_force = []
        self.y_force = []