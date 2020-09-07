from src.model_part import RodElementModelPart


class Soil(RodElementModelPart):
    def __init__(self):
        super(Soil, self).__init__()

        self.stiffness = None
        self.damping = None

        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None
        self.aux_mass_matrix = None

        self.nodal_ndof = 1

    # def set_1_d_geometry(self, top_nodes, bottom_nodes):
    #     self.nodes = np.append(self.nodes, top_nodes)
    #     self.nodes = np.append(self.nodes, bottom_nodes)
    #
    #     soil_elements = []
    #     for i in range(len(top_nodes)):
    #         element = Element()
    #         element.index = len(self.elements) + i
    #         element.nodes = [top_nodes[i], bottom_nodes[i]]
    #         element.add_model_part("SOIL")
    #         soil_elements.append(element)
    #
    #     self.elements = np.append(self.elements, soil_elements)
