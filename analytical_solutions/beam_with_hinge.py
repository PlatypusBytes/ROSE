
class BeamWithHinge():
    """
    This class contains a beam which is connected with a rigid support at the left, and a hinge support at the left.
    Also, a hinge is present in somewhere in the middle of the beam.

    |
    |-------------o--------------⧋
    |

    """

    def __init__(self,left_l, right_l, EI, load):
        self.left_length = left_l
        self.right_length = right_l
        self.EI = EI
        self.load = load

    def __calculate_left_part(self,x):
        """
        Calculates left part of the beam as a cantilever
        :param x:
        :return:
        """
        delta_max = self.load*x**2/(6*self.EI) * (3*self.left_length - x)
        return delta_max

    def __calculate_right_part(self,x):
        """
        Calculates right part of the beam as a simply supported beam

        :param x:
        :return:
        """
        load_at_hinge = (1-x/self.right_length) * self.load
        delta_max = load_at_hinge*self.left_length**3/(3*self.EI)
        return delta_max

    def calculate_max_disp(self, x):
        """
        Calculates max displacement at the beam, i.e. disp at the hinge, for a load on x.
        :param x:
        :return:
        """
        if 0< x < self.left_length:
            delta_max = self.__calculate_left_part(x)
        elif self.left_length <= x <= self.right_length + self.left_length:
            delta_max = self.__calculate_right_part(x - self.left_length)
        else:
            delta_max = 0

        return delta_max
