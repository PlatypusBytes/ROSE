import pytest

from src.train_model.train_model import TrainModel


class TestTrainModel:

    @pytest.mark.workinprogress
    def test_cart(self):

        train_model = TrainModel()
        train_model.mass_cart = 77000
        train_model.mass_bogie = 1100
        train_model.mass_wheel = 1200
        train_model.inertia_cart = 1.2e6
        train_model.inertia_bogie = 760
        train_model.prim_stiffness = 2.14e6
        train_model.sec_stiffness = 5.32e6
        train_model.prim_damping = 4.9e4
        train_model.sec_damping = 7e4

        train_model.length_cart = 3
        train_model.length_bogie = 1


#
#     def test_stiffness_matrix_track(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
#
#         utrack = UTrack(3)
#
#         utrack.rail = set_up_rail
#         utrack.sleeper = set_up_sleeper
#         utrack.rail_pads = set_up_rail_pad
#         utrack.soil = set_up_soil
#
#         utrack.set_geometry()
#         utrack.calculate_n_dofs()
#
#         utrack.set_global_stiffness_matrix()
#
#         plt.spy(utrack.global_stiffness_matrix)
#         plt.show()
#
#     def test_stiffness_matrix_global(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
#
#         n_sleepers = 3
#
#         global_system = GlobalSystem(n_sleepers)
#
#         utrack = UTrack(n_sleepers)
#         utrack.rail = set_up_rail
#         utrack.sleeper = set_up_sleeper
#         utrack.rail_pads = set_up_rail_pad
#
#         global_system.track = utrack
#         global_system.soil = set_up_soil
#
#         global_system.set_geometry()
#
#         global_system.set_global_stiffness_matrix()
#
#         utrack.set_geometry()
#         utrack.calculate_n_dofs()
#
#         utrack.set_global_stiffness_matrix()
#
#         plt.spy(utrack.global_stiffness_matrix)
#         plt.show()
#
#
#     def test_mass_matrix_track(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
#         utrack = UTrack(3)
#
#         utrack.rail = set_up_rail
#         utrack.sleeper = set_up_sleeper
#         utrack.rail_pads = set_up_rail_pad
#         utrack.soil = set_up_soil
#
#         utrack.set_geometry()
#         utrack.calculate_n_dofs()
#         utrack.set_global_mass_matrix()
#
#         plt.spy(utrack.global_mass_matrix)
#         plt.show()
#
#     def test_damping_matrix_track(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
#         utrack = UTrack(3)
#
#         utrack.rail = set_up_rail
#         utrack.sleeper = set_up_sleeper
#         utrack.rail_pads = set_up_rail_pad
#         utrack.soil = set_up_soil
#
#         utrack.set_geometry()
#         utrack.calculate_n_dofs()
#
#         utrack.set_global_stiffness_matrix()
#         utrack.set_global_mass_matrix()
#         utrack.set_global_damping_matrix()
#
#         plt.spy(utrack.global_damping_matrix)
#         plt.show()
#
#     def test_force_array(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
#         utrack = UTrack(3)
#
#         utrack.rail = set_up_rail
#         utrack.sleeper = set_up_sleeper
#         utrack.rail_pads = set_up_rail_pad
#         utrack.soil = set_up_soil
#
#
#         utrack.set_geometry()
#         utrack.calculate_n_dofs()
#
#         utrack.set_force()
#
#         utrack.apply_no_disp_boundary_condition()
#
#         plt.spy(utrack.force)
#         plt.show()
#
#     def test_calculate_point_load(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
#         utrack = UTrack(3)
#
#         utrack.rail = set_up_rail
#         utrack.sleeper = set_up_sleeper
#         utrack.rail_pads = set_up_rail_pad
#         utrack.soil = set_up_soil
#
#         utrack.calculate_n_dofs()
#
#         utrack.set_global_stiffness_matrix()
#         utrack.set_global_mass_matrix()
#         utrack.set_global_damping_matrix()
#
#         utrack.set_force()
#
#         utrack.apply_no_disp_boundary_condition()
#         solver = Solver(utrack.n_dof_track-3)
#
#         settings_newmark = {"gamma": 0.5,
#                             "beta": 0.25}
#
#         time = utrack.time
#
#         solver.newmark(settings_newmark, utrack.global_mass_matrix, utrack.global_damping_matrix, utrack.global_stiffness_matrix, utrack.force, time[1] - time[0], time[-1], t_start=time[0])
#
#
#         plt.plot(solver.time, solver.u[:, 1])
#         plt.plot(solver.time, solver.u[:, 4])
#         plt.plot(solver.time, solver.u[:, 7])
#         plt.plot(solver.time, solver.u[:, 9])
#         plt.plot(solver.time, solver.u[:, 10])
#         plt.plot(solver.time, solver.u[:, 11])
#
#         plt.legend(["1","4","7","9","10","11"])
#         plt.show()
#
#
#
#
#
# @pytest.fixture
# def set_up_soil():
#     soil = Soil()
#     soil.stiffness = 300e6
#     soil.damping = 0
#     return soil
#
#
#
# # @pytest.fixture()
# # def set_up_rail():
