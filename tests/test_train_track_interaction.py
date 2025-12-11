import numpy as np
from unittest.mock import MagicMock

from rose.model.train_track_interaction import CoupledTrainTrack


class TestTrainTrackInteraction:

    def test_get_interaction_forces_history(self):
        """
        Test the calculation of interaction forces history between train and track.

        """
        mock_solver = MagicMock()
        mock_solver.F_out = np.array([[10, 20, 30], [40, 50, 60]])

        mock_train = MagicMock()
        mock_train.wheels = [MagicMock(mass=5), MagicMock(mass=10)]
        mock_train.contact_dofs = [0, 2]
        coupled_system = CoupledTrainTrack()
        coupled_system.solver = mock_solver
        coupled_system.train = mock_train
        coupled_system.g = 9.81

        result = coupled_system.get_interaction_forces_history()

        expected = -9.81 * np.array([5, 10]) - np.array([[10, 30], [40, 60]])
        np.testing.assert_array_almost_equal(result, expected)