import pytest
import os
import numpy as np

from rose.utils.Hmax import HrsmMax
from rose.utils.random_field import create_rf


TEST_PATH = "tests"


class TestUtils:
    """
    Test the utility functions in the utils module.
    """

    def test_Hmax(self):
        """
        Test the Hmax calculation against the MATLAB implementation.
        """

        with open(os.path.join(TEST_PATH, "./test_data/Hrms.txt"), "r") as fi:
            sig = fi.read().splitlines()

        sig = list(map(float, sig))
        dx = 0.25

        # run
        h = HrsmMax(np.array(sig), dx)

        rms_band_matlab = np.array([2087.55705457531, 1139.29553877343, 793.047457091564, 548.095561097181,
                                    648.015015656438, 521.608760233929, 790.563948013129, 886.097705321285,
                                    1342.44544888507])
        h_max_matlab = np.array([6557.20346546820, 3764.49040821894, 2505.54201229720, 1727.21064286318,
                                 1208.73808675389, 1182.77849807824, 1996.50964604940, 2682.54554936051,
                                 3681.41028314498])
        h_max_dx_matlab = np.array([293.263231128877, 783.641358934000, 1300.63422139907, 524.680132894043,
                                    139.978582308885, 583.726452878631, 662.410999843909, 1240.10908108192,
                                    1127.56942541745])

        # test against matlab
        tol = 1e-3
        assert all((h.rms_bands - rms_band_matlab) / rms_band_matlab <= tol)
        assert all((h.max_fast - h_max_matlab) / h_max_matlab <= tol)
        assert all((h.max_fast_Dx - h_max_dx_matlab) / h_max_dx_matlab <= tol)


    def test_normal_random_field_statistics(self):
        """
        Test normal random field statistics.
        """
        # Setup
        mean = 5.0
        cov = 0.3
        nodes = np.linspace(0, 1000, 10000)

        # Execute
        rf = create_rf(mean, cov, 1.0, 0.0, nodes, seed=14)

        # Verify
        assert rf.shape == nodes.shape
        assert np.isclose(np.mean(rf), mean, rtol=0.1)
        assert np.isclose(np.std(rf), mean * cov, rtol=0.1)

    def test_lognormal_random_field_statistics(self):
        """
        Test log-normal random field statistics.
        """
        # Setup
        mean = 5.0
        cov = 0.3
        nodes = np.linspace(0, 1000, 10000)

        # Execute
        rf = create_rf(mean, cov, 1.0, 0.0, nodes, seed=14, log_normal=True)

        # Verify
        assert rf.shape == nodes.shape
        assert np.isclose(np.mean(rf), mean, rtol=0.15)
        empirical_cov = np.std(rf) / np.mean(rf)
        assert np.isclose(empirical_cov, cov, rtol=0.15)
        assert np.all(rf > 0)
