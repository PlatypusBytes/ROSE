from rose.pre_process.mesh_utils import *
import pytest

class TestMeshUtils:
    @pytest.mark.skip("work in progress")
    def test_create_horizontal_track(self):

        time = np.linspace(0, 10, 1000)

        element_model_parts, mesh = create_horizontal_track(3, 2, 1)
        bottom_boundary = add_no_displacement_boundary_to_bottom(
            element_model_parts["soil"]
        )
        load = add_moving_point_load_to_track(
            element_model_parts["rail"], time, 10, y_load=-15000
        )

        model_parts = [
            list(element_model_parts.values()),
            list(bottom_boundary.values()),
            list(load.values()),
        ]
        model_parts = list(itertools.chain.from_iterable(model_parts))

        pass
