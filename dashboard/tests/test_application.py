import pytest

from dashboard.application import app
from dashboard.tests.utils import TestUtils

import json
app.config["TESTING"] = True

def test_run_empty_file():

    with app.test_client() as c:

        # test file
        data = {'SOS_Segment_Input': None,
                'Sensar_Input': None,
                'Rila_Input': None,
                'InfraMon_Input': None,
                }

        # run runner
        rv = c.post("/runner", json=data)

        # get results
        res_dict = rv.get_json()

        # set expected results
        expected_res_dict = {'data': {},
                             'exist': False,
                              # 'message': "Input file not valid; InfraMon input is not valid",
                             'message': "Input file not valid",
                             'running': False,
                             "valid": False}

        # assert results
        assert res_dict == expected_res_dict

def test_valid_calculation():

    with open("test_data/test_rose_input.json", "r") as file:
        sos_segment_input = json.load(file)

    # test file
    data = {'SOS_Segment_Input': sos_segment_input,
            'Sensar_Input': None,
            'Rila_Input': None,
            'InfraMon_Input': None,
            }
    with app.test_client() as c:
        # run runner

        rv = c.post("/runner", json=data)

        # get results
        res_dict = rv.get_json()

        # set expected results
        expected_res_dict = {'data': {},
                             'exist': False,
                             'message': "Calculation running",
                             'running': True,
                             "valid": False}

        # assert dictionary
        assert res_dict == expected_res_dict

        # wait for all threads to finish
        TestUtils.wait_for_subthreads()

        # clean calculation project data
        TestUtils.delete_calculation_data('proj1')


def test_valid_calculation_ricardo():

    with open("test_data/test_rose_input.json", "r") as file:
        sos_segment_input = json.load(file)

    with open("test_data/test_ricardo_input.json", "r") as file:
        ricardo_input = json.load(file)

    # test file
    data = {'SOS_Segment_Input': sos_segment_input,
            'Sensar_Input': None,
            'Rila_Input': None,
            'InfraMon_Input': ricardo_input,
            }

    with app.test_client() as c:
        # run runner
        rv = c.post("/runner", json=data)

        # get results
        res_dict = rv.get_json()

        # set expected results
        expected_res_dict = {'data': {},
                             'exist': False,
                             'message': "Calculation running",
                             'running': True,
                             "valid": False}

        # assert dictionary
        assert res_dict == expected_res_dict

        # wait for all threads to finish
        TestUtils.wait_for_subthreads()

        # clean calculation project data
        TestUtils.delete_calculation_data('proj1')


def test_get_settlement():
    with open("test_data/test_proj/data.json", "r") as file:
        all_data = json.load(file)

    # start test client
    with app.test_client() as c:

        # set session data
        with c.session_transaction() as sess:
            sess['data'] = all_data

        # run get settlement and get output json
        rv = c.get("/settlement?time_index=50&value_type=cumulative_settlement_mean")
        output = rv.get_json()

        # set expected result
        expected_output_except_coord = {"features": [{"geometry": {"coordinates": [],
                                                                   "type": "LineString"},
                                                      "properties": {"segmentId": "Segment 1001",
                                                                     "value": 120.09},
                                                      "type": "Feature"},
                                                     {"geometry": {"coordinates": [],
                                                                   "type": "LineString"},
                                                      "properties": {"segmentId": "Segment 1002",
                                                                     "value": 124.42},
                                                      "type": "Feature"}
                                                     ],
                                        "type": "FeatureCollection"}

        # assert segment id and value of each feature in feature collection
        for i in range(len(output["features"])):

            assert output["features"][i]["properties"]["segmentId"] == \
                   expected_output_except_coord["features"][i]["properties"]["segmentId"]

            assert pytest.approx(output["features"][i]["properties"]["value"]) == \
                   expected_output_except_coord["features"][i]["properties"]["value"]
