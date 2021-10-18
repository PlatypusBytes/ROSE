import threading

from dashboard.application import app
from dashboard.tests.utils import TestUtils

from flask import request

import json

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
    from datetime import timedelta

    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(seconds=5)

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



