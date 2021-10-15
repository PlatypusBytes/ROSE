from dashboard.application import app

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
        expected_res_dict={'data': {},
                           'exist': False,
                           # 'message': "Input file not valid; InfraMon input is not valid",
                           'message': "Input file not valid",
                           'running': False,
                           "valid": False}

        # assert results
        assert res_dict == expected_res_dict

