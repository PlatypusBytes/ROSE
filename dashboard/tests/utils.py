import shutil
from pathlib import Path
import json
import threading

class TestUtils():

    @staticmethod
    def delete_calculation_data(project_name):
        """
        Deletes calculation data and removes hash from calculations.json
        :param project_name:
        :return:
        """

        # remove calculation data
        calculation_path = Path("../dash_calculations", project_name)
        if calculation_path.exists():
            shutil.rmtree(calculation_path)

        # read calculations.json
        calculations_json = Path("../dash_calculations/calculations.json")
        with open(calculations_json, "r") as json_file:
            calculation_data = json.load(json_file)

        # remove current project from calculations.json
        hash_key = list(calculation_data.keys())[list(calculation_data.values()).index(project_name)]
        calculation_data.pop(hash_key, None)

        # rewrite calculations.json
        with open(calculations_json, "w") as json_file:
            json.dump(calculation_data, json_file)

    @staticmethod
    def wait_for_subthreads():
        """
        Waits for sub-threads to finish
        :return:
        """
        for thread in threading.enumerate():
            if thread.__class__ is threading.Thread:
                thread.join()






