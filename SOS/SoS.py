import json
import matplotlib.pylab as plt


class ReadSosScenarios:
    def __init__(self, profile_filename: str, segment_filename: str) -> None:
        """
        Reads the SOS csv files. Plots and creates json file

        :param profile_filename: CSV file with profile
        :param segment_filename: CSV file with segments
        """
        with open(profile_filename, "r") as f:
            data = f.read().splitlines()
        self.profiles = [d.split(";") for d in data[1:]]

        with open(segment_filename, "r") as f:
            data = f.read().splitlines()
        self.segments = [d.split(";") for d in data[1:]]

        return






ReadSosScenarios(r"N:\Projects\11203500\11203592\B. Measurements and calculations\TKI Data\Output\soilprofiles.csv",
                 r"N:\Projects\11203500\11203592\B. Measurements and calculations\TKI Data\Output\segments.csv")
