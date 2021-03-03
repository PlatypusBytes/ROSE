from rose.SOS import SoS

sos = SoS.ReadSosScenarios(r"N:\Projects\11203500\11203592\B. Measurements and calculations\TKI Data\Output\soilprofiles.csv",
                           r"N:\Projects\11203500\11203592\B. Measurements and calculations\TKI Data\Properties\20201102_Prorail_parameters_SOS.csv",
                           r"N:\Projects\11203500\11203592\B. Measurements and calculations\TKI Data\Output\segments.csv",
                           r"N:\Projects\11204500\11204953\B. Measurements and calculations\SOS\TKI Data\Segmenten\Segments_TKI_v2.shp")
sos.create_segments()
sos.dump("./SOS/SOS.json")
sos.plot_sos(output_folder="./SOS/results")
