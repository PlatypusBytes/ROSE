import numpy as np

from pathlib import Path


def write_krdz_coordinates_to_csv(filename, coordinates):

    with open(Path(filename + '.csv'), 'w') as f:
        for coord in coordinates:
            f.write(f'{coord[0]};{coord[1]}\n')


def read_krdz_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    res = {"coordinates": None}
    coords = []
    for line in lines:
        splitted_line = line.split()
        coords.append([float(splitted_line[1]), float(splitted_line[2])])

    coords = np.array(coords)
    res["coordinates"] = coords

    return res



fn1 = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ\Amsterdam_Utrecht_201811.KRDZ"
fn2 = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ\DenBosch_Eindhoven_201811.KRDZ"
fn3 = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ\Utrecht_DenBosch_201811.KRDZ"

res1 = read_krdz_file(fn1)
res2 = read_krdz_file(fn2)
res3 = read_krdz_file(fn3)

write_krdz_coordinates_to_csv(Path(fn1).stem, res1["coordinates"])
write_krdz_coordinates_to_csv(Path(fn2).stem, res1["coordinates"])
write_krdz_coordinates_to_csv(Path(fn3).stem, res1["coordinates"])

