import math
import csv
import numpy as np

def distance_calc(id_1, id_2) -> float:
    """
    Calculates distance between two hand landmarks

    :Returns float value
    """
    dist = math.pow((id_1["x"]-id_2["x"])**2 + (id_1["y"]-id_2["y"])**2, 0.5)

    return dist

def write_to_csv(data: list) -> None:
    """
    Writes each data obtained from one frame of hand recognition
    to csv

    :Returns None
    """
    with open("thumb_data.csv", 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['thumb', 'fing1', 'fing2', 'fing3', 'fing4'])
        writer.writerows(data)

def read_csv():
    """Returns an array of data retrieved beforehand

    """
    x = []

    with open("thumb_data.csv", 'r', newline='') as file:
        reader = csv.reader(file)

        for row in reader:
            x.append(row)

    return np.array(x)


    
