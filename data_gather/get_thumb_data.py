import math
import csv

def distance_calc(id_1, id_2) -> float:
    """
    Calculates distance between two hand landmarks

    :Returns float value
    """
    dist = math.pow((id_1.x-id_2.x)**2 + (id_1.y-id_2.y)**2)

    return dist

def write_to_csv(data: list) -> None:
    """
    Writes each data obtained from one frame of hand recognition
    to csv

    :Returns None
    """
    with open("thumb_data.csv", 'w', encoding="UFT8") as file:
        writer = csv.writer(file)
    writer.writerow(data)


