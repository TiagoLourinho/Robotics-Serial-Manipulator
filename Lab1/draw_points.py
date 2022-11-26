from random import shuffle
from datetime import datetime
from time import perf_counter as time
from typing import Callable

from adts import Point, Writer


WRITE_TO_SERIAL = False


def log(message, end="\n"):
    print(f'[{str(datetime.now()).split(".")[0]}]: {message}', end=end)


def is_error(ans: str):
    """Checks whether the response of the robot represents an error"""

    return ans.strip() != "Done."


def create_vector_in_robot(points: list[Point]):
    """Creates a vector by following the steps in the special notes of the lab slides"""

    # Step 1.1
    writer.send_command(f"DIMP points[{len(points)}]")

    for i, point in enumerate(points):

        # Step 1.2
        writer.send_command(f"HERE points[{i+1}]")

        # Step 3 (preventing the creation of a temporary point outside the working zone)
        enc_coords = point.get_encoded_cartesian_coordinates()
        keys = list(enc_coords.keys())

        error = True
        while error:
            for coord, enc_value in [(key, enc_coords[key]) for key in keys]:

                # Step 1.3
                ans = writer.send_command(f"SETPVC points[{i+1}] {coord} {enc_value}")

                if is_error(ans):
                    shuffle(keys)
                    break
            else:
                error = False

        """ 
        # Step 4
        writer.send_command(f"TEACH points[{i+1}]")
        writer.send_command(f"MOVED points[{i+1}]")  
        writer.send_command(f"HERE points[{i+1}]") 
        """


def get_starting_point() -> Point:
    """Retrieves the initial point from the robot"""

    writer.send_command("DEFP cur")
    writer.send_command("HERE cur")
    tokens = writer.send_command("LISTPV cur").split()

    # Writing to a file
    if len(tokens) == 1:
        return Point()

    # Extract the values from the following string (after spliting):
    # 1:0     2:1791  3:2746   4:0      5:-1
    # X:1690  Y:0	  Z:6011   P:-636   R:-1
    coords = {
        key: int(tokens[i][tokens[i].index(":") + 1 :])
        for i, key in enumerate([1, 2, 3, 4, 5, "X", "Y", "Z", "P", "R"])
    }

    return Point(coords["X"], coords["Y"], coords["Z"], encoded=True)


def main():

    log("Extracting the relevant points from the image to draw")
    points = [
        Point(100, 0, 0),
        Point(100, 100, 0),
        Point(0, 0, 0),
        Point(0, 0, 50),
    ]

    log("Retrieving starting point of the robot")
    starting_point = get_starting_point()

    # Transform the points to the new reference frame
    points = list(filter(lambda point: point + starting_point, points))

    log(
        "Setting up by creating the trajectory (vector of points) to follow inside the robot"
    )
    create_vector_in_robot(points)

    log("Starting the draw")
    writer.send_command(f"MOVES points 1 {len(points)}")

    log("Exiting")
    writer.close()


if __name__ == "__main__":

    writer = Writer(write_to_serial=WRITE_TO_SERIAL)

    main()
