from datetime import datetime

from adts import Point, Writer, Robot
from extract_points import get_list_points_to_draw

IMAGE_PATH = "Lab1/images/house.png"
SERIAL_PORT = "/dev/ttyUSB0"  # COM4
WRITE_TO_SERIAL = True
SHOW_CONTOURS = False
ELEVATION = 10
CONTOUR_MAX_ERROR = 0.01
SPEED = 5
PIXEL_TO_MM = 0.03
SLEEPING_TIME = 0


def log(message, end="\n"):
    """Prints a message with a timestamp"""

    print(f'[{str(datetime.now()).split(".")[0]}]: {message}', end=end)


def main():
    writer = Writer(SERIAL_PORT, WRITE_TO_SERIAL, SLEEPING_TIME)
    robot = Robot(writer, SPEED)

    log("Extracting the relevant points from the image to draw")
    points = get_list_points_to_draw(
        IMAGE_PATH, CONTOUR_MAX_ERROR, ELEVATION / PIXEL_TO_MM, SHOW_CONTOURS
    )

    log("Retrieving starting point of the robot")
    starting_point = robot.get_starting_point(WRITE_TO_SERIAL)

    log("Transfering points to robot's referencial")
    points = list(map(lambda point: point * PIXEL_TO_MM + starting_point, points))

    points.insert(0, starting_point + Point(0, 0, ELEVATION))
    points.append(starting_point + Point(0, 0, ELEVATION))

    log(
        "Setting up by creating the trajectory (vector of points) to follow inside the robot"
    )
    robot.create_vector_of_points(points)

    log("Starting the draw")
    writer.send_command(f"MOVES points 1 {len(points)}")

    log("Exiting")
    writer.close()


if __name__ == "__main__":
    main()
