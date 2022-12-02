from adts import Point, Writer, Robot
from extract_points import get_list_points_to_draw, find_contours
from utils import log, get_scale

### Hyperparameters ###

IMAGE_PATH = "Lab1/images/house.png"
SERIAL_PORT = "/dev/ttyUSB0"  # Others: COM4, ttyUSB1

WRITE_TO_SERIAL = False
SHOW_CONTOURS_INFO = True

CONTOUR_MAX_ERROR = 0.01
DRAWING_AREA = 5000  # mm^2
ELEVATION = 10  # mm
TIME_PER_POINT = 150  # ms


def main():
    writer = Writer(SERIAL_PORT, WRITE_TO_SERIAL)
    robot = Robot(writer)

    log("Extracting the relevant points from the image and creating the path")
    contours, is_closed = find_contours(
        IMAGE_PATH, CONTOUR_MAX_ERROR, SHOW_CONTOURS_INFO
    )
    scale = get_scale(DRAWING_AREA, contours[0])
    points = get_list_points_to_draw(contours, is_closed, ELEVATION / scale)

    log("Retrieving starting point of the robot")
    starting_point = robot.get_starting_point(WRITE_TO_SERIAL)

    log("Transfering points to robot's referencial")
    points = list(map(lambda point: point * scale + starting_point, points))

    points.insert(0, starting_point + Point(0, 0, ELEVATION))
    points.append(starting_point + Point(0, 0, ELEVATION))

    log("Creating the trajectory (vector of points) to follow inside the robot")
    robot.create_vector_of_points(points)

    log("Starting the draw")
    writer.send_command(
        f"MOVES points 1 {len(points)} {int(TIME_PER_POINT*len(points))}"
    )

    log("Exiting")
    writer.close()


if __name__ == "__main__":
    main()
