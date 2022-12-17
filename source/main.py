from adts import Point, Writer, Robot
from extract_points import get_list_points_to_draw, find_contours
from utils import log, get_scale


### Hyperparameters ###

IMAGE_PATH = "images/test_draw_2.png"
SERIAL_PORT = "/dev/ttyUSB0"  # Others: COM4, ttyUSB1

# Debug
WRITE_TO_SERIAL = True
SHOW_CONTOURS_INFO = True

# Contour control
CONTOUR_MAX_ERROR = 10
JOIN_CONTOURS_THRESHOLD = 0.01  # Percentage of the cropped image diagonal

# Drawing control
ALLOW_LIFT_PEN = False
USE_ROLL = False
DRAWING_AREA = 10000  # mm^2
ELEVATION = 20  # mm
TIME_PER_POINT = 100  # ms


def main():
    writer = Writer(SERIAL_PORT, WRITE_TO_SERIAL)
    robot = Robot(writer)

    log("Extracting the relevant points from the image and creating the path")
    contours, is_closed, max_area = find_contours(
        IMAGE_PATH,
        CONTOUR_MAX_ERROR,
        JOIN_CONTOURS_THRESHOLD,
        SHOW_CONTOURS_INFO,
        ALLOW_LIFT_PEN,
    )
    scale = get_scale(DRAWING_AREA, max_area)
    points = get_list_points_to_draw(contours, is_closed, ELEVATION / scale)

    log("Retrieving starting point of the robot")
    starting_point = robot.get_starting_point(WRITE_TO_SERIAL, ELEVATION)

    log("Transfering points to robot's referencial")
    points = list(
        map(lambda point: point.flip_horizontally() * scale + starting_point, points)
    )

    points.append(starting_point + Point(0, 0, ELEVATION))

    log("Creating the trajectory (vector of points) to follow inside the robot")

    if USE_ROLL:
        points = robot.add_rolls(points)

    robot.create_vector_of_points(points, USE_ROLL)

    log("Starting the draw")
    writer.send_command(
        f"MOVES points 1 {len(points)} {round(TIME_PER_POINT*len(points))}"
    )

    log("Exiting")
    writer.close()


if __name__ == "__main__":
    main()
