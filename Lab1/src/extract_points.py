import numpy as np
import cv2 as cv
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from sklearn.preprocessing import binarize
import colorsys
from skimage.draw import line_nd

from adts import Point
from utils import log


BLACK = 0
WHITE = 255


def get_list_points_to_draw(
    contours: list[np.array], is_closed: list[bool], elevation: int
) -> list[Point]:
    """Connects all the contours into an unique list of `Point`'s considering also the elevation"""

    points = []

    for i, cnt in enumerate(contours):
        # First point of the contour elevated
        points.append(Point(*cnt[0][0], elevation))

        # Add all the points of the contour
        for pnt in cnt:
            points.append(Point(*pnt[0], 0))

        # If contour is closed close the line, otherwise just lift pen up
        if is_closed[i]:
            points.append(Point(*cnt[0][0], 0))
            points.append(Point(*cnt[0][0], elevation))

        else:
            points.append(Point(*cnt[-1][0], elevation))

    return points


def find_contours(
    image: str,
    contour_max_error: float,
    join_contours_threshold: float,
    show_contours_info: bool = False,
) -> tuple[list[np.array] | list[bool] | float]:
    """Finds the contours of the `image` and filters them"""

    original_img = cv.imread(os.path.normpath(image))
    original_img[original_img >= 127] = 255
    original_img[original_img < 127] = 0

    # Find the skeletonized image (1 pixel wide)
    skeleton_img = binarize(
        invert(cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)), threshold=127
    )
    skeleton_img = (skeletonize(skeleton_img) * WHITE).astype(np.uint8)

    # Find the contours
    contours, _ = cv.findContours(
        skeleton_img,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_NONE,
    )

    # Sort by area
    contours = sorted(list(contours), key=lambda cnt: cv.contourArea(cnt), reverse=True)

    # Crop image
    x, y, w, h = get_crop_info(original_img)
    original_img = original_img[y : y + h + 1, x : x + w + 1]
    skeleton_img = skeleton_img[y : y + h + 1, x : x + w + 1]

    # Adjust the contours to take into account the cropping
    for i in range(len(contours)):
        contours[i] = contours[i] - np.array([x, y])

    diagonal = np.sqrt(original_img.shape[0] ** 2 + original_img.shape[1] ** 2)

    # Filter contours
    contours, is_closed = classify_contours(contours)
    contours, is_closed = remove_duplicate_parts_in_contours(
        skeleton_img, contours, is_closed
    )
    contours, is_closed = join_open_contours(
        contours, is_closed, diagonal, join_contours_threshold
    )
    contours, is_closed = remove_point_contours(
        contours, is_closed, diagonal, join_contours_threshold
    )

    # Reduce the number of points per contour
    contours_reduced = [None] * len(contours)
    for i, cnt in enumerate(contours):

        contours_reduced[i] = cv.approxPolyDP(
            cnt,
            contour_max_error,
            is_closed[i],
        )

        if show_contours_info:
            copy_img = original_img.copy()
            for point in contours_reduced[i]:

                cv.circle(
                    copy_img,
                    tuple(point[0]),
                    int(0.005 * diagonal),
                    (0, 0, 255),
                    -1,
                )

            cv.namedWindow(f"Countour {i}", cv.WINDOW_NORMAL)
            cv.imshow(f"Countour {i}", copy_img)

    if show_contours_info:
        cv.waitKey(0)
        cv.destroyAllWindows()

    contours = contours_reduced

    log(
        f"Found {len(contours)} contours with a total of {sum(len(cnt) for cnt in contours)} points"
    )

    if show_contours_info:
        draw_contours(original_img, contours, is_closed)

    return contours, is_closed, w * h


def classify_contours(
    contours: list[np.array],
) -> tuple[list[np.array] | list[bool]]:
    """Classify contours in open or close

    Strategy:   A contour is open if one of it's points has the same point before and after it (because it goes back).
                Otherwise, it's closed
    """

    new_contours = []
    is_closed = []

    for cnt in contours:

        # Loop over contour points
        for i in range(1, len(cnt) - 1):

            # Point whose predecessor and sucessor are equal (open contour)
            if np.array_equal(cnt[i - 1], cnt[i + 1]):

                # Divide the open contour into smaller open contours
                for divided_contour in divide_open_contour(cnt):
                    new_contours.append(divided_contour)
                    is_closed.append(False)

                break

        # Close contour
        else:
            new_contours.append(cnt)
            is_closed.append(True)

    return new_contours, is_closed


def divide_open_contour(contour: np.array) -> list[np.array]:
    """Divides an open contour into smaller open contours

    In OpenCV, when a contour is open the returned points are repeated
    because it couldn't close the contour so it goes back

    Strategy:   Keep track of the visited points and check it it's going back or not
                A contour start is identified by finding the first not visited point while going back
                A contour end is identified by finding the first visited point while not going back
    """

    divided_contours = []

    visited_points = set()  # Keep track of visited points
    start_index = 0  # Start index of the divided contour
    going_back = False  # Flag to check if points are going back a path or not

    for i in range(len(contour)):

        point = tuple(contour[i][0].reshape((2,)))

        if point not in visited_points:
            visited_points.add(point)

            # If the point wasn't visited yet and it was going back,
            # then the start of the next contour was found
            if going_back:
                going_back = False
                start_index = i

        # If the point was already visited and it was not going back,
        # the end of the next contour was found
        elif not going_back:
            divided_contours.append(contour[start_index:i])
            going_back = True

    # Append the rest if it was meaningful (not going back)
    if not going_back:
        divided_contours.append(contour[start_index : len(contour)])

    return divided_contours


def remove_duplicate_parts_in_contours(
    skeleton: np.array, contours: list[np.array], is_closed: list[bool]
) -> tuple[list[np.array] | list[bool]]:
    """Removes points from contours that would create a duplicate line

    Strategy:   If the point from point A to point B would paint something new in the image, then point A is important
    """

    img = invert(skeleton).copy()
    filtered_contours = [list() for _ in range(len(contours))]

    for i, cnt in enumerate(contours):

        for j in range(len(cnt) - 1):

            pnt1 = tuple(cnt[j][0].reshape((2,)))
            pnt2 = tuple(cnt[j + 1][0].reshape((2,)))

            rr, cc = line_nd(pnt1, pnt2, endpoint=True)

            # If the line would make a diffence and paint something white (meaningful line)
            if np.sum(img[cc, rr] == BLACK):

                cv.line(
                    img,
                    pnt1,
                    pnt2,
                    WHITE,
                    1,
                )
                filtered_contours[i].append(cnt[j])

        # Check if the closing line would make a difference
        if is_closed[i]:
            pnt1 = tuple(cnt[-1][0].reshape((2,)))
            pnt2 = tuple(cnt[0][0].reshape((2,)))

            rr, cc = line_nd(pnt1, pnt2, endpoint=True)

            # If the line would make a diffence and paint something white (meaningful line)
            if np.sum(img[cc, rr] == BLACK):

                cv.line(
                    img,
                    pnt1,
                    pnt2,
                    WHITE,
                    1,
                )
                filtered_contours[i].append(cnt[-1])

    # Eliminate contours that were reduced to 0 points (completely duplicate)
    new_is_closed = [
        is_closed[i] for i, cnt in enumerate(filtered_contours) if len(cnt)
    ]
    filtered_contours = [np.array(cnt) for cnt in filtered_contours if len(cnt)]

    return validate_contours(filtered_contours, new_is_closed)


def validate_contours(
    contours: list[np.array], is_closed: list[bool]
) -> tuple[list[np.array] | list[bool]]:
    """Check if the contours resulted from the removal of duplicate parts are valid

    Strategy:   If while following the points of the contour there was a jump (2 pixels that are not neighbors)
                then the contour needs to be divided
                The start of the divided contour is found after the first jump
                The end of the divided contour is found after the second jump
    """

    new_contours = []
    new_is_closed = []

    for i, cnt in enumerate(contours):

        j = 0  # Variable to index contour points (need to use % to index to keep going around the contour)
        start_index = None  # Start index of the new contour (None means that there wasn't a jump yet)
        added_points = 0  # Number of points added
        while True:

            # All the points from the original contour were used to create new contours
            if added_points == len(cnt):
                break

            # If they are not neighbors (there was a jump)
            if np.linalg.norm(cnt[j % len(cnt)] - cnt[(j + 1) % len(cnt)]) > np.sqrt(2):

                # First jump
                if start_index is None:
                    start_index = j + 1
                    j += 1
                    continue

                sliced_contour = slice_contour(
                    cnt, start_index % len(cnt), (j + 1) % len(cnt)
                )
                added_points += len(sliced_contour)

                new_contours.append(sliced_contour)
                new_is_closed.append(False)

                start_index = j + 1

            # Valid contour (there wasn't a jump after looping over the contour)
            if j > len(cnt) and start_index is None:
                new_contours.append(cnt)
                new_is_closed.append(is_closed[i])
                break

            j += 1

    return new_contours, new_is_closed


def join_open_contours(
    contours: list[np.array],
    is_closed: list[bool],
    diagonal: float,
    join_contours_threshold: float,
) -> tuple[list[np.array] | list[bool]]:
    """Joins open contours that are close to each other

    Strategy:   If the end of one contour is close to the start of another contour, then join them and keep going
    """

    joined = [False] * len(contours)
    new_contours = []
    new_is_closed = []

    for i, cnt in enumerate(contours):

        if not joined[i]:
            if not is_closed[i]:

                # If a open and not joined contour was found
                # loop over the subsequent contours to check if its possible to join
                # and then update current contour

                current_cnt = cnt
                for j in range(i + 1, len(contours)):

                    if (
                        not joined[j]
                        and not is_closed[j]
                        and are_close(
                            np.array([current_cnt[-1], contours[j][0]]),
                            diagonal,
                            join_contours_threshold,
                        )
                    ):

                        current_cnt = np.append(current_cnt, contours[j], 0)

                        joined[j] = True

                new_contours.append(current_cnt)
                new_is_closed.append(
                    are_close(
                        np.array([current_cnt[0], current_cnt[-1]]),
                        diagonal,
                        join_contours_threshold,
                    )
                )

            else:
                new_contours.append(cnt)
                new_is_closed.append(is_closed[i])

    return new_contours, new_is_closed


def remove_point_contours(
    contours: list[np.array],
    is_closed: list[bool],
    diagonal: float,
    join_contours_threshold: float,
) -> tuple[list[np.array] | list[bool]]:
    """Remove contours which points are all close to each other

    Strategy:    If all the points are all close to the center point then remove that contour
    """

    new_contours = []
    new_is_closed = []

    for i, cnt in enumerate(contours):

        if not are_close(cnt, diagonal, join_contours_threshold):
            new_contours.append(cnt)
            new_is_closed.append(is_closed[i])

    return new_contours, new_is_closed


def are_close(points: list[np.array], diagonal: float, threshold: float) -> bool:
    """Check if all the pointss are close (inside of a circle centered in the mean and whose radius is defined by `diagonal` and `threshold`)"""

    center = np.mean(points, axis=0)
    radius = threshold * diagonal

    for pnt in points:
        if np.linalg.norm(center - pnt) > radius:
            return False

    return True


def get_crop_info(img: np.array) -> tuple[int]:
    """Get the top left point and the width and height to crop the image"""

    # Left barrier
    for i in range(img.shape[1]):
        if np.sum(img[:, i] == BLACK):
            break
    x = i

    # Top barrier
    for i in range(img.shape[0]):
        if np.sum(img[i, :] == BLACK):
            break
    y = i

    # Right barrier
    for i in range(img.shape[1] - 1, -1, -1):
        if np.sum(img[:, i] == BLACK):
            break
    w = i - x + 1

    # Bottom barrier
    for i in range(img.shape[0] - 1, -1, -1):
        if np.sum(img[i, :] == BLACK):
            break
    h = i - y + 1

    return x, y, w, h


def slice_contour(contour: np.array, start: int, end: int) -> np.array:
    """Slices contours taking into account relative position of start and end"""

    if start == end:
        return contour

    if start < end:
        return contour[start:end]

    if start > end:
        return np.append(contour[start:], contour[:end], 0)


def draw_contours(img: np.array, contours: list[np.array], is_closed: list[bool]):
    """Draws the points from the contours in sequence"""

    copy_img = img.copy()

    # Get distinct colors
    num = len(contours)
    diagonal = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    HSV_tuples = [(x * 1.0 / (num + 1), 1.0, 1.0) for x in range(num)]
    RGB_tuples = (
        np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))) * WHITE
    )
    RGB_tuples = list(map(lambda x: (int(x[0]), int(x[1]), int(x[2])), RGB_tuples))

    for i, cnt in enumerate(contours):
        for j in range(len(cnt) - 1):
            cv.line(
                copy_img,
                tuple(cnt[j][0].reshape((2,))),
                tuple(cnt[j + 1][0].reshape((2,))),
                RGB_tuples[i],
                max(1, int(0.005 * diagonal)),
            )
            cv.namedWindow("Drawing contours", cv.WINDOW_NORMAL)
            cv.imshow("Drawing contours", copy_img)
            cv.waitKey(500)

        if is_closed[i]:
            cv.line(
                copy_img,
                tuple(cnt[-1][0].reshape((2,))),
                tuple(cnt[0][0].reshape((2,))),
                RGB_tuples[i],
                max(1, int(0.005 * diagonal)),
            )
            cv.namedWindow("Drawing contours", cv.WINDOW_NORMAL)
            cv.imshow("Drawing contours", copy_img)
            cv.waitKey(500)

    cv.destroyAllWindows()
    cv.namedWindow("Final result", cv.WINDOW_NORMAL)
    cv.imshow("Final result", copy_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
