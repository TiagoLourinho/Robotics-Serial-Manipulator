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


def get_list_points_to_draw(
    contours: list[np.array], is_closed: list[bool], elevation: int
):
    """Connects all the contours into an unique list of `Point`'s"""

    ret = []

    for i, cnt in enumerate(contours):
        # First point of the contour elevated
        ret.append(Point(*cnt[0][0], elevation))

        # All the points of the contour
        for pnt in cnt:
            ret.append(Point(*pnt[0], 0))

        if is_closed[i]:
            # First point of the contour (to close the line)
            ret.append(Point(*cnt[0][0], 0))

            # Move pen up
            ret.append(Point(*cnt[0][0], elevation))
        else:
            ret.append(Point(*cnt[-1][0], elevation))

    return ret


def draw_contours(img: np.array, contours: list[np.array], is_closed: list[bool]):
    """Draws the points from the contours in sequence"""

    copy_img = img.copy()

    # Get distinct colors
    num = len(contours)
    diagonal = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    HSV_tuples = [(x * 1.0 / (num + 1), 1.0, 1.0) for x in range(num)]
    RGB_tuples = (
        np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))) * 255
    )
    RGB_tuples = list(map(lambda x: (int(x[0]), int(x[1]), int(x[2])), RGB_tuples))

    for n_cnt, cnt in enumerate(contours):
        for i in range(len(cnt) - 1):
            cv.line(
                copy_img,
                tuple(cnt[i][0].reshape((2,))),
                tuple(cnt[i + 1][0].reshape((2,))),
                RGB_tuples[n_cnt],
                int(0.005 * diagonal),
            )
            cv.namedWindow("Drawing contours", cv.WINDOW_NORMAL)
            cv.imshow("Drawing contours", copy_img)
            cv.waitKey(500)

        if is_closed[n_cnt]:
            cv.line(
                copy_img,
                tuple(cnt[-1][0].reshape((2,))),
                tuple(cnt[0][0].reshape((2,))),
                RGB_tuples[n_cnt],
                int(0.005 * diagonal),
            )
            cv.namedWindow("Drawing contours", cv.WINDOW_NORMAL)
            cv.imshow("Drawing contours", copy_img)
            cv.waitKey(500)

    cv.destroyAllWindows()


def handle_open_contours(
    contours: list[np.array],
) -> list[dict[str : list[np.array] | bool]]:
    """Classify contours in open or close"""

    new_contours = []
    closed = []

    for cnt in contours:

        for i in range(1, len(cnt) - 1):

            # Open contour (goes back)
            if np.array_equal(cnt[i - 1], cnt[i + 1]):
                for divided_contour in divide_open_contour(cnt):
                    new_contours.append(divided_contour)
                    closed.append(False)

                break
        # Close contour
        else:
            new_contours.append(cnt)
            closed.append(True)

    return new_contours, closed


def divide_open_contour(contour: np.array) -> list[np.array]:
    """Divides an open contour into smaller open contours"""

    divided_contours = []
    visited = set()

    start_index = 0
    going_back = False

    for i in range(len(contour)):

        pnt = tuple(contour[i][0].reshape((2,)))

        if pnt not in visited:
            visited.add(pnt)

            if going_back:
                going_back = False
                start_index = i

        elif not going_back:
            divided_contours.append(contour[start_index:i])
            going_back = True

    if not going_back:
        divided_contours.append(contour[start_index : len(contour)])

    return divided_contours


def remove_duplicate_contours(
    skeleton: np.array, contours: list[np.array], is_closed: list[bool]
) -> tuple:
    """Removes points from contours that would create a duplicate line"""

    img = skeleton.copy()
    filtered_contours = [list() for _ in range(len(contours))]

    for n_cnt, cnt in enumerate(contours):

        for n_pnt in range(len(cnt) - 1):

            pnt1 = tuple(cnt[n_pnt][0].reshape((2,)))
            pnt2 = tuple(cnt[n_pnt + 1][0].reshape((2,)))

            rr, cc = line_nd(pnt1, pnt2, endpoint=True)

            # The line would make a difference in the image
            if np.sum(img[cc, rr] == 255):

                cv.line(
                    img,
                    pnt1,
                    pnt2,
                    0,
                    1,
                )
                filtered_contours[n_cnt].append(cnt[n_pnt])

                if n_pnt == len(cnt) - 2:
                    filtered_contours[n_cnt].append(cnt[n_pnt + 1])

    ret = {"contours": [], "is_closed": []}
    for i in range(len(filtered_contours)):
        if len(filtered_contours[i]):
            ret["contours"].append(np.array(filtered_contours[i]))
            ret["is_closed"].append(is_closed[i])

    return ret["contours"], ret["is_closed"]


def validate_contours(contours: list[np.array], is_closed: list[bool]) -> tuple:
    """Final check to assure that contours are valid"""

    new_contours = []
    new_is_closed = []

    for n_cnt, cnt in enumerate(contours):
        for n_pnt in range(len(cnt) - 1):

            # If the pixels are not neighbors
            if np.linalg.norm(cnt[n_pnt] - cnt[n_pnt + 1]) > np.sqrt(2):

                for n_pnt_inv in range(len(cnt) - 1, 0, -1):
                    if np.linalg.norm(cnt[n_pnt_inv] - cnt[n_pnt_inv - 1]) > np.sqrt(2):
                        break

                new_contours.append(np.vstack([cnt[n_pnt_inv:], cnt[0 : n_pnt + 1]]))
                new_contours.append(cnt[n_pnt + 1 : n_pnt_inv])
                new_is_closed.append(False)
                new_is_closed.append(False)
                break
        else:
            new_contours.append(cnt)
            new_is_closed.append(is_closed[n_cnt])

    return new_contours, new_is_closed


def find_contours(
    image: str,
    contour_max_error: float = 0.01,
    show_contours_info: bool = False,
):
    """Finds the contours of the `image` and reduces the number of points per contour (the returned contours are sorted by area)"""

    original_img = cv.imread(os.path.normpath(image))

    # Find the skeletonized image (1 pixel wide)
    skeleton_img = binarize(
        invert(cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)), threshold=127
    )
    skeleton_img = (skeletonize(skeleton_img) * 255).astype(np.uint8)

    # Find the contours
    contours, _ = cv.findContours(
        skeleton_img,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_NONE,
    )

    # Crop image
    contours = sorted(list(contours), key=lambda cnt: cv.contourArea(cnt), reverse=True)
    x, y, w, h = cv.boundingRect(contours[0])
    margin_x, margin_y = int(0.05 * w), int(0.05 * h)
    shape = original_img.shape
    original_img = original_img[
        max(0, y - margin_y) : min(shape[0], y + h + margin_y) + 1,
        max(0, x - margin_x) : min(shape[1], x + w + margin_x) + 1,
    ]
    skeleton_img = skeleton_img[
        max(0, y - margin_y) : min(shape[0], y + h + margin_y) + 1,
        max(0, x - margin_x) : min(shape[1], x + w + margin_x) + 1,
    ]

    for cnt in contours:
        cnt -= np.array([max(0, x - margin_x), max(0, y - margin_y)])

    # Filter contours

    contours, is_closed = handle_open_contours(contours)

    contours, is_closed = remove_duplicate_contours(skeleton_img, contours, is_closed)

    contours, is_closed = validate_contours(contours, is_closed)

    # Reduce the number of points per contour
    contours_reduced = [None] * len(contours)
    diagonal = np.sqrt(original_img.shape[0] ** 2 + original_img.shape[1] ** 2)
    for i, contour in enumerate(contours):

        contours_reduced[i] = cv.approxPolyDP(
            contour,
            contour_max_error * cv.arcLength(contour, is_closed[i]),
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

    return contours, is_closed
