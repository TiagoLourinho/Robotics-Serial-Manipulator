import numpy as np
import cv2 as cv
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from sklearn.preprocessing import binarize


from adts import Point
from utils import log


def get_list_points_to_draw(contours: list[np.array], elevation: int):
    """Connects all the contours into an unique list of `Point`'s"""

    ret = []

    for cnt in contours:
        # First point of the contour elevated
        ret.append(Point(cnt[0][0][0], cnt[0][0][1], elevation))

        # All the points of the contour
        for pnt in cnt:
            ret.append(Point(pnt[0][0], pnt[0][1], 0))

        # First point of the contour (to close the line)
        ret.append(Point(cnt[0][0][0], cnt[0][0][1], 0))

        # Move pen up
        ret.append(Point(cnt[0][0][0], cnt[0][0][1], elevation))

    return ret


def find_contours(
    image: str, contour_max_error: float = 0.01, show_contours: bool = False
):
    """Finds the contours of the `image` and reduces the number of points per contour (the returned contours are sorted by area)"""

    original_img = cv.imread(os.path.normpath(image))

    # Find the skeletonized image (1 pixel wide)
    inverted_img = binarize(
        invert(cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)), threshold=127
    )
    skeleton_img = (skeletonize(inverted_img) * 255).astype(np.uint8)

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
    original_img = original_img[
        max(0, y - margin_y) : min(y + h + margin_y) + 1,
        max(x - margin_x) : min(x + w + margin_x) + 1,
    ]
    for cnt in contours:
        cnt -= np.array([max(x - margin_x), max(0, y - margin_y)])

    # Reduce the number of points per contour
    contours_reduced = [None] * len(contours)
    for i, contour in enumerate(contours):

        contours_reduced[i] = cv.approxPolyDP(
            contour, contour_max_error * cv.arcLength(contour, True), True
        )

        if show_contours:
            copy_img = original_img.copy()
            for point in contours_reduced[i]:
                cv.circle(copy_img, tuple(point[0]), 1, (0, 0, 255), 5)

            cv.imshow(f"Countour {i}", copy_img)

    if show_contours:
        cv.waitKey(0)
        cv.destroyAllWindows()

    contours = contours_reduced

    log(
        f"Found {len(contours)} contours with a total of {sum(len(cnt) for cnt in contours)} points"
    )

    return contours
