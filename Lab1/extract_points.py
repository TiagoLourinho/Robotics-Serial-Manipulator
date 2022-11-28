import numpy as np
import cv2 as cv
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from sklearn.preprocessing import binarize

# Hyperparameters
IMAGE_PATH = "Lab1/images/house.png"
MAX_ERROR = 0.01

original_img = cv.imread(os.path.normpath(IMAGE_PATH))

# Find the skeletonized image (1 pixel wide)
inverted_img = binarize(
    invert(cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)), threshold=127
)
skeleton_img = (skeletonize(inverted_img) * 255).astype(np.uint8)

# Find the contours
contours, hierarchy = cv.findContours(
    skeleton_img,
    cv.RETR_TREE,
    cv.CHAIN_APPROX_NONE,
)


# Crop image
contours = sorted(list(contours), key=lambda cnt: cv.contourArea(cnt), reverse=True)
x, y, w, h = cv.boundingRect(contours[0])
margin_x, margin_y = int(0.05 * w), int(0.05 * h)
original_img = original_img[
    y - margin_y : y + h + margin_y + 1,
    x - margin_x : x + w + margin_x + 1,
]
for cnt in contours:
    cnt -= np.array([x - margin_x, y - margin_y])


# Reduce the number of points per contour
contours_reduced = [None] * len(contours)
for i, contour in enumerate(contours):
    copy_img = original_img.copy()
    contours_reduced[i] = cv.approxPolyDP(
        contour, MAX_ERROR * cv.arcLength(contour, True), True
    )
    for point in contours_reduced[i]:
        cv.circle(copy_img, tuple(point[0]), 1, (0, 0, 255), 5)
    cv.imshow(f"Countour {i}", copy_img)
contours = contours_reduced


cv.waitKey(0)
cv.destroyAllWindows()

print(
    f"Found {len(contours)} contours with a total of {sum(len(cnt) for cnt in contours)} points"
)
