import numpy as np
import cv2 as cv
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from sklearn.preprocessing import binarize

IMAGE_PATH = "Lab1/images/house.png"


original_img = cv.imread(os.path.normpath(IMAGE_PATH))


inverted_img = binarize(
    invert(cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)), threshold=127
)

skeleton_img = (skeletonize(inverted_img) * 255).astype(np.uint8)

contours, hierarchy = cv.findContours(
    skeleton_img,
    cv.RETR_TREE,
    cv.CHAIN_APPROX_TC89_KCOS,
)

contour_image = np.full_like(original_img, 255)

for i, cnt in enumerate(contours):
    color = tuple(np.random.choice(range(256), size=3).tolist())

    cv.drawContours(contour_image, [cnt], 0, color, 3)


cv.imshow("Original image", original_img)
cv.imshow("Skeletonized image", invert(skeleton_img))
cv.imshow(f"Contours found ({len(contours)})", contour_image)


cv.waitKey(0)
cv.destroyAllWindows()
