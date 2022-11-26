import numpy as np
import cv2 as cv
import os

IMAGE_PATH = "Lab1/images/house.png"


img = cv.imread(os.path.normpath(IMAGE_PATH))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

contours, hierarchy = cv.findContours(
    cv.bitwise_not(img),
    cv.RETR_TREE,
    cv.CHAIN_APPROX_SIMPLE,
)

blank_image = 255 * np.ones_like(img)
cv.drawContours(blank_image, contours, -1, 0, 1)

cv.imshow("Original image", img)
cv.imshow(f"Contours found ({len(contours)})", blank_image)


cv.waitKey(0)
cv.destroyAllWindows()
