import numpy as np
import cv2 as cv
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
import colorsys

IMAGE_PATH = "Lab1/images/house.png"


def plot_contours(contours):
    for i, contour in enumerate(contours):
        plt.subplot(1, len(contours), i + 1)
        plt.scatter(contour[:, 0, 0], contour[:, 0, 1])

        plt.xlim([0, original_img.shape[1]])
        plt.ylim([0, original_img.shape[0]])

        plt.xticks([])
        plt.yticks([])

        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")

    plt.show()


original_img = cv.imread(os.path.normpath(IMAGE_PATH))


inverted_img = binarize(
    invert(cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)), threshold=127
)

skeleton_img = (skeletonize(inverted_img) * 255).astype(np.uint8)

contours, hierarchy = cv.findContours(
    skeleton_img,
    cv.RETR_TREE,
    cv.CHAIN_APPROX_NONE,
)

contour_image = np.full_like(original_img, 255)

num = len(contours)
HSV_tuples = [(x * 1.0 / (num + 1), 1.0, 1.0) for x in range(num)]
RGB_tuples = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))) * 255
RGB_tuples = list(map(lambda x: (int(x[0]), int(x[1]), int(x[2])), RGB_tuples))
contour_shortenned = [None] * len(contours)
for i, contour in enumerate(contours):
    copy_img = original_img.copy()
    contour_shortenned[i] = cv.approxPolyDP(
        contour, 0.01 * cv.arcLength(contour, True), True
    )
    for point in contour_shortenned[i]:
        cv.circle(copy_img, tuple(point[0]), 1, RGB_tuples[i], 2)
    cv.imshow(f"Countour {i}", copy_img)

cv.waitKey(0)

for i, cnt in enumerate(contours):
    color = tuple(np.random.choice(range(256), size=3).tolist())

    cv.drawContours(contour_image, [cnt], 0, color, 3)


cv.imshow("Original image", original_img)
cv.imshow("Skeletonized image", invert(skeleton_img))
cv.imshow(f"Contours found ({len(contours)})", contour_image)


cv.waitKey(0)
cv.destroyAllWindows()

plot_contours(contours)
plot_contours(contour_shortenned)
