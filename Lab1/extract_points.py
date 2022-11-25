import numpy as np
import cv2


pixel_to_mm = 0.2645833333


def create_point_list(img):
    pass


img = cv2.imread("Lab1/images/house.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow("image", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
