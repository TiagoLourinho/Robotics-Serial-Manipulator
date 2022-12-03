from datetime import datetime
import numpy as np
import cv2 as cv


def log(message, end="\n"):
    """Prints a message with a timestamp"""

    print(f'[{str(datetime.now()).split(".")[0]}]: {message}', end=end)


def get_scale(desired_area: float, max_area: float) -> float:
    """Gets the scale to convert from pixels to mm"""

    return np.sqrt(desired_area / (max_area))
