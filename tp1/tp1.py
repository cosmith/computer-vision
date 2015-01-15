import cv2
import numpy as np

import os


def get_images(directory):
    result = []
    for f in os.listdir(directory):
        if f.endswith(".jpg") or f.endswith(".png"):
            result.append(directory + "/" + f)

    return result


for imgfile in get_images("ucd-db"):
    img = cv2.imread(imgfile)

cv2.waitKey(0)
cv2.destroyAllWindows()
