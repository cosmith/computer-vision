import cv2
import numpy as np

import os

import matplotlib.pyplot as plt



def get_images(directory):
    result = []
    for f in os.listdir(directory):
        if f.endswith(".jpg") or f.endswith(".png"):
            result.append(directory + "/" + f)

    return result


def display_image(image):
    img = cv2.imread(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_skin_naive(img):
    """
    Use simple classifier to detect skin
    """
    skin_ratio = 0
    height, width, depth = img.shape

    for col in img:
        for pixel in col:
            skin_ratio += 1 if is_skin_naive(pixel) else 0

    skin_ratio /= float(height * width)

    return skin_ratio


def is_skin_naive(pixel):
    """
    Simple classifier: return true if the pixel verifies a set of color constraints
    """
    b, g, r = pixel

    return (
        r > 95
        and g > 40
        and b > 20
        and max([r, g, b]) - min([r, g, b]) > 15
        and abs(int(r) - int(g)) > 15
        and r > g
        and r > b
    )

lab_img = cv2.cvtColor(cv2.imread("skin/00.png"), cv2.COLOR_BGR2LAB)
hist = cv2.calcHist([lab_img], [1,2], None, [32, 32], [0, 32, 0, 32])
plt.plot(hist)
plt.xlim([0,32])
plt.show()
