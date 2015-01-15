import cv2
import numpy as np

import os


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

print "Skin" + "=" * 70

for i in get_images("skin"):
    img = cv2.imread(i)
    print "%s%%" % (img_skin_naive(img) * 100)

print "Not Skin" + "=" * 70

for i in get_images("notskin"):
    img = cv2.imread(i)
    print "%s%%" % (img_skin_naive(img) * 100)
