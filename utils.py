import glob
import cv2
import numpy as np


def import_images(folder):    
    grey = []
    img = []
    for files in glob.glob(folder):
        image = cv2.imread(files)
        imgGray = cv2.imread(files,0)
        imgGray = cv2.medianBlur(imgGray, 3)
        img.append(image)
        grey.append(imgGray)
    return img, grey


def make_label(length):
    labels = []
    for i in range(length):
        if i < int(length/2):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels, dtype='int8')