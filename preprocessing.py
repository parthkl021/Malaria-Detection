import cv2
import numpy as np
import mahotas
from skimage import feature
from skimage import filters
from skimage.filters import unsharp_mask

def resize_image(image, size, rsize):
    np_img = []
    for i in range(size):
        resized = cv2.resize(image[i],(rsize,rsize))
        np_img.append(resized)
    return np.array(np_img)


def canny_preprocessing(images, size):
    canny_images = []
    for i in range(size):
        im = feature.canny(images[i], sigma=0.1)
        im = np.int8(im)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        canny_images.append(im)
    return canny_images

def clahe_preprocessing(images, size):
    clahe_im = []
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(16,16))  #Define tile size and clip limit. 
    for i in range(size):
        img = images[i]
        clahe_img = clahe.apply(img)
        ret2,bin_img = cv2.threshold(clahe_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        clahe_im.append(bin_img)
    return clahe_im


def sobel_preprocessing(images, size):
    sobel_img = []
    for i in range(size):
        img = images[i]
        sobel = filters.sobel(img)   
        sobel = unsharp_mask(sobel, radius=10, amount=10)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        sobel_img.append(sobel)
    return sobel_img

def scharr_preprocessing(images, size):
    scharr_img = []
    for i in range(size):
        img = images[i]
        scharr = filters.scharr(img)    
        scharr = unsharp_mask(scharr, radius=10, amount=10)
        scharr = cv2.normalize(scharr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        scharr_img.append(scharr)
    return scharr_img

def global_feature_extractor(img, length):
    global_features = []
    for i in range(length):
        # HUMoments for extracting shape
        humoment = cv2.HuMoments(cv2.moments(img[i])).flatten()
        
        # Haralick for extracting texture 
        haralick = mahotas.features.haralick(img[i]).mean(axis=0)

        #Histogram for color extraction
        hist  = cv2.calcHist(img[i], [0], None, [8], [0, 256])
        cv2.normalize(hist, hist)
        hist_flat = hist.flatten()

        features = np.hstack([humoment, hist_flat, haralick])
        global_features.append(features)
    return np.array(global_features)