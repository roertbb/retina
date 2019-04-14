import cv2
import matplotlib.pyplot as plt
from skimage.filters import frangi
import numpy as np
from pipeop import pipes

path = './all/'

def load_image(name):
    img = cv2.imread(path + "images/" + name + ".JPG")
    mask = cv2.imread(path + "mask/" + name + "_mask.tif")
    return cv2.bitwise_and(img, mask)

def plot_image(img, final=False):
    if final:
        plt.imshow(img)
        plt.show()
    return img

def green(img):
    return img[:, :, 1]

@pipes
# opening and closing
def morph(img, radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius,radius))
    return img >> cv2.morphologyEx(cv2.MORPH_OPEN, kernel) >> cv2.morphologyEx(cv2.MORPH_CLOSE, kernel)

@pipes
# opening and closing
def close(img, radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius,radius))
    return img >>  cv2.morphologyEx(cv2.MORPH_CLOSE, kernel)

@pipes
# histogram norm
def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    return img >> clahe.apply

@pipes
def sub_morphed(img):
    clahed = img >> clahe
    morphed = clahed >> morph(5) >> morph(10) >> morph(15) >> morph(30)
    return morphed >> cv2.subtract(clahed) >> clahe

def thresh(img):
    _, thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return thr

def remove_noise(img, min_area = 150):
    # create mask containg all contours with area smaller than min_area and remove it, remove it from ones to make bitwise_and after
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= min_area:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    return cv2.bitwise_and(img, mask)

@pipes
def main():
    # tryout: frangi
    img = load_image("01_dr")
    img = img >> green >> sub_morphed >> thresh >> remove_noise >> close(10) >> plot_image(True) 

if __name__ == "__main__":
    main()
    