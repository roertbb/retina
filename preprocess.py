import cv2
import numpy as np
from nn import create_model, train, predict

path = './all/'

def get_data_from_image(img, mask, manual, proportion = 0.5, num_of_patches = 1000, sample_size=32):
    positive = 0
    negative = 0
    patches = []
    patches_value = []

    img = cv2.bitwise_and(img, mask)
    h, w, _ = img.shape
    padding = int(sample_size/2)

    while positive + negative < num_of_patches:
        i = np.random.randint(int(sample_size/2), h - int(sample_size/2))
        j = np.random.randint(int(sample_size/2), w - int(sample_size/2))

        allWhite = int(np.sum(mask[i-padding:i+padding, j-padding:j+padding])) == 0

        if not allWhite:
            if manual[i, j] == 255 and positive < num_of_patches * proportion:
                patches.append(img[i-padding:i+padding, j-padding:j+padding])
                patches_value.append(1)
                positive+=1
            elif manual[i, j] == 0 and negative < num_of_patches * (1-proportion):
                patches.append(img[i-padding:i+padding, j-padding:j+padding])
                patches_value.append(0)
                negative+=1

    return patches, patches_value

def load_image_for_prepocessing(name):
    img = cv2.imread(path + "images/" + name + ".JPG")
    mask = cv2.imread(path + "mask/" + name + "_mask.tif")
    manual = cv2.imread(path + "manual1/" + name + ".tif", cv2.IMREAD_GRAYSCALE)
    return img,mask,manual

if __name__ == "__main__":
    img, mask, manual = load_image_for_prepocessing("01_dr")
    patches, patches_value = get_data_from_image(img, mask, manual)
    
    img2, mask2, manual2 = load_image_for_prepocessing("02_dr")
    patches_test, patches_test_values = get_data_from_image(img2, mask2, manual2)

    model = create_model(patches)
    model = train(model, patches, patches_value, patches_test, patches_test_values)

    predict(model, img2)
