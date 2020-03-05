from Utils import *

import cv2
import numpy as np
from tensorflow import set_random_seed
from skimage.segmentation import flood_fill


def __get_img__(file, shape):
    """
    local function to get images from file
    :param file: the file to read
    :return: the image and title of that image (title contains malignancy and nodule class)
    """
    # read the Output class containing the pixel data, malignancy value and nodule class
    x = np.load(file, allow_pickle=True)
    img = x[0].pixel_array
    img = img.reshape(shape)
    # img = cv2.resize(img, (50, 50))
    # resize the image to chosen input size
    title = "Nodule = {}, Malign = {}".format(x[0].nodule_class, x[0].malignancy)
    return img, title


def floodfill(coords, oldColor, newColor):
    """
    Depricated due to time complexity
    :param coords:
    :param oldColor:
    :param newColor:
    :return:
    """
    x, y = coords
    global global_img
    w = global_img.shape[0] - 1
    h = global_img.shape[1] - 1
    # assume surface is a 2D image and surface[x][y] is the color at x, y.

    theStack = [ (x, y) ]

    while len(theStack) > 0:
        x, y = theStack.pop()

        if global_img[x][y] != oldColor:
            continue

        global_img[x][y] = newColor
        if x < w:
            theStack.append((x + 1, y))  # right
        if x > 0:
            theStack.append((x - 1, y))  # left
        if y < h:
            theStack.append((x, y + 1))  # down
        if y > 0:
            theStack.append((x, y - 1))  # up


def crop_image(img, img2,tol=0):
    """
    Crops an image based on a mask
    :param img: the image to crop
    :param img2: the image to use as a mask
    :param tol: how low a pixel must be to be considered croppable
    :return: cropped image
    """
    # img is image data
    # tol  is tolerance
    mask = img2<tol
    return img[np.ix_(mask.any(1),mask.any(0))]


def preprocess(img, new_shape, output):
    """
    uses the preprocess_return_all to preprocess but only returns the wanted type of output
    :param img: the image to process
    :param new_shape: the shape to reshape to
    :param output: the output type, can be None, "tozero", "resized", "morphed" or "max_morphed"
    :return:
    """

    thresh, flood, crop, resized, tozero, morphed, max_morphed = preprocess_return_all(img, new_shape)

    if output == "resized":
        return resized
    if output == "tozero":
        return tozero
    if output == "morphed":
        return morphed
    elif output == "max_morphed":
        return max_morphed
    else:
        raise ValueError('Processing version not recognized as valid value, got: {}'.format(output))



def preprocess_return_all(img, new_shape):
    """
    Preprocesses an image
    :param img: the image to process
    :param new_shape: the new shape of image after processing
    :return: processed image
    """

    # sets some variables
    threshold = 255
    morph = cv2.MORPH_GRADIENT
    kernel = np.ones((2, 2), np.uint8)

    # thresholds the image to binary.
    ret, thresh = cv2.threshold(img, threshold, np.max(img), cv2.THRESH_BINARY)

    # fills the four corners of the image to remove the frame
    flood = flood_fill(thresh, (0, 0), np.max(thresh))
    flood = flood_fill(flood, (0, flood.shape[0]-1), np.max(flood))
    flood = flood_fill(flood, (flood.shape[1]-1, 0), np.max(flood))
    flood = flood_fill(flood, (flood.shape[1]-1, flood.shape[0]-1), np.max(flood))

    # crops the image using the flood filled threshold image as a mask
    crop = crop_image(img, flood, tol=np.max(flood))

    # try to resize the image, however if the image is only one color when cropping using mask,
    # the output will only be zeros. You cannot reshape an empty array thus, np.zeros are used
    try:
        resized = cv2.resize(crop, (new_shape[0], new_shape[1]))
    except:
        resized = np.zeros(new_shape)

    # thresholds the image to zero which means any value below the norm becomes zero
    # and any value above keeps its original value
    _, tozero = cv2.threshold(resized, threshold, np.max(resized), cv2.THRESH_TOZERO)

    # morphology using a gradient is used on the thresholded image to create a fuzzy outline of the lung
    morphed = cv2.morphologyEx(tozero, morph, kernel)

    # first it erodes the tozero version of the image, after which the image is thresholded with using a binary type
    # then the image is morphed using a gradient to create a hard outline of the image
    max_morphed = cv2.erode(tozero, np.ones((2, 2), np.uint8), iterations=4)
    ret, max_morphed = cv2.threshold(max_morphed, threshold, np.max(max_morphed), cv2.THRESH_BINARY)
    max_morphed = cv2.morphologyEx(max_morphed, morph, kernel)

    return thresh, flood, crop, resized, tozero, morphed, max_morphed


i = 0
if __name__ == "__main__":
    set_random_seed(1)
    np.random.seed(1)
    # shape = img.shape
    s = 352
    shape = (s, s)
    in_shape = (512, 512, 1)

    for i in ["002", "224", "225"]:
        img, title = __get_img__(os.path.join("dataset_300/0000000{}.npy".format(i)), in_shape)
        thresh, flood, crop, resized, tozero, morphed, max_morphed = preprocess_return_all(img, shape)
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        imgs = [img, thresh, flood, crop, resized, tozero, morphed, max_morphed]
        # plot_preprocessing_images([img, thresh, flood, resized, tozero, morphed, max_morphed], title)
        x = 0
        for i in imgs:
            plot_image(i, "0{}".format(x))
            x += 1