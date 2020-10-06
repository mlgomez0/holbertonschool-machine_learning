#!/usr/bin/env python3
"""loads images from a directory or file"""
import numpy as np
import cv2
import glob
import os
import csv


def load_images(images_path, as_array=True):
    """Returns: images, filenames
       images is either a list/numpy.ndarray
       of all images
       filenames is a list of the filenames
       associated with each image in images"""
    image_paths = sorted(glob.glob(images_path + "/*"))
    images_names = [i.split("/")[-1] for i in image_paths]
    rgb_imgs = []
    for img in image_paths:
        bgr_img = cv2.imread(img)
        b, g, r = cv2.split(bgr_img)
        rgb_img = cv2.merge([r, g, b])
        rgb_imgs.append(rgb_img)
    if as_array is True:
        rgb_imgs = np.array(rgb_imgs)
    return rgb_imgs, images_names


def load_csv(csv_path, params={}):
    """loads the contents of a csv
       file as a list of lists"""
    with open(csv_path, 'r') as csvfile:
        csvlist = csv.reader(csvfile)
        csv_to_listlist = [row for row in csvlist]
    return csv_to_listlist
