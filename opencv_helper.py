import itertools
import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import pandas as pd
from pdf2image import pdf2image

import constants


def pdfs2jpgs():
    input = 'pdfs'
    output = 'images'
    pdfs = [f for f in listdir(input) if isfile(join(input, f))]
    for j, pdf in enumerate(pdfs):
        images = pdf2image.convert_from_path(join(input, pdf))
        for i, image in enumerate(images):
            fname = f'pdf_{j}_page_{i}.jpg'
            print('save', join(output, fname))
            image.save(join(output, fname), "JPEG")


def get_images():
    return [join('images', _) for _ in listdir('images')]


def resize_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def show(images, write2file=False):
    if type(images) is not list:
        images = [images]

    for i, img in enumerate(images):
        if type(img) is tuple:
            img, window_name = img
        else:
            window_name = f'Image #{i}'
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, img)
        if write2file:
            print('window_name', window_name)
            basename = os.path.basename(window_name)  # returns "pdf_0_page_0.jpg"
            file_name = join('output', basename)
            print('file_name', file_name)
            rt = cv2.imwrite(file_name, img)
            print(rt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_text(img, pos, text):
    # Set the font and other parameters for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org = (50, 50)
    font_scale = 0.4
    color = (0, 0, 255)
    thickness = 1

    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    box_coords = ((pos[0], pos[1] - text_height - 10), (pos[0] + text_width + 10, pos[1]))

    # Draw the white background rectangle
    # cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)

    # Draw the text on the image
    cv2.putText(img, text, pos, font, font_scale, color, thickness)


def draw_rectangle(img, x, y, w, h, color=(0, 0, 255), thickness=3):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=thickness)


def draw_line(img, p1, p2, color=(0, 0, 255), thickness=2):
    cv2.line(img, p1, p2, color, thickness)


def draw_bounding_box_around_pandas(img, df, color=(0, 0, 255), thickness=1):
    if type(df) is pd.DataFrame:
        box = bounding_box(df)

        draw_rectangle(img, *box, color=color, thickness=thickness)
    elif type(df) is pd.Series:
        box = df['left'], df['top'], df['width'], df['height']
        draw_rectangle(img, *box, color=color, thickness=thickness)


def bounding_box(df):  # -> 'left, top, width, height':
    top = df['top']
    left = df['left']
    bottom = df['bottom']
    end = df['end']
    if type(df) is pd.DataFrame:
        top = top.min()
        left = left.min()
        bottom = bottom.max()
        end = end.max()

    width = end - left
    height = bottom - top

    return left, top, width, height


def image_size(img):
    height, width = img.shape
    return width, height


def detect_objects(img):

    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert the input image to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Apply thresholding
    thresh_value =177 #127  # threshold value
    max_value = 255  # max value to use for thresholding
    ret, binary = cv2.threshold(img, thresh_value, max_value, cv2.THRESH_BINARY)

    # Apply Canny edge detection
    edges = cv2.Canny(binary, 50, 150)

    # Create a kernel for dilation
    kernel = np.ones((3, 3), np.uint8)

    # Apply dilation repeatedly
    num_dilations = 5
    img = cv2.dilate(edges, kernel, iterations=num_dilations)

    # Displa


    # Detect faces in the input image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)



    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print('detect_objects HIT')
    show(binary)