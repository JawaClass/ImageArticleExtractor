from io import BytesIO

import cv2
import fitz
import numpy as np
import pandas as pd
from fitz import get_text_length, TEXT_ENCODING_LATIN
from pdf2image import pdf2image
from textblob import TextBlob

from opencv_helper import draw_rectangle, show, resize_img
import tabula


def detect_language(string):
    b = TextBlob(string)
    return b.detect_language()

