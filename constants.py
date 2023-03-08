import itertools
import random
COLOR = {
    'Black': (0, 0, 0),
    #'White': (255, 255, 255),
    'Red': (255, 0, 0),
    'Lime': (0, 255, 0),
    'Blue': (0, 0, 255),
    'Yellow': (255, 255, 0),
    'Cyana': (0, 255, 255),
    'Magenta': (255, 0, 255),
    #'Silver': (192, 192, 192),
    #'Gray': (128, 128, 128),
    'Maroon': (128, 0, 0),
    'Olive': (128, 128, 0),
    'Green': (0, 128, 0),
    'Purple': (128, 0, 128),
    'Teal': (0, 128, 128),
    'Navy': (0, 0, 128),
}


def make_color_cycle():
    colors = list(COLOR.values())
    random.shuffle(colors)
    return itertools.cycle(colors)

