import PySimpleGUI as sg
import cv2


def show_gui(cv2_images):

    encoded_images = [cv2.imencode('.jpg', _)[1].tobytes() for _ in cv2_images]

    # define your layout with image and text elements
    layout = [[sg.Image(data=_), sg.Text('Image Text')] for _ in encoded_images]

    # create the window and display the layout
    window = sg.Window('Image Viewer', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

    # close the window
    window.close()
