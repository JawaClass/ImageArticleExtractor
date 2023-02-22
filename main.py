# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
from pdf_parser import parse, show


def main():
    # images = get_images()
    # shuffle(images)
    images_result = []
    for img_path in ['images/pdf_11_page_6.jpg']:
        img = parse(img_path)
        images_result.append((img, img_path))

    for img, path in images_result:
        show(img, path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
