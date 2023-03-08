# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import itertools
from os.path import join
from random import sample

import cv2

# from gui import show_gui
import numpy as np
import pandas as pd

import constants
import kn_data
import pdf_parser
from OrderParser import OrderParser
from opencv_helper import show, resize_img, draw_bounding_box_around_pandas, draw_line, bounding_box, get_images, \
    draw_text, pdfs2jpgs, pdf_filenames
from pdf_parser import parse2lines, group_df_by_columns
from text_parser import assign_df_is_article, assign_df_is_property


def main():
    images = get_images()
    # images = sample(images, 3)
    # images = ['images/pdf_4_page_0.jpg']
    # important examples
    # ['images/pdf_4_page_0.jpg', 'images/pdf_9_page_1.jpg', 'images/pdf_5_page_1.jpg', 'images/pdf_8_page_1.jpg', 'images/pdf_10_page_0.jpg']
    images_result = []
    for img_path in images:
        print(img_path)

        df_lines, img = parse2lines(img_path, zoom_in_percentage=2, return_zoom=0.55)

        # test
        ###df_lines = df_lines[~ df_lines['uid'].isin([17, 24, 36, 40, 88, 92, 135, 141])]
        #

        df_colums = group_df_by_columns(df_lines)
        if df_colums is None:
            continue
            # no columns found

        gaps = pdf_parser.line_gaps(df_colums)
        # gaps = pdf_parser.line_gaps(df_lines)
        # print(gaps)
        # print('gaps___')
        median_line_gap = gaps[(gaps < 50) & (gaps > 0)].median()
        # TODO: test (cry in pain)
        median_line_gap *= 2.5  # there is an error, check with image 4_0
        if np.isnan(median_line_gap):
            median_line_gap = 12

        draw_text(img, (100, 100), f'median_line_gap: {median_line_gap}')

        divided_columns = df_colums.groupby('header', group_keys=False).apply(pdf_parser.group_df_by_line_gaps,
                                                                              median_line_gap=median_line_gap).sort_values(
            ['header_num', 'column_row_cnt', 'line_num', 'word_num'], ascending=True)
        # print('divided_columns', type(divided_columns))
        pdf_parser.combine_col_rows_2_table_rows(divided_columns)
        # input()
        # divided_columns = divided_columns[divided_columns['GAP'] > 0]

        # divided_columns.groupby(['header_num', 'column_row_cnt']).apply(
        #     lambda x: draw_bounding_box_around_pandas(img, x, color=next(iter(constants.COLOR.values()))))  # (img, df, color=(0, 0, 255), thickness=1)
        # # header_num, column_row_cnt, line_num, word_num

        table_rows = pdf_parser.make_table_rows(divided_columns)

        # print('df_colums')
        # print(df_colums.to_string(index=False))
        # print('divided_columns')
        # print(divided_columns.to_string(index=False))
        # print('divided_columns2', type(divided_columns))

        # table_rows.groupby(['column_row_cnt']).apply(df_is_text_article_maybe_count)

        df = assign_df_is_property(assign_df_is_article(table_rows))

        # pdf_parser.print_df(df)

        images_result.append((img, img_path))

    # # show_gui([_[0] for _ in images_result])
    # return
    show(images_result, write2file=True)


if __name__ == '__main__':
    pdfs = pdf_filenames()[2:3]
    #pdfs = sample(pdfs, 2)  # len(pdfs)
    #pdfs = [join('pdfs', 'BE89129-1.pdf'), join('pdfs', '20230227143641569.pdf'), join('pdfs', 'Bestellung BE2201331 vom 15.12.2022.pdf')]
    for _ in pdfs:  # sample(pdfs, len(pdfs) - 1): # ['pdfs\\Bestellung Nr. 2022100079 Capital.pdf']:#sample(pdfs, len(pdfs) - 1):  # ['pdfs\\Bestellung Nr. 2022100079 Capital.pdf']:
        parser = OrderParser(_)
        parser.parse_document()
    print('wait key...')
    cv2.waitKey(0)

    cv2.destroyAllWindows()
