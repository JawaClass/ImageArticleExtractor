import collections
import itertools
import logging
import math
import os
import re
from os import listdir
from os.path import isfile, join
from random import random, shuffle

import numpy as np
import pandas as pd
import pdf2image
import pytesseract  # C:\Program Files\Tesseract-OCR
import cv2
from fuzzywuzzy import fuzz

import constants
import opencv_helper
from opencv_helper import draw_rectangle, draw_text, resize_img, draw_bounding_box_around_pandas, draw_line, show, \
    bounding_box, image_size
from kn_data import pre_condition_is_token_article, token2article_data, print_article_data

# set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# C:\Program Files\Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
os.environ[
    'TESSDATA_PREFIX'] = r'C:\Users\masteroflich\Documents\tressdata'  # r'C:\Program Files\Tesseract-OCR\tessdata'  #  r'C:\Users\masteroflich\Documents\tressdata'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

img_global = None


def format_text_token(token_df, df):
    # TODO: when we remove characters from token we also have to update left, top, width, height
    token = token_df['text']
    conf = token_df['conf']

    if type(token) is str:
        ...
    elif token is None:
        return None
    elif np.isnan(token):
        return None

    token = token.strip('|!_.')

    while '  ' in token:
        token = token.replace('  ', ' ')

    if re.match('^[_|.!]+$', token):
        return None
    if token == '':
        return None
    # TODO: proper way to filter the trash
    if conf < 85 and len(token) < 5:
        _ = next_inline_token(df, token_df)
        # return None
    return token


def ocr_parse(img_path, zoom_in_percentage):
    """
      Page segmentation modes (psm):
      0    Orientation and script detection (OSD) only.
      1    Automatic page segmentation with OSD.
      2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
      3    Fully automatic page segmentation, but no OSD. (Default)
      4    Assume a single column of text of variable sizes.
      5    Assume a single uniform block of vertically aligned text.
      6    Assume a single uniform block of text.
      7    Treat the image as a single text line.
      8    Treat the image as a single word.
      9    Treat the image as a single word in a circle.
     10    Treat the image as a single character.
     11    Sparse text. Find as much text as possible in no particular order.
     12    Sparse text with OSD.
     13    Raw line. Treat the image as a single text line,
           bypassing hacks that are Tesseract-specific.
    """
    global img_global
    img_global = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # show(img_global)

    img_global = resize_img(img_global, zoom_in_percentage)
    zoom_out_percentage = 1 / zoom_in_percentage

    custom_config = r'--oem 1 --psm 6'
    df = pytesseract.image_to_data(img_global, output_type=pytesseract.Output.DATAFRAME, lang='deu',
                                   config=custom_config)
    # print_df(df)
    img_global = resize_img(img_global, zoom_out_percentage)
    df[['left', 'top', 'width', 'height']] = df[['left', 'top', 'width', 'height']].apply(
        lambda x: x * zoom_out_percentage).astype('int64')
    # show(img_global)
    return df


def line_gaps(df: pd.DataFrame):
    t1 = df.groupby('line_num')['bottom'].apply(pd.DataFrame.max)
    t2 = df.groupby('line_num')['top'].apply(pd.DataFrame.min).shift(-1)
    # 22 ...
    # print(t2 - t1)
    # input('t2 - t1')
    return t2 - t1


def parse2lines(img_path, zoom_in_percentage=2, return_zoom=0.66):
    """
    given a path to a pdf image file
    returns a pandas df with detected lines
    """
    global img_global
    df = ocr_parse(img_path, zoom_in_percentage)
    img_width, img_height = image_size(img_global)
    df = df[df['top'] < img_height * 0.90]
    df = df.dropna()
    df = df.drop(['level', 'page_num', 'par_num', 'block_num'], axis=1)
    df['bottom'] = df.apply(lambda row: row['top'] + row['height'], axis=1)
    df['end'] = df.apply(lambda row: row['left'] + row['width'], axis=1)
    # df['text'] = df['text'].apply(format_text_token)
    df['old_text'] = df['text'].apply(lambda x: f'_{x}_')
    df['text'] = df.apply(lambda row: format_text_token(row, df), axis=1)
    df = df.dropna()
    # TODO: spellchecker for tokens with low confidence
    df = group_df_by_lines(df)
    df = df.apply(lambda x: x.sort_values('word_num', ascending=True))
    df = df.reset_index(drop=True)
    df['uid'] = df.index

    df[['left', 'top', 'width', 'height', 'bottom', 'end']] = df[
        ['left', 'top', 'width', 'height', 'bottom', 'end']].apply(
        lambda x: x * return_zoom).astype('int64')
    img_global = resize_img(img_global, return_zoom)
    img_global = cv2.cvtColor(img_global, cv2.COLOR_GRAY2BGR)

    return df, img_global


def group_df_by_lines(df, threshold=3):
    """
    necessary because default line_num from pytesseract has line_num
    depenend on paragraphs when found which cant be turned off
    """
    global img_global
    group_name = 'line_num'
    df[group_name] = (((df['bottom'] * 0.47 + df['top'] * 0.53) * 0.5).diff() > threshold)
    df[group_name] = (df[group_name]).cumsum()

    rt = df.groupby(group_name, group_keys=False)
    gaps = line_gaps(pd.concat([_[1] for _ in rt]))
    for i, (line_nr, line_df) in enumerate(rt):
        line_height = int(((line_df['top'] + line_df['bottom']) * 0.5).mean())
        draw_text(img_global, (10, line_height), f'{line_nr} . {gaps[i]}')
    return rt


def merge_df_tokens(df: pd.DataFrame):
    """
    takes n rows of token_dfs an merges into one token_df
    concatenates the token's text's and recalculates the
    boundaries
    """
    return pd.Series({
        'line_num': df['line_num'].min(),
        'word_num': df['word_num'].min(),
        'left': df['left'].min(),
        'top': df['top'].min(),
        'width': df['end'].max() - df['left'].min(),
        'height': df['height'].max(),
        'conf': df['conf'].mean(),
        'text': ' '.join(df['text']),
        'bottom': df['bottom'].max(),
        'end': df['end'].max(),
        'score': df['score'].mean(),
        'gap': df['gap'].max(),
        'column_group': df['column_group'].min()
    })


def token_df_row_intersection(token_df, df, buffer=3):
    left, top, width, height = bounding_box(token_df)
    left -= buffer
    top -= buffer
    width += buffer
    height += buffer

    mask = (df['top'] >= top) & (df['bottom'] <= (top + height))
    return df[mask]


def df_has_horizontal_intersection(df1, df2, buffer=3):
    _, top1, _, height1 = bounding_box(df1)
    _, top2, _, height2 = bounding_box(df2)

    return (top2 <= top1 <= (top2 + height2)) or (top1 <= top2 <= (top1 + height1))

    # mask1 = (df1['top'] >= top2) & (df1['bottom'] <= (top2 + height2))
    # mask2 = (df2['top'] >= top1) & (df2['bottom'] <= (top1 + height1))
    # return df1[(mask1) | (mask2)]


def is_token_df_inside_df(token_df, df):
    mask = (df['text'] == token_df['text']) & (df['left'] == token_df['left']) & (df['top'] == token_df['top'])
    return not df[mask].empty


def space_between_tokens_for_df_line(df):
    if df is None or df.empty:
        raise ValueError('space_for_df_line: df needs to be pd.Dataframe')

    height = df['height'].max()
    text = ' '.join(df['text'])

    if not any(c.isupper() for c in text):
        height = height * 1.66
    # TODO: difficult number to assess
    max_space_between_words = height * 0.60  # 0.75  # 15
    return max_space_between_words


def group_df_by_columns(df: pd.DataFrame):
    """
    find table header
    make area beam downwards
    select each line that gets hit from beam
    and group text tokens that are outside of beam to
    corresponding table column based on heuristics
    """
    global img_global
    header_terms = {'Position', 'Pos', 'Pos.', 'Pos.Nr.', 'Menge', 'Bezeichnung', 'Bez', 'Artikelnummer',
                    'Artikelnr', 'Artikelnr.', 'Preis', 'Gesamtpreis', 'G-Preis', 'Summe', 'Hinweis', 'Anzahl',
                    'Artikel', 'Rabatt'}

    def intersect(df_line: pd.DataFrame):
        intersections = set(df_line['text']).intersection(header_terms)
        df_line['score'] = len(intersections)
        return df_line

    df = df.groupby('line_num', group_keys=False).apply(intersect)

    max_score = df['score'].max()

    # no header found
    if max_score == 0:
        return

    # filter the DataFrame by the maximum score
    header_df = df[df['score'] == max_score].copy()

    header_df['gap'] = header_df['left'] - header_df['end'].shift(1)

    # calc this gap based on header text size
    max_space_between_header_words = space_between_tokens_for_df_line(
        header_df)  # header_df['height'].max() * 0.75  # 15
    header_df['column_group'] = (header_df['gap'] > max_space_between_header_words).cumsum()
    header_df = header_df.groupby('column_group', group_keys=False)

    header_df = header_df.apply(merge_df_tokens)

    beamed_down_dfs = header_df.apply(beam_down, axis=1, args=(df,))
    beamed_down_dfs = pd.concat([_ for _ in beamed_down_dfs])

    beamed_down_dfs['header_num'] = beamed_down_dfs['header_start'].rank(method='dense').astype(int)

    # print(beamed_down_dfs.to_string(index=False))
    # print(type(beamed_down_dfs))
    grouped = beamed_down_dfs.groupby(['header_num', 'line_num'])
    colors = itertools.cycle(constants.COLOR.values())
    color = next(colors)
    header_line_key = list(grouped.groups.keys())[0]
    # print('header_line_num', header_line_key)
    # input()

    # print(df.to_string(index=False))
    # input('df...')

    # found colums by simple beams down,
    # now search every column every line if there are tokens part of that line outwards of the beam
    rt = []
    remove_tokens = set()
    for i, ((header_num, line_num), line_df) in enumerate(grouped):
        if len(remove_tokens) > 0:
            match = remove_tokens.intersection(set(line_df['uid'].values))  # line_df['uid'].isin(remove_tokens)
            if match:
                # print(type(match))
                # print(match)
                # print('----')

                # print('remove', remove_tokens, 'from line', line_df['uid'].values, line_df['text'].values)

                line_df = line_df[~ line_df['uid'].isin(match)]
                # print('new linedf')
                if line_df.empty:
                    continue

        is_header = header_line_key[1] == line_num
        if is_header:
            color = next(colors)
        # last token of beamed column line
        right_token_df = line_df.iloc[-1]

        _ = next_inline_token(df, right_token_df)
        while _:
            next_token_df, gap = _
            is_connected_by_sentence = gap <= space_between_tokens_for_df_line(line_df)
            is_free_token = beamed_down_dfs[beamed_down_dfs['uid'] == next_token_df['uid']].empty
            if is_connected_by_sentence or is_free_token:
                # TODO: make link to to header token df instead (by uid)
                next_token_df['header'] = line_df.iloc[-1]['header']
                next_token_df['header_start'] = line_df.iloc[-1]['header_start']
                next_token_df['header_num'] = line_df.iloc[-1]['header_num']
                line_df = concat_pandas(line_df, next_token_df)
                remove_tokens.add(next_token_df['uid'])
                _ = next_inline_token(df, next_token_df)
            else:
                _ = None

        rt.append(line_df)

        draw_bounding_box_around_pandas(img_global, line_df, color, thickness=1)
        # print('\n' * 2)
        # input()
    return pd.concat(rt)
    # input('---')


def concat_pandas(p1: pd.DataFrame, p2: pd.DataFrame):
    if type(p1) not in [pd.DataFrame, pd.Series]:
        raise ValueError('p1 is not pandas')
    if type(p2) not in [pd.DataFrame, pd.Series]:
        raise ValueError('p2 is not pandas')
    if type(p1) is pd.Series:
        p1 = p1.to_frame().T
    if type(p2) is pd.Series:
        p2 = p2.to_frame().T
    # print(f'concat_pandas: p1: {type(p1)} , p2: {type(p2)}')
    rt = pd.concat([p1, p2], ignore_index=True)
    # print(rt)
    # input()
    return rt


def beam_down(col_header: pd.Series, df_lines: pd.DataFrame = None):
    """
    takes a table header and the rest of the the df
    returns the text tokens that horizontally intersect with a vertical beam
    of the header left and right dimension
    """
    # print('beam_down:')
    # print(type(col_header))
    # print(col_header.to_string(index=False))

    # return 'COLUMN:HII'
    header_line_left = col_header['left']
    header_line_text = col_header['text']
    # print('_'*5, 'BEAM COLUMN', header_line_text, '_'*5)
    header_line_num = col_header['line_num']
    beam_left = col_header['left'] - 10
    beam_right = col_header['end'] + 10

    df_lines_col = df_lines.copy()
    df_lines_col = df_lines_col[df_lines_col['line_num'] >= header_line_num]
    mask_start = ((df_lines_col['left'] >= beam_left) | (df_lines_col['end'] >= beam_left))
    mask_end = (df_lines_col['left'] <= beam_right)
    df_lines_col = df_lines_col[mask_start & mask_end]

    df_lines_col = df_lines_col.reset_index(drop=True).reset_index(drop=True)
    # print('beam returns 1')
    # print(type(df_lines_col))
    # print(df_lines_col)
    # input('beam returns 2')
    df_lines_col['header'] = header_line_text
    df_lines_col['header_start'] = header_line_left

    # print(df_lines_col)
    # print('return df_lines_col', type(df_lines_col))
    # input(f'beam_down of {header_line_text}')
    return df_lines_col


def next_inline_token(df: pd.DataFrame, token_df):
    return query_inline_token(df, token_df, 1)


def prev_inline_token(df: pd.DataFrame, token_df):
    return query_inline_token(df, token_df, -1)


def query_inline_token(df: pd.DataFrame, token_df, offset):
    token_num = token_df['word_num']
    line_num = token_df['line_num']
    df = df[df['line_num'] == line_num]
    next_token_df = df[df['word_num'] == token_num + offset]

    if next_token_df.empty:
        return None
    next_token_df = next_token_df.iloc[0]

    # text = token_df['text']
    # next_token = next_token_df['text']
    gap = gap_between_tokens(token_df, next_token_df)
    # print(f'next_token from: {text} -> {next_token} ---- (gap: {gap})')
    return next_token_df, gap


def gap_between_tokens(token1: pd.DataFrame, token2: pd.DataFrame):
    if type(token1) is pd.DataFrame:
        token1 = token1.iloc[0]
    if type(token2) is pd.DataFrame:
        token2 = token2.iloc[0]
    return token2['left'] - token1['end']


def area_of_df(df):
    left, top, width, height = bounding_box(df)
    return width * height


def height_of_df(df):
    left, top, width, height = bounding_box(df)
    return height


def group_df_by_line_gaps(df: pd.DataFrame, median_line_gap):
    global img_global
    gaps = line_gaps(df)
    gaps = gaps.tolist()
    groups = df.groupby('line_num')
    rt = []
    group = []
    column_row_cnt = 0

    print('median_line_gap', median_line_gap)
    for i, (group_name, group_df) in enumerate(groups):
        gap = gaps[i]
        group_df['GAP'] = gap
        group_df['column_row_cnt'] = column_row_cnt
        group.append(group_df)

        draw_line(img_global, (200, 200), (200, 200 + int(median_line_gap)))

        if gap > median_line_gap and len(group) > 0:
            rt.append(pd.concat(group.copy()))
            # draw_bounding_box_around_pandas(img_global, pd.concat(group.copy()), color=(44,22,200), thickness=4)
            column_row_cnt += 1
            group = []

    rt.append(pd.concat(group.copy()))
    rt = pd.concat(rt)


    return rt


def make_table_rows(df: pd.DataFrame):
    global img_global
    print('divided_columns')

    header_groups = df.groupby('header', group_keys=False)

    def f(x):
        x['LEN'] = len(list(x['text'].values))
        return x

    header_groups = header_groups.apply(f)

    max_header_name = 'Artikelnummer'
    max_token_len = header_groups['LEN'].max()
    max_header_name = header_groups[header_groups['LEN'] == max_token_len]['header'].iloc[0]

    # print('header_groups')
    # print(header_groups)
    # input('header_groups222')
    groups = df.groupby('column_row_cnt')

    # df.groupby(['header', 'column_row_cnt']) -> 25
    # df.groupby(column_row_cnt') -> 5
    # for nr, g in groups:
    #     print(nr)
    #     print(g)
    #     print('\n'*2)
    # print('LEN_GROUPS', len(groups))
    # input('-.----')
    rt = []
    seen = set()
    for index, (group_nr, group_df) in enumerate(groups):
        # print('group_df')
        # print(group_df)
        # input('group_df____')
        maxed = group_df.groupby('header').apply(lambda x: (x['header'].iloc[0], height_of_df(x), x))
        _, _, max_df = max(maxed, key=lambda x: x[1])
        if group_df[group_df['header'] == max_header_name].empty:
            continue
        intersection = token_df_row_intersection(max_df, df)

        intersection = intersection[~ intersection['uid'].isin(seen)]
        seen.update(set(intersection['uid'].values))
        if intersection.empty:
            continue

        # print(intersection)
        # intersection['table_row_id'] = index
        rt.append(intersection)
        draw_bounding_box_around_pandas(img_global, intersection)
        # print('INTERSECTIO', index)
        # print(intersection.to_string(index=False))
        # print('------> from')
        # print(intersection.to_string(index=False))
        # input('____')

    # print('key::')
    # print(pd.DataFrame(rt).columns)
    # print(type(pd.DataFrame(rt)))
    rt = pd.concat(rt).sort_values(['header_num', 'line_num', 'word_num'])
    # rt.groupby('table_row_id').apply(lambda x: draw_bounding_box_around_pandas(img_global, x, color=next(iter(constants.COLOR.values()))))
    # exit(0)
    return rt


def print_df(df):
    print('print_df[1]')
    print(df.to_string(index=False))
    input('print_df[2]')


def combine_col_rows_2_table_rows(df):
    # TODO: fix this mess
    global img_global
    # print('combine_col_rows_2_table_rows')
    result = []
    rows = df.groupby(['header', 'column_row_cnt'])
    seen = set()
    return_value_tables_row = []
    table_rows_index = 0
    for group_index, group_df in rows:
        group = set()
        df_droup = []
        intersections_counter = 0
        for other_group_index, other_group_df in rows:
            if group_index == other_group_index:
                continue
            if df_has_horizontal_intersection(group_df, other_group_df):
                intersections_counter += 1
                if group_index not in group:
                    df_droup.append(group_df)
                if other_group_index not in group:
                    df_droup.append(other_group_df)
                if group_index not in seen:
                    group.add(group_index)
                if other_group_index not in seen:
                    group.add(other_group_index)

                seen.add(group_index)
                seen.add(other_group_index)
            else:
                ...
                # print('no intersection ;(')
                # input()
        if intersections_counter == 0:
            # print('NO AT ALL!!!!!')
            # input()
            df_droup.append(group_df)
            group.add(group_index)
        if group and group not in result:
            result.append(group.copy())
            # print('group')
            # print(group)
            yeaah = pd.concat(df_droup)
            yeaah['table_row'] = table_rows_index
            table_rows_index += 1
            return_value_tables_row.append(yeaah)
            draw_bounding_box_around_pandas(img_global, yeaah, color=(44,22,200), thickness=4)
            #print(yeaah.to_string(index=False))
            #print('- ' * 40)
            #input()
    rt = pd.concat(return_value_tables_row)
    # print(rt.to_string(index=False))
    # print('rt')
    # input()
    return rt
