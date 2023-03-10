import collections
import itertools
import logging
import os
from os import listdir
from os.path import isfile, join
from random import random, shuffle
import pandas as pd
import pdf2image
import pytesseract  # C:\Program Files\Tesseract-OCR
import cv2
from fuzzywuzzy import fuzz

from opencv_helper import draw_rectangle, draw_text, resize_img
from kn_data import pre_condition_is_token_article, token2article_data, print_article_data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# C:\Program Files\Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Users\masteroflich\Documents\tressdata'  # r'C:\Program Files\Tesseract-OCR\tessdata'  #  r'C:\Users\masteroflich\Documents\tressdata'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

img_global = None
"""
branch: test
"""


def word_similarity(word1, word2):
    return fuzz.ratio(word1, word2)


#  def group_df_by_lines(df: pd.DataFrame) -> pd.DataFrame:
def group_df_by_lines(df):
    threshold = 3
    group_name = 'line_nr'
    # df['bottom_dif'] = df['bottom'].diff()
    # df['top_dif'] = df['top'].diff()
    df['new_group'] = (((df['bottom'] + df['top']) * 0.5).diff() > threshold)

    df[group_name] = (df['new_group']).cumsum()

    grouped = df.groupby(group_name)

    sorted_groups = grouped.apply(lambda x: x.sort_values('left', ascending=True))

    return sorted_groups


# ä def group_df_by_chunks(df: pd.DataFrame) -> pd.DataFrame:
def group_df_by_chunks(df):
    df = df.copy()  # .reset_index(drop=True)
    threshold = 55
    group_name = 'chunk'
    # df['bottom_dif'] = df['bottom'].diff()
    # df['top_dif'] = df['top'].diff()
    df['new_group'] = (((df['bottom'] + df['top']) * 0.5).diff() > threshold)

    df[group_name] = (df['new_group']).cumsum()

    grouped = df.groupby(group_name, group_keys=True)

    for group_name, group_df in grouped:
        # print(f'Group name: {group_name}')
        # print(f'Group DataFrame:\n{group_df}\n')
        # ignore tiny boxes that contain no uselful information
        if area_of_df(group_df) > 1000:
            draw_bounding_box_around_pandas(group_df, color=(100, 200, 200), thickness=6)

    return grouped.apply(lambda x: x)


def area_of_df(df):
    left, top, width, height = bounding_box(df)
    return width * height


def find_table_header(df):
    search_terms = ['Position', 'Pos', 'Pos.', 'Pos.Nr.', 'Menge', 'Bezeichnung', 'Bez', 'Artikelnummer', 'Artikelnr',
                    'Artikelnr.', 'Preis', 'Gesamtpreis', 'G-Preis', 'Summe', 'Hinweis', 'Anzahl', 'Artikel', 'Rabatt']
    search_terms = [_.upper() for _ in search_terms]
    found = df[df['text_upper'].isin(search_terms)]

    if found.empty:
        # print('could not find table header')
        return None
    else:
        groups = found['line_nr'].values
        # print('groups')
        # print(groups)
        count_dict = collections.Counter(groups)

        best_group_id = max(count_dict.items(), key=lambda x: x[1])[0]

        rt = df[df['line_nr'] == best_group_id].copy()  # copy to avoid SettingWithCopyWarning

        rt['end'] = rt['left'] + rt['width']

        rt['previous_end'] = (rt['left'] + rt['width']).shift(1)

        rt['new_group'] = ((rt['left'] - rt['previous_end']) > 11).cumsum()

        def combine_rows(df) -> pd.Series:
            x = df['left'].min()
            y = df['top'].min()
            h = df['height'].max()

            most_right = df[df['left'] == df['left'].max()].iloc[0]
            w = most_right['left'] + most_right['width'] - x

            text = ' '.join(df['text'])

            return pd.Series([x, y, w, h, text])

        groups = rt.groupby('new_group')
        result = groups.apply(combine_rows)  # groups.agg(combine_rows)
        result.columns = ['left', 'top', 'width', 'height', 'text']
        result = result.reset_index()

        # print('find_table_header')
        # print(result)
        # input('--')

        return result



def find_columns(df):  # -> pd.DataFrame:
    global img_global
    headers = []
    for index, row in df.iterrows():
        text = row['text']
        x, y, w, h = row['left'], row['top'], row['width'], row['height']

        headers.append((x, y, w, h, text))
        # draw_rectangle(x, y, w, h, color=(255, 0, 0), thickness=3)

    last = headers[-1]
    # add padding to last column
    last_col_padding = 50
    headers.append((last[0] + last[2] + last_col_padding, last[1], last[2], last[3]))
    lst = list(headers)

    colors = itertools.cycle([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    data = []
    for start, end in zip(lst, lst[1:]):
        # print(f"Pair: {start}, {end}")
        offset = 0
        x, y = start[0], min([start[1], end[1]])
        w, h = end[0] - x, (start[3] + end[3]) // 2

        text = start[4]

        # print(x,y,w,h)

        draw_rectangle(img_global, x, y, w, h, color=next(colors), thickness=5)
        data.append((x, y, w, h, text))
        # yield x, y, w, h
    return pd.DataFrame(data, columns=['left', 'top', 'width', 'height', 'text'])


def search_columns(columns, df):
    padding = 15
    vertical_cut = columns['top'].min() - padding
    # print('vertical_cut', vertical_cut)
    df = df[df['top'] > vertical_cut]
    # group df by columns

    header_2_df_chunks = {}

    for index, col in columns.iterrows():

        left, top, width, height, text = col
        left -= padding
        # width += padding

        is_inside = (df['left'] >= left) & ((df['left'] + df['width']) <= (left + padding + width))
        df_col = df[is_inside].copy()

        df_col['header_text'] = text
        draw_bounding_box_around_pandas(df_col, color=(255, 255, 0))

        header_2_df_chunks[text] = divide_columns_vertical(df_col)

        # draw text tokens
        for i, row in df_col.iterrows():
            draw_bounding_box_around_pandas(row)

    align_col_chunks_horizontally(header_2_df_chunks)


def boxes_intersect(box1, box2):
    padding = 8
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    y1 -= padding
    y2 -= padding
    h1 += padding
    h2 += padding

    if y1 < y2 + h2 and y1 + h1 > y2:
        return True
    return False


def align_col_chunks_horizontally(cols):  # : dict[pd.DataFrame]
    global img_global
    # hard guessing this is the column containing the important article information
    article_description_col = max(cols.items(), key=lambda kv: kv[1].size)
    header, df = article_description_col

    other_cols = {k: v for k, v in cols.items() if k != header}

    # group by chunk_id index
    chunks = df.groupby(level=0)

    rows = []

    for chunk_id, chunk_df in chunks:
        # print(f'- GROUP #{chunk_id}')
        # print(chunk_df)
        box = bounding_box(chunk_df)

        rows.append(chunk_df.copy())

        for other_header, other_df in other_cols.items():
            other_chunks = other_df.groupby(level=0)
            for other_chunk_id, other_chunk_df in other_chunks:
                other_box = bounding_box(other_chunk_df)

                if boxes_intersect(box, other_box):
                    rows[-1] = pd.concat([rows[-1], other_chunk_df], axis=0)

    # skip header row
    # rows = rows[1:]
    for row_nr, row_df in enumerate(rows):

        row_df = row_df.sort_values(by=['top', 'left'], ascending=[True, True])

        # print(f'row #{row_nr}')
        # print(row_df)
        # input(':::')
        df_area_size = area_of_df(row_df)
        draw_bounding_box_around_pandas(row_df, (200, 0, 200), 10)
        left, top, _, _ = bounding_box(row_df)
        global img_global
        draw_text(img_global, (left + 10, top + 50), f'row #{row_nr}')

        text_tokens = row_df['text'].values

        possible_article_tokens = [_ for _ in text_tokens if pre_condition_is_token_article(_)]

        # print('text_tokens')
        # print(text_tokens)
        # print('possible_article_tokens')
        # print(possible_article_tokens)

        if len(' '.join(text_tokens)) > 100 and df_area_size > 200_000 and len(possible_article_tokens) == 0:
            logging.debug(f'No possible article_token found in text_token but row seemed plausible!\n{text_tokens}')

        articles_df = [_ for _ in [token2article_data(_) for _ in possible_article_tokens] if _ is not None]

        if len(possible_article_tokens) != len(articles_df):
            logging.debug(
                f'token found that passed possible_article_tokens() but not found in data {possible_article_tokens}')
        if len(articles_df) > 1:
            logging.debug(
                f'found {len(articles_df)} article_dfs for tokens (expected: 1): {possible_article_tokens}')

        if len(articles_df) > 1:
            logging.debug(f'len(articles_df) == {len(articles_df)}. expected: 1')
        elif len(articles_df) == 1:
            article_df = articles_df[0]
            article_nr = article_df['article_nr'].iloc[0]
            print('article', article_nr)
            # print_article_data(article_df)
            pdf_prop2val = pdf_row_df2property_value_lines(row_df)

            print('pdf_prop2val')
            print(pdf_prop2val)
            input(';;;')

            mask = (article_df['scope'] == 'C') & (article_df['need_input'] == 1)
            configurable_props2text_df = article_df[mask][['property', 'prop_text']]
            configurable_props2text_df = configurable_props2text_df.drop_duplicates(subset=['property'])
            configurable_props2text_df['seen'] = 0
            configurable_props2text_df = configurable_props2text_df.copy()
            for index, (ofml_prop, ofml_prop_text, seen) in configurable_props2text_df.iterrows():
                for pdf_prop_text, pdf_val_text, df_prop in pdf_prop2val:
                    similarity = word_similarity(pdf_prop_text, ofml_prop_text)
                    if similarity > 70:
                        mask = configurable_props2text_df['property'] == ofml_prop
                        configurable_props2text_df.loc[mask, 'seen'] = 1
                        draw_bounding_box_around_pandas(df_prop, thickness=6, color=(55, 25, 25))
                        print(f'PDF_PROP: {pdf_prop_text}, PDF_VAL: {pdf_val_text}')

            print(configurable_props2text_df.to_string(index=False))

            mean_scope_c_props_seen = configurable_props2text_df['seen'].describe()['mean']
            not_found_props = configurable_props2text_df[configurable_props2text_df['seen'] == 0]['property']
            print(f'Properties not found: {not_found_props.values}')
            print(f'Properties found: {mean_scope_c_props_seen*100}%')

            print('\n' * 2)


def pdf_row_df2property_value_lines(df):
    global img_global
    df_lines = df.groupby(df['line_nr'])
    df_lines_list = list(iter(df_lines))
    line_contains_prop = lambda x: x is not None and ':' in x
    df2joined_text = lambda x: None if x is None else ' '.join(x['text'].values)
    rt = []
    print('df_lines_list')
    print(df_lines_list)
    input('df_lines_list___')
    for i, _ in enumerate(df_lines_list):

        prev_line = None if i == 0 else df_lines_list[i - 1][1]
        line = df_lines_list[i][1]
        next_line = None if i == len(df_lines_list) - 1 else df_lines_list[i + 1][1]

        prev_line_text = df2joined_text(prev_line)
        line_text = df2joined_text(line)
        next_line_text = df2joined_text(next_line)

        single_line_prop = (line_contains_prop(line_text) and line_contains_prop(next_line_text)) or i == len(
            df_lines_list) - 1

        double_line_prop = line_contains_prop(prev_line_text) and (not line_contains_prop(line_text))

        found_new_prop = single_line_prop or double_line_prop
        if found_new_prop:
            if double_line_prop:
                line_text = ' '.join([prev_line_text, line_text])
                df_prop = pd.concat([prev_line, line])
            else:
                df_prop = line

            draw_bounding_box_around_pandas(df_prop, thickness=3, color=(0, 255, 255))

            prop, *value = line_text.split(':', maxsplit=1)
            value = tokenize_string(' '.join([_.strip() for _ in value]))
            rt.append((prop, value, df_prop))
    return rt


def tokenize_string(string):
    while (' ' * 2) in string:
        string = string.replace(' ' * 2, '')
    return string.split(' ')


def divide_columns_vertical(df):
    # header_text = df['header_text'].iloc[0]
    return group_df_by_chunks(df)





def parse(img_path):
    """
    - step 1: get pandas df of all words with pytesseract
    - step 2: group df by lines
    - step 3: search for the order table header
    - step 4: find the columns
    - step 5: search the columns for the articles
    """
    global img_global
    print('IMAGE_PATH::', img_path)
    img_global = cv2.imread(img_path)
    # resize img for better accuracy
    # img_global = resize_img(img_global, 200)
    # img_global = resize_img(img_global, 150)

    gray = cv2.cvtColor(img_global, cv2.COLOR_BGR2GRAY)
    print('pytesseract 1')
    # Run OCR on the pre-processed image
    custom_config = r'--oem 1 --psm 6'
    df = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DATAFRAME, lang='deu', config=custom_config)
    print('pytesseract 2')
    # img_global = resize_img(img_global, 50)
    df = df.dropna()
    df = df.drop(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num'], axis=1)
    # remove weird tokens with only white space. some had strangely large bounding boxes
    df['text'] = df['text'].str.strip()
    df = df[df['text'].str.len() > 0]
    df['text_upper'] = df['text'].apply(lambda x: str(x).upper())
    df['text_test'] = df['text'].apply(lambda x: f'_{str(x)}_')
    df['bottom'] = df.apply(lambda row: row['top'] + row['height'], axis=1)
    df['end'] = df.apply(lambda row: row['left'] + row['width'], axis=1)
    df = df.sort_values(['bottom'], ascending=True)

    df = group_df_by_lines(df)
    header_df = find_table_header(df)
    if header_df is not None:
        columns = find_columns(header_df)
        search_columns(columns, df)
        # print('show img', img_global)
    # img_global = resize_img(img_global, 50)
    return img_global.copy()
