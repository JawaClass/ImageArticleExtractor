import itertools
import logging
import os
import re
from os.path import join

import cv2
import fitz
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from pdf2image import pdf2image

import constants
import kn_data
from opencv_helper import draw_bounding_box_around_pandas, draw_rectangle, resize_img, draw_rectangle_p1_p2, draw_text

from pandas.core.groupby import DataFrameGroupBy

import logging

log_format = '%(levelname)s: %(message)s'
time_format = '%H:%M:%S'
logging.basicConfig(format=log_format, filename='image_article_extractor.log', level=logging.INFO)
logging = logging.getLogger()
ALL_ARTICLE_NR = set(list(kn_data.read_all_articles().values))


def print_panda(df):
    t = type(df)
    if t is pd.DataFrame:
        print(df.to_string(index=False))
    elif t is DataFrameGroupBy:
        for gnr, g in df:
            print(gnr)
            print(g.to_string(index=False))
    elif t is pd.Series:
        print(df)


def format_article_number(s: str):
    return re.sub('[,().]:;_', '', s)


# def cutoff_article_groups(df: pd.DataFrame):
#     rt = []
#     gaps = []
#     is_in_article = False
#     last_line = None
#     article_group_id = -1
#     for _, line in df.groupby('line_num'):
#         line['group_article'] = article_group_id
#         if is_in_article:
#             avg = lambda x: sum(x) / len(x)
#             gap = line.y2.mean() - last_line.y2.mean()
#
#             if len(gaps) > 0 and gap > avg(gaps):
#                 print('GAP TOO LARGE', gap, avg(gaps))
#                 print(line)
#                 print(last_line)
#                 # input()
#                 is_in_article = False
#                 gaps.clear()
#                 gaps = [gap]
#                 article_group_id += 1
#                 line['group_article'] = article_group_id
#             else:
#                 gaps.append(gap)
#                 # print('gap', gap, 'avgGap', (avg(gaps)))
#                 # print(line)
#
#         if any(line.is_article):
#             # print(line)
#             is_in_article = True
#             article_group_id += 1
#             line['group_article'] = article_group_id
#         last_line = line.copy()
#         rt.append(line.copy())
#
#     return pd.concat(rt)


def bbox_horizontal_intersection(bbox1, bbox2):
    _, y0, _, y1 = bbox1
    _, _y0, _, _y1 = bbox2
    return (_y0 <= y0 <= _y1) or (_y0 <= y1 <= _y1) or (y0 <= _y0 <= y1) or (y0 <= _y1 <= y1)


def bbox_vertical_intersection(bbox1, bbox2):
    x0, _, x1, _ = bbox1
    _x0, _, _x1, _ = bbox2
    return (_x0 <= x0 <= _x1) or (_x0 <= x1 <= _x1) or (x0 <= _x0 <= x1) or (x0 <= _x1 <= x1)


def bbox(df: 'pd.DataFrame | pd.Series'):
    if type(df) is pd.DataFrame:
        return df.x0.min(), df.y0.min(), df.x1.max(), df.y1.max()
    elif type(df) is pd.Series:
        return df.x0, df.y0, df.x1, df.y1
    else:
        raise ValueError(f'wrong argument type: {type(df)}')


def draw_bbox(img, bbox, color=(0, 0, 255), thickness=1):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=thickness)


def center(df: 'pd.DataFrame | pd.Series'):
    x0, y0, x1, y1 = bbox(df)
    x = (x0 + x1) // 2
    y = (y0 + y1) // 2
    return x, y


def is_property_line(line: str):
    return bool(re.match(r'[\w ]+:[\t ]*[\w ]+', line))


def search_article(df: pd.DataFrame):
    article = df.query('is_article == True').reset_index(drop=True)

    if article.shape[0] == 1:
        article_number = article.loc[0].word
        return article_number
    elif article.shape[0] == 0:
        return None
    else:
        logging.warning(
            f'(Not yet supported) search_article found {article.shape[0]} article numbers. Expected: 1\n\n{df}')
        # raise ValueError(f'(Not yet supported) search_article found {article.shape[0]} article numbers. Expected: 1\n\n{df}')


def prepare_fuzzy(s: str):
    pattern = r'[^a-zA-Z0-9]'
    return re.sub(pattern, '', s)


def fuzzy_compare(s1: str, s2: str):
    if s1 == s2:
        return 100
    else:
        s1 = prepare_fuzzy(s1)
        s2 = prepare_fuzzy(s2)
        return fuzz.ratio(s1, s2)


class Word:
    """
    functions for specific word
    """

    @staticmethod
    def horizontal_intersections(this: pd.DataFrame, df: pd.DataFrame):
        padding = 3
        this_y0 = this.y0.min() - padding
        this_y1 = this.y1.max() + padding
        df_y0 = df.y0
        df_y1 = df.y1
        result = df.query('@df_y0 >= @this_y0 and @df_y1 <= @this_y1')
        return result.drop(index=this.index)

    @staticmethod
    def vertical_intersections(self, this: pd.DataFrame):
        padding = 3
        this_x0 = this.x0.min() - padding
        this_x1 = this.x1.max() + padding
        df_x0 = self.df.x0
        df_x1 = self.df.x1
        result = self.df.query('@df_x0 >= @this_x0 and @df_x1 <= @this_x1')
        return result.drop(index=this.index)


class Block:
    """
    functions for specific block
    """

    @staticmethod
    def intersections(this_group_index: int, groups: DataFrameGroupBy, intersections_func):
        this: pd.DataFrame = groups.get_group(this_group_index)
        this_bbox = bbox(this)
        result = groups.apply(lambda _: intersections_func(bbox(_), this_bbox)).drop(index=this_group_index)
        mask = result[result == True]
        rt = []
        for i in mask.index:
            block = groups.get_group(i)
            rt.append(block)
        if len(rt) == 0:
            return None
        return pd.concat(rt)

    @staticmethod
    def neighbors(this_group_index: int, blocks: DataFrameGroupBy):
        h = Block.intersections(this_group_index, blocks, bbox_horizontal_intersection)
        v = Block.intersections(this_group_index, blocks, bbox_vertical_intersection)
        rt = {'west': None, 'east': None, 'north': None, 'south': None}
        this: pd.DataFrame = blocks.get_group(this_group_index)
        x0, y0, x1, y1 = bbox(this)

        def max_group(g: DataFrameGroupBy):
            i = g.apply(lambda x: x.y0).idxmax()
            if type(i) is pd.Series:
                i = i.reset_index(drop=True)
            return g.get_group(i[0])

        def min_group(g: DataFrameGroupBy):
            i = g.apply(lambda x: x.y0).idxmin()
            if type(i) is pd.Series:
                i = i.reset_index(drop=True)
            return g.get_group(i[0])

        if h is not None:
            west = h.query('x1 < @x0')
            if not west.empty:
                rt['west'] = max_group(west.groupby('block_no'))
            east = h.query('x0 > @x1')
            if not east.empty:
                rt['east'] = min_group(east.groupby('block_no'))
        if v is not None:
            north = v.query('y1 < @y0')
            if not north.empty:
                rt['north'] = max_group(north.groupby('block_no'))
            south = v.query('y0 > @y1')
            if not south.empty:
                rt['south'] = min_group(south.groupby('block_no'))

        # print(' THIS neighbors:')
        # print(this.to_string(index=False))
        # for k, v in rt.items():
        #     if v is not None:
        #         print(f'{k}:')
        #         print(v.to_string(index=False))

        return rt

    def __init__(self, df):
        self.df = df
        self.lines: DataFrameGroupBy = df.groupby('real_line_no')
        self.has_property: bool = ':' in ''.join(self.df.word)
        print('self.lines')
        for _, g in self.lines:
            print_panda(g)
        input('!!!')
    def search_properties(self):
        if not self.has_property:
            return []
        prop_val_list = []
        # print('search_properties')
        lst = list(self.lines)
        for i, _ in enumerate(lst):

            last_line = lst[i - 1] if i > 0 else None
            line = lst[i][1]
            next_line = lst[i + 1] if i < len(lst) - 1 else None

            words = ' '.join(line.word)
            prop, *value = words.split(':')
            value = ':'.join(value)
            if is_property_line(' '.join(line.word)):
                prop_val_list.append((prop, value))
                print('is_property_line', (prop, value))

            elif last_line and len(prop_val_list) > 0:  # is_property_line(' '.join(last_line[1].word)):
                # TODO: support properties over more than 2 lines
                last_prop, last_val = prop_val_list[-1]
                prop_val_list[-1] = (last_prop, ' '.join([last_val, prop]))
                print('isELIF_property_line', (prop, value))
            else:
                print('isNOOOO_property_line', (prop, value))
        # for prop, value in prop_val_list:
        #    print(f'PROP: {prop}, VALUE: {value}')
        return prop_val_list


class ArticleParser:

    def __init__(self, article_number_block: pd.DataFrame, blocks: DataFrameGroupBy, article_number: str):
        self.article_number_block = article_number_block
        self.blocks = blocks
        self.article_number = article_number

        self.current_block = article_number_block
        self.blocks_visited = []

        self.prop_value_list = []

        self.ofml_data = kn_data.token2article_data(article_number)
        self.ofml_data['selected'] = 0

        self.configuration = None

    def parse_configuration(self):

        block = Block(self.current_block)
        found_prop_values = block.search_properties()
        self.prop_value_list += found_prop_values
        self.blocks_visited.append(self.current_block)

        block_no = self.current_block.iloc[0].block_no

        west, east, north, south = Block.neighbors(block_no, self.blocks).values()

        # TODO: smarter way to find configuration blocks for this article
        if south is not None and search_article(south) is None:
            self.current_block = south
            # print('parse SOUTH')
            self.parse_configuration()
            self.configuration = self.assign_properties()

        return self.configuration

    def assign_properties(self):
        rt = []
        for prop, val in self.prop_value_list:

            print(self.article_number, prop, val)

            self.ofml_data['prop_similarity'] = self.ofml_data.apply(
                lambda x: fuzzy_compare(x.prop_text, prop), axis=1)
            self.ofml_data['val_similarity'] = self.ofml_data.apply(
                lambda x: fuzzy_compare(x.pval_text, val), axis=1)

            max_prop_similarity = self.ofml_data.prop_similarity.max()
            prop_df = self.ofml_data[self.ofml_data.prop_similarity == max_prop_similarity]
            # TODO: returns multiple if multiple rows have max_prop_similarity

            max_val_similarity = self.ofml_data.val_similarity.max()
            prop_df = prop_df[prop_df.val_similarity == max_val_similarity]
            # TODO: returns multiple if multiple rows have max_val_similarity

            self.ofml_data = self.ofml_data.drop(['prop_similarity', 'val_similarity'], axis=1)

            rt.append(prop_df)

        if len(rt) == 0:
            rt = pd.DataFrame()
        else:
            rt = pd.concat(rt)
            rt['selected'] = 1
            self.ofml_data.loc[rt.index, 'selected'] = 1

        self.configuration = rt
        # self.evaluate_accuracy()
        return self.configuration

    def evaluate_accuracy(self):
        print('evaluate_accuracy', self.article_number)
        relevant_props = self.ofml_data.query('need_input == 1 and scope == "C"')
        props_with_1 = relevant_props.query('selected == 1').property
        props_with_1 = relevant_props.query('property in @props_with_1')
        props_without_1 = relevant_props.drop(index=props_with_1.index)

        statistics = {
            'prop_similarity_min': self.configuration.prop_similarity.min(),
            'prop_similarity_mean': self.configuration.prop_similarity.mean(),

            'val_similarity_min': self.configuration.val_similarity.min(),
            'val_similarity_mean': self.configuration.val_similarity.mean(),

            'config_props_active': props_with_1.property.unique(),
            'config_props_active_size': len(props_with_1.property.unique()),
            'config_props_inactive': props_without_1.property.unique(),
            'config_props_inactive_size': len(props_without_1.property.unique()),

        }

        # for k, v in statistics.items():
        #    print(f'{k}: {v}')
        return statistics

    def has_configuration(self):
        return not (self.configuration is None or self.configuration.empty)

    def configuration_string(self):
        article = f'{self.article_number}'
        config = [f'{_[0]}: {_[1]}' for _ in self.configuration[
            ['property', 'value_from']].values.tolist()] if self.has_configuration() else ''
        return '\n'.join([article, *config])


class OrderParser:

    def __init__(self, filename):
        """
        fitz column info:
        https://github.com/pymupdf/PyMuPDF/blob/ee7faa09ccdaf47621d14b649950cba0ac9f1958/docs/textpage.rst#id14
        """
        print('FILE', filename)
        self.document: fitz.Document = fitz.open(filename)

        if not self.document.is_pdf:
            input('is not pdf!')

        doc_bytes = self.document.write()
        image_pil = pdf2image.convert_from_bytes(doc_bytes)
        self.image_cv: list = [cv2.cvtColor(np.array(_), cv2.COLOR_RGB2BGR) for _ in
                               image_pil]

        self.document_name = self.document.name
        self.filename, _ = os.path.splitext(os.path.basename(self.document_name))

        self.document_language = self.document.language
        self.document_page_count = self.document.page_count

        self.df: pd.DataFrame = ...
        self.blocks: DataFrameGroupBy = ...

    def show_page(self, page_nr):
        img = self.image_cv[page_nr]
        fname = join('images', f'{self.filename}_{page_nr}.jpg')
        fname = fname.encode('utf8').decode('ascii', errors='ignore')
        img = resize_img(img, 0.55)

        # cv2.namedWindow(fname, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(fname, 1000, 1200)
        # cv2.imshow(fname, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(fname, img)
        # os.startfile(fname)

    def draw_text(self, img):
        for _, word in self.df.iterrows():
            x = word.x
            y = word.y
            x2 = word.x2
            y2 = word.y2
            w = x2 - x
            h = y2 - y
            draw_rectangle(img, *[int(_ * 2.78) for _ in [x, y, w, h]], thickness=1)

    def parse_document(self):
        article_parser = None
        for page_nr, page in enumerate(list(self.document.pages())):
            print('page', page_nr)
            page: fitz.Page = page
            # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            words = page.get_text('words', sort=True)
            df = pd.DataFrame(words, columns=['x0', 'y0', 'x1', 'y1', 'word', 'block_no', 'line_no', 'word_no'])

            threshold = 3
            df['real_line_no'] = df.y1.diff() > threshold
            df['real_line_no'] = df.real_line_no.cumsum()
            df = df.sort_values(['real_line_no', 'x0'])
            df[['x0', 'y0', 'x1', 'y1']] = (df[['x0', 'y0', 'x1', 'y1']] * 2.78).astype(int)

            df['real_block_no'] = df.block_no

            df['is_article'] = df.word.apply(format_article_number).isin(ALL_ARTICLE_NR)
            is_articles_df = df.query('is_article == True')['word'].apply(format_article_number)
            df.loc[is_articles_df.index, 'word'] = is_articles_df

            df = df.sort_values(by=['real_line_no', 'x0'])

            self.df = df
            self.blocks = df.groupby('block_no')

            if len(self.blocks) == 0:
                print('page is image!')
                continue
            counter_configurations = 0
            colors = constants.make_color_cycle()

            for block_nr, block in self.blocks:
                color = next(colors)
                cv2.circle(self.image_cv[page_nr], center(block), 5, color, -1)

                draw_bbox(self.image_cv[page_nr], bbox(block), thickness=1, color=color)
                west, east, north, south = Block.neighbors(int(block_nr), self.blocks).values()

                if west is not None:
                    cv2.line(self.image_cv[page_nr], center(block), center(west), color, 5)
                if east is not None:
                    cv2.line(self.image_cv[page_nr], center(block), center(east), color, 5)
                if north is not None:
                    cv2.line(self.image_cv[page_nr], center(block), center(north), color, 5)
                if south is not None:
                    cv2.line(self.image_cv[page_nr], center(block), center(south), color, 5)

                article_number = search_article(block)
                if article_number is not None:
                    print(f'parse configuration for _{article_number}_')
                    article_parser = ArticleParser(block, self.blocks, article_number)
                    article_parser.parse_configuration()
                    # print_panda(configuration)
                    if 1:  # configuration is not None and not configuration.empty:
                        counter_configurations += 1
                        print(article_parser.configuration_string())
                        print(article_parser.blocks_visited)
                        # print('article_parser.blocks_visited', len(article_parser.blocks_visited))
                        for i, text in enumerate(article_parser.configuration_string().split('\n')):
                            x0, y0, x1, y1 = bbox(pd.concat(article_parser.blocks_visited[:1]))

                            cv2.putText(self.image_cv[page_nr], text, (int(page.mediabox_size[0]*2.78*0.75), y1 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if counter_configurations > 0:
                self.show_page(page_nr)
