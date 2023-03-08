import re
from os import listdir
from os.path import join

import pandas as pd

_program_id2program_data_buffer = {}

_program_id2program_data = {
    # desks
    'WP': ['workplace'],
    'OT': ['activet'],
    'TL': ['talos'],
    'LS': ['lifes'],
    'D4': ['doit4'],
    'B4': ['basic4', 'basic4bench'],
    'PK': ['plenumk'],
    'AJ': ['astra'],
    'BT': ['besprechungstische'],
    # container
    'CO': ['co2'],
    'EC': ['ecos'],
    # cabinets
    'S6': ['s6'],
    'S8': ['s8', 'locker'],
    # chairs
    'AZ': ['agenda2'],
    'AY': ['auray'],
    'JD': ['jet3'],
    'LA': ['lamiga'],
    'OZ': ['okay2'],
    'OD': ['okay3'],
    'PA': ['publica'],
    # other
    'PN': ['screens'],
    'NE': ['networkplace'],
    'NO': ['networkplaceorganic'],
    'IN': ['inside25'],
    'IF': ['inside50'],
    'IB': ['insidebase'],
    'IC': ['insidecube'],
}


def pre_condition_is_token_article(token):
    # 3 <= len <= 20
    pattern = r'^[A-Z0-9][A-Z0-9]([A-Z0-9-_]){1,18}$'
    pre = re.match(pattern, token)
    # pre = 2 < len(token) < 20 and token[0].isalpha() and token[1].isalnum()
    if pre:
        program_id = token[:2]
        return _program_id2program_data.get(program_id, None) is not None
    return False


def program_id2program(program_id):
    return _program_id2program_data.get(program_id, None)


def token2article_data(article_id):
    if not pre_condition_is_token_article(article_id):
        return None

    program_id = article_id[:2]
    programs = _program_id2program_data.get(program_id, None)

    if programs is None:
        return None

    result_df = None
    for p in programs:
        df = _program_id2program_data_buffer.get(p, None)
        if df is None:
            # print('pandas read csv!!!')
            path = join('data', f'kn_data_{p}.csv')
            _program_id2program_data_buffer[p] = pd.read_csv(path, low_memory=False, index_col=0)

        df = _program_id2program_data_buffer.get(p, None)
        if df is not None:
            df['article_nr_upper'] = df['article_nr'].str.upper()
            df = df[df['article_nr_upper'] == article_id.upper()]
            result_df = pd.concat([result_df, df])

    if result_df.empty:
        return None
    return result_df.sort_values(['pos_prop', 'pos_pval', 'is_default'], ascending=True)


def print_article_data(df, info_level=0):
    cols = ['property', 'need_input', 'restrictable', 'scope', 'is_default',
            'value_from', 'value_to', 'prop_text', 'pval_text'] if info_level == 1 else \
        ['need_input', 'property', 'value_from', 'prop_text', 'pval_text']
    df = df[cols]
    print(df.to_string(index=False))


def write_all_articles():
    # articles = pd.Series(name='article_nr', dtype=str)
    articles = set()
    for f in listdir('data'):
        f = join('data', f)
        _ = pd.read_csv(f, dtype=str)
        print(f)
        articles.update(_.article_nr.values)
    print(len(articles))
    s = pd.Series(name='article_nr', dtype=str, data=sorted(list(articles)))
    s.to_csv('articlenumbers.csv', index=False)
    s.to_pickle('articlenumbers.pkl')


def read_all_articles():
    # Read the Series back from the binary file
    return pd.read_pickle('articlenumbers.pkl')


#         articles = pd.concat([articles, _.article_nr])
#     print(articles.shape)  # 2.5 million
#     print(articles.unique())
# #
# def article_df2article_config(df):
#
#     cols = ['series', 'prop_class', 'property', 'pos_prop', 'need_input', 'restrictable', 'scope', 'value_from', 'pos_pval', 'article_nr',
#             'article_nr_upper']
#     df = df[cols]
#
#     df = df[df['scope'] == 'C']
#     df = df.sort_values(['series', 'article_nr', 'pos_prop', 'pos_pval'], ascending=True)
#     df = df.reset_index(drop=True)
#
#     article_id = df['article_nr'].iloc[0]
#
#     #  print(df)
#
#     #print('article_id', type(article_id), article_id)
#
#     df_props = df.groupby('property', sort=False)
#
#     for group_id, group in df_props:
#         group = group.reset_index(drop=True)
#         #print('GROUP::', group_id)
#         #print(group.to_string(index=True))
#
#         #prop = group['property'].iloc[0]
#         #prop_values = group['value_from'].values
#
#     #     print(prop, ':', prop_values)
#     #
#     # print('- ' * 200)
#     # input('.')


def extract_prop_value_from_text_line_df(df):
    ...
