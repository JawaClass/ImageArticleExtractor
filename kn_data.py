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
    # container
    'CO': ['co2'],
    # cabinets
    'S6': ['s6'],
    'S8': ['s8', 'locker'],
    # chairs
    'AZ': ['agenda2'],
    'AY': ['auray'],
    'JD': ['jet3'],
    'LA': ['lamiga'],
    'OZ': ['okay2'],
    'OD': ['okay3']

}


def pre_condition_is_token_article(token):
    pre = 2 < len(token) < 20 and token[0].isalpha() and token[1].isalnum()
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
            _program_id2program_data_buffer[p] = pd.read_csv(path, low_memory=False)

        df = _program_id2program_data_buffer.get(p, None)
        if df is not None:
            df['article_nr_upper'] = df['article_nr'].str.upper()
            df = df[df['article_nr_upper'] == article_id.upper()]
            result_df = pd.concat([result_df, df])

    if result_df.empty:
        return None
    return result_df


def article_df2article_config(df):

    cols = ['series', 'prop_class', 'property', 'pos_prop', 'need_input', 'restrictable', 'scope', 'value_from', 'pos_pval', 'article_nr',
            'article_nr_upper']
    df = df[cols]

    df = df[df['scope'] == 'C']
    df = df.sort_values(['series', 'article_nr', 'pos_prop', 'pos_pval'], ascending=True)
    df = df.reset_index(drop=True)

    article_id = df['article_nr'].iloc[0]

    #  print(df)

    #print('article_id', type(article_id), article_id)

    df_props = df.groupby('property', sort=False)

    for group_id, group in df_props:
        group = group.reset_index(drop=True)
        #print('GROUP::', group_id)
        #print(group.to_string(index=True))

        #prop = group['property'].iloc[0]
        #prop_values = group['value_from'].values

        #print(prop, ':', prop_values)

    #print('- ' * 200)
    # input('.')