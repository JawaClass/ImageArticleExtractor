import pandas as pd

from kn_data import pre_condition_is_token_article, token2article_data


def assign_df_is_article(df: pd.DataFrame):
    is_token_article = df['text'].apply(pre_condition_is_token_article)
    df['is_token_article'] = is_token_article

    # df['is_token_article2'] = df[df['is_token_article'] == True]['text'].apply(lambda x: len(x) > 6)

    return df


def cnt_maybe_article_in_df_text(df: pd.DataFrame):
    counts = df['text'].apply(pre_condition_is_token_article).value_counts().to_dict()
    cnt_maybe = counts.get(True, 0)
    cnt_no = counts.get(False, 0)
    return cnt_maybe, cnt_no


def assign_df_is_property(df: pd.DataFrame):
    rt = []
    for group_nr, group_df in df.groupby('line_num'):
        group_df['word_gap'] = group_df['left'].shift(-1) - group_df['end']

        text_line = ' '.join(group_df['text'])
        if ':' in text_line:
            group_df['is_property_line'] = True
        else:
            group_df['is_property_line'] = False
        rt.append(group_df)
    return pd.concat(rt)

