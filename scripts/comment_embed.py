import os
import re

from bertopic import BERTopic
from bertopic.backend._utils import select_backend
import ftlangdetect
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pytok

def preprocess(raw_text):
    text = gensim.utils.to_unicode(raw_text, 'utf8', errors='ignore')
    text = text.lower()
    text = gensim.utils.deaccent(text)
    text = re.sub('@[^ ]+', '@user', text)
    text = re.sub('http[^ ]+', 'http', text)
    return text

def check_english(text):
    try:
        result = ftlangdetect.detect(text)
        return result['lang'] == 'en'
    except Exception as e:
        if str(e) == 'No features in text.':
            return False
        else:
            raise Exception('Unknown error')

def check_for_repeating_tokens(tokens):
    num_tokens = len(tokens)
    num_distinct_tokens = len(set(tokens))
    return (num_tokens / num_distinct_tokens) > 4


def load_comments_df():
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    df_path = os.path.join(data_dir_path, 'all_comments.csv')
    comment_df = pytok.utils.get_comment_df(df_path)

    comment_df = comment_df[comment_df['text'].notna()]
    comment_df = comment_df[comment_df['text'] != '']
    comment_df = comment_df[comment_df['text'] != 'Nan']
    comment_df = comment_df[comment_df['text'] != 'Null']

    regex_whitespace = '^[\s ︎]+$' # evil weird whitespace character
    comment_df = comment_df[~comment_df['text'].str.fullmatch(regex_whitespace)]

    # get only english comments
    comment_df['english'] = comment_df['text'].apply(check_english)
    english_comments_df = comment_df[comment_df['english']]

    # tokenize
    english_comments_df['text_processed'] = english_comments_df['text'].apply(preprocess)

    english_comments_df = english_comments_df[english_comments_df['text_processed'].notna()]
    english_comments_df = english_comments_df[english_comments_df['text_processed'] != '']
    english_comments_df = english_comments_df[english_comments_df['text_processed'] != 'Nan']

    return english_comments_df

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    df_path = os.path.join(data_dir_path, 'all_english_comments.csv')
    if not os.path.exists(df_path):
        final_comments_df = load_comments_df()
        final_comments_df.to_csv(df_path)

    final_comments_df = pd.read_csv(df_path)

    eng_raw_docs = list(final_comments_df['text'].values)
    docs = list(final_comments_df['text_processed'].values)
    timestamps = list(final_comments_df['createtime'].values)

    # Train the model on the corpus.
    pretrained_model = 'cardiffnlp/twitter-roberta-base'

    num_topics = 40
    topic_model = BERTopic(embedding_model=pretrained_model, nr_topics=num_topics)

    #model_path = os.path.join(data_dir_path, 'cache', 'model')

    #if not os.path.exists(model_path):
    # get embeddings so we can cache
    embeddings_cache_path = os.path.join(data_dir_path, 'all_english_comment_twitter_roberta_embeddings.npy')
    topic_model.embedding_model = select_backend(pretrained_model,
                                    language=topic_model.language)
    embeddings = topic_model._extract_embeddings(docs,
                                                method="document",
                                                verbose=topic_model.verbose)

    with open(embeddings_cache_path, 'wb') as f:
        np.save(f, embeddings)


if __name__ == '__main__':
    main()