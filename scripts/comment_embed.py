import os
import re

from bertopic import BERTopic
from bertopic.backend._utils import select_backend
import ftlangdetect
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytok import utils

import tweet_normalizer

def check_english(text):
    try:
        result = ftlangdetect.detect(text)
        return result['lang'] == 'en'
    except Exception as e:
        if str(e) == 'No features in text.':
            return False
        else:
            raise Exception('Unknown error')

def load_comments_df():
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    df_path = os.path.join(data_dir_path, 'all_comments.csv')
    comment_dir_path = os.path.join(this_dir_path, '..', '..', 'polar-seeds', 'data', 'comments')

    file_paths = [os.path.join(comment_dir_path, file_name, 'video_comments.json') for file_name in os.listdir(comment_dir_path)]
    comment_df = utils.get_comment_df(df_path, file_paths=file_paths)

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
    english_comments_df['text_processed'] = english_comments_df['text'].apply(tweet_normalizer.normalizeTweet)

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

    docs = list(final_comments_df['text_processed'].values)

    # Train the model on the corpus.
    pretrained_model = 'vinai/bertweet-base'

    topic_model = BERTopic(embedding_model=pretrained_model)

    #model_path = os.path.join(data_dir_path, 'cache', 'model')

    #if not os.path.exists(model_path):
    # get embeddings so we can cache
    embeddings_cache_path = os.path.join(data_dir_path, 'all_english_comment_bertweet_embeddings.npy')
    topic_model.embedding_model = select_backend(pretrained_model,
                                    language=topic_model.language)
    topic_model.embedding_model.embedding_model.max_seq_length = 128
    embeddings = topic_model._extract_embeddings(docs,
                                                method="document",
                                                verbose=topic_model.verbose)

    with open(embeddings_cache_path, 'wb') as f:
        np.save(f, embeddings)


if __name__ == '__main__':
    main()