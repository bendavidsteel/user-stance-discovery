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

def load_videos_df():
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    df_path = os.path.join(data_dir_path, 'all_videos.csv')

    hashtag_dir_path = os.path.join(this_dir_path, '..', '..', 'polar-seeds', 'data', 'hashtags')
    searches_dir_path = os.path.join(this_dir_path, '..', '..', 'polar-seeds', 'data', 'searches')
    file_paths = [os.path.join(hashtag_dir_path, file_name) for file_name in os.listdir(hashtag_dir_path)] \
            + [os.path.join(searches_dir_path, file_name) for file_name in os.listdir(searches_dir_path)]

    video_df = utils.get_video_df(df_path, file_paths=file_paths)

    video_df = video_df[video_df['desc'].notna()]
    video_df = video_df[video_df['desc'] != '']
    video_df = video_df[video_df['desc'] != 'Nan']
    video_df = video_df[video_df['desc'] != 'Null']

    regex_whitespace = '^[\s ︎]+$' # evil weird whitespace character
    video_df = video_df[~video_df['desc'].str.fullmatch(regex_whitespace)]

    # tokenize
    video_df['desc_processed'] = video_df['desc'].apply(tweet_normalizer.normalizeTweet)

    video_df = video_df[video_df['desc_processed'].notna()]
    video_df = video_df[video_df['desc_processed'] != '']
    video_df = video_df[video_df['desc_processed'] != 'Nan']

    return video_df

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    df_path = os.path.join(data_dir_path, 'all_video_desc.csv')
    if not os.path.exists(df_path):
        final_videos_df = load_videos_df()
        final_videos_df.to_csv(df_path)

    final_videos_df = pd.read_csv(df_path)

    docs = list(final_videos_df['desc_processed'].values)

    # Train the model on the corpus.
    pretrained_model = 'vinai/bertweet-base'

    topic_model = BERTopic(embedding_model=pretrained_model)

    embeddings_cache_path = os.path.join(data_dir_path, 'all_video_desc_bertweet_embeddings.npy')
    
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