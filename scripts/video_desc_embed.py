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

def load_videos_df():
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    df_path = os.path.join(data_dir_path, 'all_videos.csv')
    video_df = pytok.utils.get_video_df(df_path)

    video_df = video_df[video_df['desc'].notna()]
    video_df = video_df[video_df['desc'] != '']
    video_df = video_df[video_df['desc'] != 'Nan']
    video_df = video_df[video_df['desc'] != 'Null']

    regex_whitespace = '^[\s ︎]+$' # evil weird whitespace character
    video_df = video_df[~video_df['desc'].str.fullmatch(regex_whitespace)]

    # tokenize
    video_df['desc_processed'] = video_df['desc'].apply(preprocess)

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

    eng_raw_docs = list(final_videos_df['desc'].values)
    docs = list(final_videos_df['desc_processed'].values)
    timestamps = list(final_videos_df['createtime'].values)

    # Train the model on the corpus.
    pretrained_model = 'cardiffnlp/twitter-roberta-base'

    topic_model = BERTopic(embedding_model=pretrained_model)

    embeddings_cache_path = os.path.join(data_dir_path, 'all_video_desc_twitter_roberta_embeddings.npy')
    
    topic_model.embedding_model = select_backend(pretrained_model,
                                    language=topic_model.language)
    topic_model.embedding_model.embedding_model.max_seq_length = 512
    embeddings = topic_model._extract_embeddings(docs,
                                                method="document",
                                                verbose=topic_model.verbose)

    with open(embeddings_cache_path, 'wb') as f:
        np.save(f, embeddings)


if __name__ == '__main__':
    main()