import os

import pandas as pd

from mining.topics import TopicModel

def main():
    sample_frac = 0.2

    # Load the data.
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    tiktok_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'tiktok')
    comment_df = pd.read_parquet(os.path.join(tiktok_dir_path, 'all_comments.parquet.gzip'))
    video_df = pd.read_parquet(os.path.join(tiktok_dir_path, 'all_videos.parquet.gzip'))

    comment_df = comment_df[comment_df['text'].notna()]
    comment_df = comment_df.merge(video_df, on='video_id', how='left')

    comment_df['all_text'] = comment_df['desc'] + ': ' + comment_df['text']
    docs = list(comment_df['all_text'].values)

    # Train the model on the corpus.
    pretrained_model = 'sentence-transformers/all-MiniLM-L12-v2'
    model_name = 'minilm'
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = tiktok_dir_path

    topic_model = TopicModel(
        pretrained_model, 
        model_name, 
        len(docs), 
        data_dir_path, 
        sample_frac=sample_frac,
        min_topic_frac=0.001,
        n_neighbours=30 # higher for less topics
    )
    topic_model.fit_transform(docs)
    if sample_frac:
        topic_model.transform(docs)


if __name__ == '__main__':
    main()