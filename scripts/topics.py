import json
import os
import random

from bertopic import BERTopic
import numpy as np
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    dim_size = 5
    num_sample = 100
    num_topics = 100
    this_run_name = f'bertweet_base_{dim_size}_{num_sample if num_sample else "all"}_{num_topics}'
    run_dir_path = os.path.join(data_dir_path, this_run_name)
    if not os.path.exists(run_dir_path):
        os.mkdir(run_dir_path)

    df_path = os.path.join(data_dir_path, 'all_english_comments.csv')
    final_comments_df = pd.read_csv(df_path, dtype={'author_name': str, 'author_id': str, 'comment_id': str, 'video_id': str, 'reply_comment_id': str})

    df_path = os.path.join(data_dir_path, 'all_video_desc.csv')
    final_videos_df = pd.read_csv(df_path, dtype={'author_name': str, 'author_id': str, 'video_id': str, 'share_video_id': str, 'share_video_user_id': str})

    embeddings_cache_path = os.path.join(data_dir_path, f'all_english_comment_bertweet_embeddings.npy')
    with open(embeddings_cache_path, 'rb') as f:
        comment_umap_embeddings = np.load(f)

    embeddings_cache_path = os.path.join(data_dir_path, f'all_video_desc_bertweet_embeddings.npy')
    with open(embeddings_cache_path, 'rb') as f:
        video_umap_embeddings = np.load(f)

    if num_sample:
        final_videos_df = final_videos_df.sample(num_sample)
        sample_video_ids = set(final_videos_df['video_id'].unique())
        final_comments_df = final_comments_df[final_comments_df['video_id'].isin(sample_video_ids)]

        video_umap_embeddings = video_umap_embeddings[final_videos_df.index.values]
        comment_umap_embeddings = comment_umap_embeddings[final_comments_df.index.values]

        final_videos_df.to_csv(os.path.join(run_dir_path, 'sample_videos.csv'))
        final_comments_df.to_csv(os.path.join(run_dir_path, 'sample_comments.csv'))

        np.save(os.path.join(run_dir_path, 'sample_video_embeddings.npy'), video_umap_embeddings)
        np.save(os.path.join(run_dir_path, 'sample_comment_embeddings.npy'), comment_umap_embeddings)

    umap_embeddings = np.concatenate([comment_umap_embeddings, video_umap_embeddings], axis=0)

    comment_docs = final_comments_df['text_processed'].values
    video_docs = final_videos_df['desc_processed'].values
    docs = np.concatenate([comment_docs, video_docs])

    # Train the model on the corpus.
    pretrained_model = 'vinai/bertweet-base'
    topic_model = BERTopic(embedding_model=pretrained_model, nr_topics=num_topics, calculate_probabilities=True)

    topics, probs = topic_model.fit_transform(list(docs), embeddings=umap_embeddings)

    with open(os.path.join(run_dir_path, 'topics.json'), 'w') as f:
        json.dump([int(topic) for topic in topics], f)

    np.save(os.path.join(run_dir_path, 'probs.npy'), probs)

    topic_df = topic_model.get_topic_info()
    topic_df.to_csv(os.path.join(run_dir_path, 'topic_info.csv'))
    
    # hierarchical_topics = topic_model.hierarchical_topics(docs)
    # hierarchical_topics.to_csv(os.path.join(run_dir_path, 'hierarchical_topics.csv'))

    # tree = topic_model.get_topic_tree(hierarchical_topics)
    # with open(os.path.join(run_dir_path, f'cluster_tree.txt'), 'w') as f:
    #     f.write(tree)

    # freq_df = topic_model.get_topic_freq()
    # freq_df.to_csv(os.path.join(run_dir_path, 'topic_freqs.csv'))

    # with open(os.path.join(run_dir_path, 'topic_labels.json'), 'w') as f:
    #     json.dump(topic_model.topic_labels_, f)


if __name__ == '__main__':
    main()