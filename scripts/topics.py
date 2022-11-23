import json
import os

from bertopic import BERTopic
import numpy as np
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data')

    df_path = os.path.join(data_dir_path, 'cache', 'all_english_comments.csv')
    final_comments_df = pd.read_csv(df_path)

    df_path = os.path.join(data_dir_path, 'cache', 'all_video_desc.csv')
    final_videos_df = pd.read_csv(df_path)

    comment_docs = list(final_comments_df['text_processed'].values)
    video_docs = list(final_videos_df['desc_processed'].values)
    docs = comment_docs + video_docs

    # Train the model on the corpus.
    pretrained_model = 'vinai/bertweet-base'

    topic_model = BERTopic(embedding_model=pretrained_model)

    embeddings_cache_path = os.path.join(data_dir_path, 'cache', 'all_english_comment_bertweet_umap_embeddings.npy')
    with open(embeddings_cache_path, 'rb') as f:
        comment_umap_embeddings = np.load(f)

    embeddings_cache_path = os.path.join(data_dir_path, 'cache', 'all_video_desc_bertweet_umap_embeddings.npy')
    with open(embeddings_cache_path, 'rb') as f:
        video_umap_embeddings = np.load(f)

    umap_embeddings = np.concatenate([comment_umap_embeddings, video_umap_embeddings], axis=1)

    # Cluster reduced embeddings
    documents, probabilities = topic_model._cluster_embeddings(umap_embeddings, docs)

    # Sort and Map Topic IDs by their frequency
    if not topic_model.nr_topics:
        documents = topic_model._sort_mappings_by_frequency(documents)

    # Extract topics by calculating c-TF-IDF
    topic_model._extract_topics(documents)

    # Reduce topics
    if topic_model.nr_topics:
        documents = topic_model._reduce_topics(documents)

    topic_model._map_representative_docs(original_topics=True)
    topic_model.probabilities_ = topic_model._map_probabilities(probabilities, original_topics=True)
    predictions = documents.Topic.to_list()

    topics, probs = predictions, topic_model.probabilities_

    this_run_name = f'bertweet_base'
    run_dir_path = os.path.join(data_dir_path, 'outputs', this_run_name)
    if not os.path.exists(run_dir_path):
        os.mkdir(run_dir_path)

    with open(os.path.join(run_dir_path, 'topics.json'), 'w') as f:
        json.dump([int(topic) for topic in topics], f)

    with open(os.path.join(run_dir_path, 'probs.npy'), 'wb') as f:
        np.save(f, probs)

    topic_df = topic_model.get_topic_info()
    topic_df.to_csv(os.path.join(run_dir_path, 'topic_info.csv'))
    
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    hierarchical_topics.to_csv(os.path.join(run_dir_path, 'hierarchical_topics.csv'))

    tree = topic_model.get_topic_tree(hierarchical_topics)
    with open(os.path.join(run_dir_path, f'cluster_tree.txt'), 'w') as f:
        f.write(tree)

    freq_df = topic_model.get_topic_freq()
    freq_df.to_csv(os.path.join(run_dir_path, 'topic_freqs.csv'))

    with open(os.path.join(run_dir_path, 'topic_labels.json'), 'w') as f:
        json.dump(topic_model.topic_labels_, f)


if __name__ == '__main__':
    main()