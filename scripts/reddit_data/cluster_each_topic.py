import json
import os

import numpy as np
import pandas as pd
import tqdm

import utils
from topics import TopicModel

def main():
    sample_frac = None

    # Load the data.
    comment_df = utils.get_comment_df()
    submission_df = utils.get_submission_df()

    submission_df['all_text'] = submission_df['title'] + ' ' + submission_df['selftext']

    docs = list(comment_df['body'].values) + list(submission_df['all_text'].values)

    # Train the model on the corpus.
    pretrained_model = 'sentence-transformers/all-MiniLM-L12-v2'
    model_name = 'minilm'
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    run_dir_path = os.path.join(data_dir_path, 'topics_minilm_0_2')

    topic_info_df = pd.read_csv(os.path.join(run_dir_path, 'topic_info.csv'))

    with open(os.path.join(run_dir_path, 'topics.json'), 'r') as f:
        topics = json.load(f)

    with open(os.path.join(data_dir_path, 'embeddings_minilm.npy'), 'rb') as f:
        embeddings = np.load(f)

    topics_to_skip = [-1, 0]

    topic_info_df = topic_info_df[~topic_info_df['Topic'].isin(topics_to_skip)]
    topic_info_df = topic_info_df.sort_values('Count')

    for _, topic in topic_info_df.iterrows():
        topic_id = topic['Topic']
        print(f"Clustering topic {topic_id}")

        # index docs and embeddings from list of topic ids
        topic_idx = [i for i, t in enumerate(topics) if t == topic_id]
        topic_docs = [docs[i] for i in topic_idx]
        topic_embeddings = embeddings[topic_idx]

        topic_model = TopicModel(
            pretrained_model, 
            model_name, 
            len(topic_docs), 
            run_dir_path, 
            sample_frac=sample_frac,
            min_topic_frac=0.001,
            n_neighbours=200, # higher for less topics
            n_components=2,
            run_dir_name=f'topic_{topic_id}',
            num_topics=10
        )
        topic_model.fit_transform(topic_docs, embeddings=topic_embeddings)

        try:
            topic_model.visualize_topics(topic_embeddings)
        except Exception:
            pass

        print(f"Finished clustering topic {topic_id}")


if __name__ == '__main__':
    main()