import json
import os
import random

from bertopic import BERTopic
import cuml
from bertopic.backend._utils import select_backend
from bertopic.representation import KeyBERTInspired
import numpy as np
import pandas as pd

import utils

class TopicModel:
    def __init__(self, model, model_name, sample_frac=None, min_topic_size=1000):
        self.sample_frac = sample_frac
        self.model = model

        sample_text = str(sample_frac).replace('.', '_') if sample_frac else "all"
        this_run_name = f'topics_{model_name}_{sample_text}'
        this_embedding_name = f'embeddings_{model_name}.npy'

        this_dir_path = os.path.dirname(os.path.abspath(__file__))
        data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit')
        run_dir_path = os.path.join(data_dir_path, this_run_name)
        if not os.path.exists(run_dir_path):
            os.mkdir(run_dir_path)

        self.embedding_path = os.path.join(data_dir_path, this_embedding_name)

        self.run_dir_path = run_dir_path

        gpu_accelerate = False

        if gpu_accelerate:
            # Create instances of GPU-accelerated UMAP and HDBSCAN
            umap_model = cuml.manifold.UMAP(
                n_components=5, 
                n_neighbors=15, 
                min_dist=0.0
            )

            hdbscan_model = cuml.cluster.HDBSCAN(
                min_cluster_size=min_topic_size, 
                min_samples=10, 
                gen_min_span_tree=True, 
                prediction_data=True
            )
        else:
            umap_model = None
            hdbscan_model = None

        representation_model = KeyBERTInspired()

        self.topic_model = BERTopic(
            embedding_model=model, 
            nr_topics='auto', 
            min_topic_size=min_topic_size,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=representation_model,
            verbose=True, 
            low_memory=True
        )

    def get_embeddings(self, docs):
        if not os.path.exists(self.embedding_path):
            self.topic_model.embedding_model = select_backend(self.model,
                                                        language=self.topic_model.language)
            embeddings = self.topic_model._extract_embeddings(docs,
                                                    method="document",
                                                    verbose=self.topic_model.verbose)
            
            with open(self.embedding_path, 'wb') as f:
                np.save(f, embeddings)
        else:
            with open(self.embedding_path, 'rb') as f:
                embeddings = np.load(f)

        return embeddings
    
    def fit(self, docs):
        docs = list(docs)
        embeddings = self.get_embeddings(docs)

        if self.sample_frac:
            # sample both docs and embeddings
            sample_size = int(self.sample_frac * len(docs))
            sample_indices = random.sample(range(len(docs)), sample_size)
            docs = [docs[i] for i in sample_indices]
            embeddings = embeddings[sample_indices]

        topics, probs = self.topic_model.fit_transform(docs, embeddings=embeddings)

        if not self.sample_frac:
            with open(os.path.join(self.run_dir_path, 'topics.json'), 'w') as f:
                json.dump([int(topic) for topic in topics], f)

            np.save(os.path.join(self.run_dir_path, 'probs.npy'), probs)

        topic_df = self.topic_model.get_topic_info()
        topic_df.to_csv(os.path.join(self.run_dir_path, 'topic_info.csv'))
        
        hierarchical_topics = self.topic_model.hierarchical_topics(docs)
        hierarchical_topics.to_csv(os.path.join(self.run_dir_path, 'hierarchical_topics.csv'))

        tree = self.topic_model.get_topic_tree(hierarchical_topics)
        with open(os.path.join(self.run_dir_path, f'cluster_tree.txt'), 'w') as f:
            f.write(tree)

        freq_df = self.topic_model.get_topic_freq()
        freq_df.to_csv(os.path.join(self.run_dir_path, 'topic_freqs.csv'))

        with open(os.path.join(self.run_dir_path, 'topic_labels.json'), 'w') as f:
            json.dump(self.topic_model.topic_labels_, f)

    def transform(self, docs):
        topics, probs = self.topic_model.transform(list(docs), embeddings=self.get_embeddings(docs))

        with open(os.path.join(self.run_dir_path, 'topics.json'), 'w') as f:
            json.dump([int(topic) for topic in topics], f)

        np.save(os.path.join(self.run_dir_path, 'probs.npy'), probs)

def main():
    sample_frac = 0.5

    # Load the data.
    comment_df = utils.get_comment_df()
    submission_df = utils.get_submission_df()

    submission_df['all_text'] = submission_df['title'] + ' ' + submission_df['selftext']

    docs = list(comment_df['body'].values) + list(submission_df['all_text'].values)

    # Train the model on the corpus.
    pretrained_model = 'sentence-transformers/all-MiniLM-L12-v2'
    model_name = 'minilm'
    
    topic_model = TopicModel(pretrained_model, model_name, sample_frac=sample_frac)
    topic_model.fit(docs)
    if sample_frac:
        topic_model.transform(docs)


if __name__ == '__main__':
    main()