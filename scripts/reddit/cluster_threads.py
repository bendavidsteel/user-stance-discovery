import json
import os
import random

import numpy as np
import polars as pl
import tqdm

from mining.topics import TopicModel

import utils

def filter_comments(comment_df):
    comment_df = comment_df.filter(pl.col('body').is_not_null() & pl.col('title').is_not_null())
    comment_df = comment_df.with_columns(pl.col('body_parent').fill_null(''))
    comment_df = comment_df.with_columns((pl.col('title') + ' ' + pl.col('body_parent') + ' ' + pl.col('body')).alias('all_text'))
    return comment_df

def main():
    random.seed(0)

    file_sample = 0.05
    num_files_sample = 0.5
    sample_frac = file_sample * num_files_sample

    comment_dir_path = './data/reddit/processed_comments'
    submission_dir_path = './data/reddit/processed_submissions'
    all_comment_df = None
    comment_files = [f for f in os.listdir(comment_dir_path) if 'filtered' in f]
    comment_files = random.sample(comment_files, int(num_files_sample * len(comment_files)))
    comment_files = sorted(comment_files)
    for comment_file in tqdm.tqdm(comment_files, desc='Loading data'):
        # find equivalent submission file
        submission_file = comment_file.replace('comments', 'submissions')
        if submission_file not in os.listdir(submission_dir_path):
            continue
        comment_df = pl.read_parquet(os.path.join(comment_dir_path, comment_file), columns=['permalink', 'id', 'parent_id', 'body', 'author'])
        submission_df = pl.read_parquet(os.path.join(submission_dir_path, submission_file), columns=['permalink', 'title', 'selftext'])
        comment_df, _ = utils.get_parents(comment_df, submission_df)
        comment_df = comment_df.sample(fraction=sample_frac, seed=0)
        comment_df = filter_comments(comment_df)
        
        if all_comment_df is None:
            all_comment_df = comment_df
        else:
            all_comment_df = pl.concat([all_comment_df, comment_df])

    docs = all_comment_df['all_text'].to_list()

    # Train the model on the corpus.
    pretrained_model = 'paraphrase-multilingual-MiniLM-L12-v2'
    model_name = 'minilm'
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '4sub_2year')

    sample_str = f"{sample_frac:.4f}".replace('.', '_')

    run_dir_name = f'topics_{model_name}_{sample_str}'
    topic_model = TopicModel(
        data_dir_path,
        embedding_model=pretrained_model,
        num_docs=len(docs),
        run_dir_name=run_dir_name,
        min_topic_frac=0.001,
        n_neighbours=30 # higher for less topics
    )
    topic_model.fit_transform(docs)
    
    all_topics = None
    topic_model.save_embeddings = False
    comment_files = [f for f in os.listdir(comment_dir_path) if 'filtered' in f]
    comment_files = sorted(comment_files)
    for comment_file in tqdm.tqdm(comment_files, desc='Transforming data'):
        # find equivalent submission file
        submission_file = comment_file.replace('comments', 'submissions')
        if submission_file not in os.listdir(submission_dir_path):
            continue
        comment_df = pl.read_parquet(os.path.join(comment_dir_path, comment_file), columns=['permalink', 'id', 'parent_id', 'body', 'author'])
        submission_df = pl.read_parquet(os.path.join(submission_dir_path, submission_file), columns=['permalink', 'title', 'selftext'])
        comment_df, submission_df = utils.get_parents(comment_df, submission_df)
        comment_df = filter_comments(comment_df)
        docs = comment_df['all_text'].to_list()

        topics, _ = topic_model.transform(docs)
        if all_topics is None:
            all_topics = topics
        else:
            all_topics = np.concatenate([all_topics, topics])
    
    np.save(os.path.join(data_dir_path, run_dir_name, 'topics.npy'), all_topics)



if __name__ == '__main__':
    main()