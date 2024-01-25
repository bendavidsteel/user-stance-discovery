import json
import math
import os

import numpy as np
import pandas as pd
import polars as pl
import tqdm

import utils
import lm
from stance import StanceClassifier, get_prompt_template

def main():
    sample_frac = None
    sample_per_topic = 100

    # Load the data.
    comment_df = utils.get_comment_df(return_pd=False)
    submission_df = utils.get_submission_df(return_pd=False)

    comment_df = comment_df.collect()
    submission_df = submission_df.collect()

    comment_df, submission_df = utils.get_parents(comment_df, submission_df)

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    run_dir_path = os.path.join(data_dir_path, 'topics_minilm_0_2')

    with open(os.path.join(run_dir_path, 'topic_stances.json'), 'r') as f:
        all_topic_stances = json.load(f)

    with open(os.path.join(run_dir_path, 'topics.json'), 'r') as f:
        doc_topics = json.load(f)

    assert len(doc_topics) == len(comment_df) + len(submission_df)

    comment_df = comment_df.with_columns(pl.Series(doc_topics[:len(comment_df)]).alias('topic'))
    submission_df = submission_df.with_columns(pl.Series(doc_topics[len(comment_df):]).alias('topic'))

    if sample_frac is not None:
        comment_df = comment_df.sample(fraction=sample_frac, seed=0)
        submission_df = submission_df.sample(fraction=sample_frac, seed=0)

    for topics_stances in all_topic_stances['topic_stances']:

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        print(f"Predicting stances for topics: {topics}")

        topic_comments = comment_df.filter(pl.col('topic').is_in(topics))
        topic_submissions = submission_df.filter(pl.col('topic').is_in(topics))

        if sample_per_topic is not None:
            # we want a deterministic sample
            topic_comments = topic_comments.sample(sample_per_topic, seed=0)
            topic_submissions = topic_submissions.sample(sample_per_topic, seed=0)

            assert len(topic_comments) >= sample_per_topic
            assert len(topic_submissions) >= sample_per_topic
        else:
            assert len(topic_comments) > 0
            assert len(topic_submissions) > 0

        if not os.path.exists(os.path.join(run_dir_path, f'topic_{topics[0]}')):
            os.mkdir(os.path.join(run_dir_path, f'topic_{topics[0]}'))

        topic_comments.write_parquet(os.path.join(run_dir_path, f'topic_{topics[0]}', 'sample_context_comments.parquet.zstd'), compression='zstd')
        topic_submissions.write_parquet(os.path.join(run_dir_path, f'topic_{topics[0]}', 'sample_context_submissions.parquet.zstd'), compression='zstd')

        topic_comments = topic_comments.to_pandas()
        topic_submissions = topic_submissions.to_pandas()

        for stance in stances:
            def map_row(r):
                r = list(r)
                if len(r) == 3:
                    if r[1] is None:
                        d = {'comment': r[0], 'post': r[2]}
                    else:
                        d = {'comment': r[0], 'parent_comment': r[1], 'post': r[2]}
                else:
                    d = {'post': r[0]}

                d = {k: t.replace('\n', ' ') if t is not None else None for k, t in d.items()}

                if 'parent_comment' in d and len(d) > 1:
                    prompt = get_prompt_template(True, True)
                elif len(d) > 1:
                    prompt = get_prompt_template(True, False)
                else:
                    prompt = get_prompt_template(False, False)
                prompt = prompt.format(**d, target=stance)

                # split prompt into readable lines
                readable_prompt = ''
                i = 0
                while i < len(prompt):
                    # seek for the next space
                    j = i + 100
                    while j < len(prompt) and prompt[j] != ' ':
                        j += 1
                    readable_prompt += prompt[i:j] + '\n'
                    i = j + 1

                return readable_prompt
            
            stance_comments_df = topic_comments.copy()
            stance_comments_df['submission_text'] = stance_comments_df[['title', 'selftext']].apply(lambda x: ' '.join([t for t in x if t is not None]) if any(x) else None, axis=1)
            stance_comments_df['prompt'] = stance_comments_df[['body', 'body_parent', 'submission_text']].apply(map_row, axis=1)
            stance_comments_df['stance'] = ''
            stance_comments_df = stance_comments_df[['prompt', 'stance']]

            stance_slug = stance.replace(' ', '_').replace('/', '_')
            stance_comments_df.to_csv(os.path.join(run_dir_path, f'topic_{topics[0]}', f'label_comments_{stance_slug}.csv'), index=False)

            stance_submissions_df = topic_submissions.copy()
            stance_submissions_df['submission_text'] = stance_submissions_df[['title', 'selftext']].apply(lambda x: ' '.join([t for t in x if t is not None]) if any(x) else None, axis=1)
            stance_submissions_df['prompt'] = stance_submissions_df[['submission_text']].apply(map_row, axis=1)
            stance_submissions_df['stance'] = ''
            stance_submissions_df = stance_submissions_df[['prompt', 'stance']]

            stance_submissions_df.to_csv(os.path.join(run_dir_path, f'topic_{topics[0]}', f'label_submissions_{stance_slug}.csv'), index=False)


    
if __name__ == '__main__':
    main()