import os
import re

import pandas as pd
import polars as pl

def get_data_dir_path():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit')
    return data_dir_path

def get_comment_df(return_pd=True, subreddits=['canada']):
    data_dir_path = get_data_dir_path()

    comments_dir_path = os.path.join(data_dir_path, 'processed_comments')
    comment_df = None
    for file_name in sorted(os.listdir(comments_dir_path)):
        if any(re.match(subreddit_name + r'_comments_filtered_\d{4}-\d{2}.parquet.gzip', file_name) for subreddit_name in subreddits):
            file_df = pl.read_parquet(os.path.join(comments_dir_path, file_name))
            if comment_df is None:
                comment_df = file_df
            else:
                comment_df = pl.concat([comment_df, file_df], how='diagonal')

    if return_pd:
        comment_df = comment_df.to_pandas()

    return comment_df

def get_submission_df(return_pd=True, subreddits=['canada']):
    data_dir_path = get_data_dir_path()

    submissions_dir_path = os.path.join(data_dir_path, 'processed_submissions')
    submission_df = None
    for file_name in sorted(os.listdir(submissions_dir_path)):
        if any(re.match(subreddit_name + r'_submissions_filtered_\d{4}-\d{2}.parquet.gzip', file_name) for subreddit_name in subreddits):
            file_df = pl.read_parquet(os.path.join(submissions_dir_path, file_name))
            if submission_df is None:
                submission_df = file_df
            else:
                submission_df = pl.concat([submission_df, file_df])

    if return_pd:
        submission_df = submission_df.to_pandas()

    return submission_df

def get_parents(comment_df, submission_df):

    if 'post_permalink' not in comment_df.columns:
        comment_df = comment_df.with_columns(pl.col('permalink').str.extract(r'(\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/)', group_index=1).alias('post_permalink'))

    # get submissions for each context comment
    comment_df = comment_df.join(submission_df.select(pl.col(['permalink', 'title', 'selftext'])), left_on='post_permalink', right_on='permalink', how='left')
    # filter unmatched submissions, probably deleted    
    # comment_df = comment_df.filter(pl.col('title').is_not_null() | pl.col('selftext').is_not_null())

    # t3 is submission, t1 is comment
    # add column indicating this
    comment_df = comment_df.with_columns(pl.col('parent_id').str.contains('t3_').alias('is_base_comment'))

    # get parent comments for each comment
    comment_df = comment_df.with_columns(pl.col('parent_id').str.extract(r't[0-9]{1}_([a-z0-9]*)', group_index=1).alias('parent_specific_id'))
    comment_df = comment_df.join(comment_df.select(pl.col(['id', 'body', 'author', 'permalink'])), left_on='parent_specific_id', right_on='id', how='left', suffix='_parent')

    return comment_df, submission_df

def filter_active_user_contexts(comment_df, submission_df, active_users):
    
    comment_df = pl.from_pandas(comment_df).lazy()
    submission_df = pl.from_pandas(submission_df).lazy()
    active_users = pl.from_pandas(active_users)

    active_users = set(active_users['author'].unique())

    active_user_comment_df = comment_df.filter(pl.col('author').is_in(active_users))

    # find context comments
    active_user_comment_df = active_user_comment_df.with_columns(pl.col('permalink').str.extract(r'(\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/)', group_index=1).alias('post_permalink'))
    comment_df = comment_df.with_columns(pl.col('permalink').str.extract(r'(\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/)', group_index=1).alias('post_permalink'))
    unique_post_permalinks = set(active_user_comment_df.collect()['post_permalink'].unique().to_list())
    context_comment_df = comment_df.filter(pl.col('post_permalink').is_in(unique_post_permalinks))
    context_submission_df = submission_df.filter(pl.col('permalink').is_in(unique_post_permalinks))

    return context_comment_df, context_submission_df


def process_quotes(comments_context):
    def process_quote(t):
        if "&gt;" in t:
            t = re.sub(r'&gt;(.*)\n', r'"\1"\n', t)
        return t

    for comment_context in comments_context:
        process_quote(comment_context[0])
        if comment_context[1] is not None:
            process_quote(comment_context[1])

    return comments_context