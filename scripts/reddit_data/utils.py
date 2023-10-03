import os

import pandas as pd

def get_data_dir_path():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit')
    return data_dir_path

def get_comment_df():
    data_dir_path = get_data_dir_path()

    comments_dir_path = os.path.join(data_dir_path, 'processed_comments')
    comment_df = None
    for file_name in os.listdir(comments_dir_path):
        file_df = pd.read_parquet(os.path.join(comments_dir_path, file_name), columns=['body'])
        if comment_df is None:
            comment_df = file_df
        else:
            comment_df = pd.concat([comment_df, file_df])

    return comment_df

def get_submission_df():
    data_dir_path = get_data_dir_path()

    submissions_dir_path = os.path.join(data_dir_path, 'processed_submissions')
    submission_df = None
    for file_name in os.listdir(submissions_dir_path):
        file_df = pd.read_parquet(os.path.join(submissions_dir_path, file_name), columns=['title', 'selftext'])
        if submission_df is None:
            submission_df = file_df
        else:
            submission_df = pd.concat([submission_df, file_df])

    return submission_df