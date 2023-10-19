import os

import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit')

    subreddits = ['canada']
    content_to_remove = ['[removed]', '[deleted]']

    comments_dir_path = os.path.join(data_dir_path, 'processed_comments')
    for filename in os.listdir(comments_dir_path):
        if filename.startswith('canada_comments'):
            file_path = os.path.join(comments_dir_path, filename)
            df = pd.read_parquet(file_path)
            df = df[df['subreddit'].isin(subreddits)]
            df = df[~df['body'].isin(content_to_remove)]
            new_file_path = file_path.replace('canada_comments', 'canada_comments_filtered')
            df.to_parquet(new_file_path, compression='gzip')

    submissions_dir_path = os.path.join(data_dir_path, 'processed_submissions')
    for filename in os.listdir(submissions_dir_path):
        if filename.startswith('canada_submissions'):
            file_path = os.path.join(submissions_dir_path, filename)
            df = pd.read_parquet(file_path)
            df = df[df['subreddit'].isin(subreddits)]
            df = df[~df['selftext'].isin(content_to_remove)]
            df = df[~df['title'].isin(content_to_remove)]
            new_file_path = file_path.replace('canada_submissions', 'canada_submissions_filtered')
            df.to_parquet(new_file_path, compression='gzip')


if __name__ == '__main__':
    main()