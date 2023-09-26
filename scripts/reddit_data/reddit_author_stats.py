import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    comments_dir_path = os.path.join(root_dir_path, 'data', 'reddit', 'processed_comments')
    submissions_dir_path = os.path.join(root_dir_path, 'data', 'reddit', 'processed_submissions')

    content_df = None
    for file_name in os.listdir(comments_dir_path):
        if file_name.endswith('.parquet.gzip'):
            file_df = pd.read_parquet(os.path.join(comments_dir_path, file_name), columns=['author', 'created_utc'])

            if content_df is None:
                content_df = file_df
            else:
                content_df = pd.concat([content_df, file_df])

    for file_name in os.listdir(submissions_dir_path):
        if file_name.endswith('.parquet.gzip'):
            file_df = pd.read_parquet(os.path.join(submissions_dir_path, file_name), columns=['author', 'created_utc'])

            content_df = pd.concat([content_df, file_df])

    content_df['created_utc'] = pd.to_datetime(content_df['created_utc'], unit='s')
    users_df = content_df.groupby(['author']).aggregate(['count', 'min', 'max'])
    users_df.columns = ['num_comments', 'first_comment', 'last_comment']
    users_df = users_df.reset_index()
    users_df['posting_duration'] = users_df['last_comment'] - users_df['first_comment']
    # convert to duration in days
    users_df['posting_duration'] = users_df['posting_duration'].dt.total_seconds() / (60 * 60 * 24)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # make log bins
    bins = np.logspace(np.log10(1), np.log10(users_df['num_comments'].max()), 100)
    axes[0].hist(users_df['num_comments'], bins=bins)
    axes[0].set_xlabel('Number of comments')
    axes[0].set_ylabel('Number of users')

    # make axes log
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    axes[1].hist(users_df['posting_duration'], bins=100)
    axes[1].set_xlabel('Posting duration (days)')
    axes[1].set_ylabel('Number of users')

    axes[1].set_yscale('log')

    fig_dir_path = os.path.join(root_dir_path, 'figs')
    fig.savefig(os.path.join(fig_dir_path, 'reddit_author_stats.png'))

if __name__ == '__main__':
    main()