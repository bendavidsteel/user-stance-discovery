import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..', '..')

    comment_df = utils.get_comment_df()
    submission_df = utils.get_submission_df()

    comment_df['created_utc'] = pd.to_datetime(comment_df['created_utc'], unit='s')
    submission_df['created_utc'] = pd.to_datetime(submission_df['created_utc'], unit='s')

    doc_topics_path = os.path.join(root_dir_path, 'data', 'reddit', '1sub_1year', 'topics_minilm_0_2', 'topics.json')
    with open(doc_topics_path, 'r') as f:
        doc_topics = json.load(f)

    assert len(doc_topics) == len(comment_df) + len(submission_df)

    comment_df['topic'] = doc_topics[:len(comment_df)]
    submission_df['topic'] = doc_topics[len(comment_df):]

    users_df = comment_df.groupby('author').agg({'created_utc': ['count', 'min', 'max']})
    users_df.columns = ['num_comments', 'first_comment', 'last_comment']

    topic_users_df = comment_df[['author', 'topic']].groupby('author').value_counts().unstack(fill_value=0)
    # topics we care about
    topics_include = [1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24, 25]
    topic_users_df = topic_users_df[topics_include]
    topic_users_df['num_comments_on_topics'] = topic_users_df.sum(axis=1)
    topic_users_df['num_unique_topics_commented_on'] = (topic_users_df > 0).sum(axis=1)
    users_df = users_df.join(topic_users_df[['num_comments_on_topics', 'num_unique_topics_commented_on']])

    users_df = users_df.reset_index()
    users_df = users_df[users_df['author'] != '[deleted]']
    users_df['posting_duration'] = users_df['last_comment'] - users_df['first_comment']
    # convert to duration in days
    users_df['posting_duration'] = users_df['posting_duration'].dt.total_seconds() / (60 * 60 * 24)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

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

    bins = np.logspace(np.log10(1), np.log10(users_df['num_comments_on_topics'].max()), 100)
    axes[2].hist(users_df['num_comments_on_topics'], bins=bins)
    axes[2].set_xlabel('Number of comments on topics')
    axes[2].set_ylabel('Number of users')

    axes[2].set_xscale('log')
    axes[2].set_yscale('log')

    bins = np.logspace(np.log10(1), np.log10(users_df['num_unique_topics_commented_on'].max()), 100)
    axes[3].hist(users_df['num_unique_topics_commented_on'], bins=bins)
    axes[3].set_xlabel('Number of unique topics commented on')
    axes[3].set_ylabel('Number of users')

    axes[3].set_yscale('log')

    fig_dir_path = os.path.join(root_dir_path, 'figs')
    fig.savefig(os.path.join(fig_dir_path, 'reddit_author_stats.png'))

    # save authors who have at least 1000 comments on topics of interest
    # and have been active for at least 100 days
    users_df = users_df[users_df['num_comments_on_topics'] >= 100]
    users_df = users_df[users_df['posting_duration'] >= 10]

    print(f"Number of users: {len(users_df)}")

    active_users_path = os.path.join(root_dir_path, 'data', 'reddit', '1sub_1year', 'topics_minilm_0_2', 'more_active_users.parquet.gzip')
    users_df.to_parquet(active_users_path, compression='gzip', index=False)

if __name__ == '__main__':
    main()