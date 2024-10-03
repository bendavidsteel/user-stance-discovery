import os
import re

import polars as pl
import tqdm

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit')

    subreddits = ['vancouver', 'canada', 'ontario', 'toronto']
    content_to_remove = ['[removed]', '[deleted]']

    comments_dir_path = os.path.join(data_dir_path, 'processed_comments')
    for subreddit in subreddits:
        print(f'Filtering {subreddit} subreddit')
        for filename in tqdm.tqdm(os.listdir(comments_dir_path), desc='Filtering comments'):
            if re.match(r'canada_comments_\d{4}-\d{2}.parquet.gzip', filename):
                file_path = os.path.join(comments_dir_path, filename)
                new_file_path = file_path.replace('canada_comments', f'{subreddit}_comments_filtered')
                if os.path.exists(new_file_path):
                    continue
                df = pl.read_parquet(file_path)
                df = df.filter(pl.col('subreddit').is_in(subreddits))
                df = df.filter(~pl.col('body').is_in(content_to_remove))
                df.write_parquet(new_file_path, compression='gzip')

        submissions_dir_path = os.path.join(data_dir_path, 'processed_submissions')
        for filename in tqdm.tqdm(os.listdir(submissions_dir_path), desc='Filtering submissions'):
            if re.match(r'canada_submissions_\d{4}-\d{2}.parquet.gzip', filename):
                file_path = os.path.join(submissions_dir_path, filename)
                new_file_path = file_path.replace('canada_submissions', f'{subreddit}_submissions_filtered')
                if os.path.exists(new_file_path):
                    continue
                df = pl.read_parquet(file_path)
                df = df.filter(pl.col('subreddit').is_in(subreddits))
                df = df.filter(~pl.col('selftext').is_in(content_to_remove))
                df = df.filter(~pl.col('title').is_in(content_to_remove))
                df.write_parquet(new_file_path, compression='gzip')


if __name__ == '__main__':
    main()