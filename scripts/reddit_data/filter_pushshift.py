import os
import re

import io
import zstandard as zstd
import ujson as json
import pandas as pd

class Zreader:

    def __init__(self, file_path):
        '''Init method'''
        self.dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        self.file_path = file_path

    def read_lines(self):
        with (
            zstd.open(self.file_path, mode='rb', dctx=self.dctx) as zfh,
            io.TextIOWrapper(zfh) as iofh
        ):
            for line in iofh:
                yield line

def process_files(raw_dir_path, processed_dir_path, col_names):

    save_file_type = 'parquet'

    if save_file_type == 'parquet':
        suffix = '.parquet.gzip'
    elif save_file_type == 'csv':
        suffix = '.csv'

    while True:
        z_files = [os.path.join(raw_dir_path, file_name) for file_name in os.listdir(raw_dir_path) if file_name.endswith('.zst')]
        p_files = [os.path.join(processed_dir_path, file_name) for file_name in os.listdir(processed_dir_path) if file_name.endswith(suffix)]

        not_done_z_files = []
        for z_file in z_files:
            z_year_month = re.search(r'\d{4}-\d{2}', z_file).group(0)
            for processed_file in p_files:
                p_year_month = re.search(r'\d{4}-\d{2}', processed_file).group(0)
                if z_year_month == p_year_month:
                    break
            else:
                not_done_z_files.append(z_file)

        if len(not_done_z_files) == 0:
            break

        z_file = not_done_z_files[0]

        print(f"Processing {os.path.basename(z_file)}")

        # Adjust chunk_size as necessary -- defaults to 16,384 if not specified
        reader = Zreader(z_file)

        # subreddits = ["worldnews"]
        subreddits = [
            "canada", 
            "ontario", 
            "PersonalFinanceCanada", 
            "onguardforthee",
            "toronto",
            "ottawa",
            "askTO",
            "alberta",
            "britishcolumbia",
            "calgary",
            "vancouver",
            "edmonton"
        ]

        filtered_objs = []

        # Read each line from the reader
        for line in reader.read_lines():
            obj = json.loads(line)
            if obj['subreddit'] in subreddits:
                filtered_objs.append(obj)

        # load to dataframe
        df = pd.DataFrame(filtered_objs)
        df = df[col_names]

        # save to csv
        z_file_name = os.path.basename(z_file)
        file_name = z_file_name.replace('RC', 'canada_comments').replace('RS', 'canada_submissions')
        
        if save_file_type == 'parquet':
            file_name = file_name.replace('.zst', '.parquet.gzip')
            df.to_parquet(os.path.join(processed_dir_path, file_name), compression='gzip')
        elif save_file_type == 'csv':
            file_name = file_name.replace('.zst', '.csv')
            df.to_csv(os.path.join(processed_dir_path, file_name), index=False)

        print(f"Saved to {file_name}")

def main():

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    raw_submissions_dir_path = os.path.join(data_dir_path, 'reddit', 'submissions')
    processed_submissions_dir_path = os.path.join(data_dir_path, 'reddit', 'processed_submissions')
    col_names = ['author', 'author_fullname', 'title', 'selftext', 'created_utc', 'id', 'num_comments', 'permalink', 'score', 'subreddit']
    process_files(raw_submissions_dir_path, processed_submissions_dir_path, col_names)

    raw_comments_dir_path = os.path.join(data_dir_path, 'reddit', 'comments')
    processed_comments_dir_path = os.path.join(data_dir_path, 'reddit', 'processed_comments')
    col_names = ['author', 'author_fullname', 'body', 'created_utc', 'id', 'parent_id', 'permalink', 'score', 'subreddit']
    process_files(raw_comments_dir_path, processed_comments_dir_path, col_names)

if __name__ == '__main__':
    main()