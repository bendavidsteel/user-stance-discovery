import os

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

def main():

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')
    comments_dir_path = os.path.join(data_dir_path, 'reddit', 'comments')

    while True:
        z_files = [os.path.join(comments_dir_path, file_name) for file_name in os.listdir(comments_dir_path) if file_name.endswith('.zst')]
        csv_files = [os.path.join(comments_dir_path, file_name) for file_name in os.listdir(comments_dir_path) if file_name.endswith('.csv')]

        not_done_z_files = []
        for z_file in z_files:
            for csv_file in csv_files:
                if z_file[-11:-4] == csv_file[-11:-4]:
                    break
            else:
                not_done_z_files.append(z_file)

        if len(not_done_z_files) == 0:
            break

        z_file = not_done_z_files[-1]

        # Adjust chunk_size as necessary -- defaults to 16,384 if not specified
        reader = Zreader(z_file)

        subreddits = ["worldnews"]

        filtered_objs = []

        # Read each line from the reader
        for line in reader.read_lines():
            obj = json.loads(line)
            if obj['subreddit'] in subreddits:
                filtered_objs.append(obj)

        # load to dataframe
        df = pd.DataFrame(filtered_objs)
        df = df[['author', 'author_fullname', 'body', 'created_utc', 'id', 'parent_id', 'permalink', 'score', 'subreddit']]

        # save to csv
        z_file_name = os.path.basename(z_file)
        df.to_csv(os.path.join(data_dir_path, z_file_name.replace('.zst', '.csv').replace('RC', 'worldnews')), index=False)

if __name__ == '__main__':
    main()