import os
import re

import polars as pl

def main():
    stance_data_path = './data/stance_targets/2022-01-01-onwards_noun_phrase_stance'
    output_path = './out/dataset_table.tex'

    file_paths = [
        os.path.join(stance_data_path, file)
        for file in os.listdir(stance_data_path)
        if re.search(r'\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', file)
    ]

    if not file_paths:
        raise ValueError("No stance data files found in the data directory")

    df = pl.read_parquet(file_paths, columns=['id', 'platform', 'seed'])
    df = df.unique(['id', 'platform'])
    df = df.with_columns(pl.col('seed').struct.field('SeedName'))

    total_num_posts = len(df)
    total_num_users = df['SeedName'].n_unique()

    platform_counts = df.group_by('platform').agg(
        pl.len().alias('num_posts'),
        pl.col('SeedName').n_unique().alias('num_users'),
    ).sort('platform')

    rows = [
        (platform.capitalize(), num_posts, num_users)
        for platform, num_posts, num_users in platform_counts.iter_rows()
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\toprule\n")
        f.write("Platform & Num Posts & Num Users \\\\\n")
        f.write("\\midrule\n")

        for platform, num_posts, num_users in rows:
            f.write(f"{platform} & {num_posts:,} & {num_users:,} \\\\\n")

        f.write("\\midrule\n")
        f.write("& Num Posts & Num People \\\\\n")
        f.write(f" & {total_num_posts:,} & {total_num_users:,} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Dataset statistics by platform.}\n")
        f.write("\\label{tab:dataset}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to {output_path}")

if __name__ == '__main__':
    main()
