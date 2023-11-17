import json
import os

import polars as pl
import tqdm

from stance import get_prompt_template

def map_row(r):
    if r[1] is None:
        return {'comment': r[0], 'post': r[2]}
    else:
        return {'comment': r[0], 'parent_comment': r[1], 'post': r[2]}

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    run_dir_path = os.path.join(data_dir_path, 'topics_minilm_0_2')

    with open(os.path.join(run_dir_path, 'topic_stances.json'), 'r') as f:
        all_topic_stances = json.load(f)

    for topics_stances in all_topic_stances['topic_stances']:

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        print(f"Predicting stances for topics: {topics}")

        topic_comments = pl.read_parquet(os.path.join(run_dir_path, f'topic_{topics[0]}', 'sample_context_comments.parquet.zstd'))
        topic_submissions = pl.read_parquet(os.path.join(run_dir_path, f'topic_{topics[0]}', 'sample_context_submissions.parquet.zstd'))

        topic_comments = topic_comments.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('post_all_text'))

        # sort idxs by length of comment
        topic_comments = topic_comments.with_columns(pl.col('body').str.len_chars().alias('body_len'))
        topic_comments = topic_comments.sort('body_len')

        # sort idxs by length of submission
        topic_submissions = topic_submissions.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('all_text'))
        topic_submissions = topic_submissions.with_columns(pl.col('all_text').str.len_chars().alias('all_text_len'))
        topic_submissions = topic_submissions.sort('all_text_len')

        gold_comment_path = os.path.join(run_dir_path, f'topic_{topics[0]}', 'annotator_1', 'sample_context_comments_gold_stance.parquet.zstd')
        gold_submission_path = os.path.join(run_dir_path, f'topic_{topics[0]}', 'annotator_1', 'sample_context_submissions_gold_stance.parquet.zstd')

        if os.path.exists(gold_comment_path) and os.path.exists(gold_submission_path):
            topic_comments = pl.read_parquet(gold_comment_path)
            topic_submissions = pl.read_parquet(gold_submission_path)

        for stance in stances:
            print(f"Predicting stance for {stance}")
            print("Predicting stance for comments")

            cols = ['body', 'body_parent', 'post_all_text']
            if f'gold_{stance}' in topic_comments.columns:
                cols.append(f'gold_{stance}')

            comment_rows = topic_comments.select(pl.concat_list(pl.col(cols))).to_series(0).to_list()
            comment_stances = []
            for i in tqdm.tqdm(range(len(comment_rows))):
                batch_row = comment_rows[i]

                if len(batch_row) == 4:
                    try:
                        label = batch_row[-1]
                        assert label in ['-2', '-1', '0', '1', '2']
                        comment_stances.append(label)
                        continue
                    except:
                        pass

                batch_input = map_row(batch_row[:-1])
                if 'parent_comment' in batch_input:
                    prompt = get_prompt_template(True, True)
                else:
                    prompt = get_prompt_template(True, False)
                prompt = prompt.format(**batch_input, target=stance)
                gold_stance = input(prompt)
                comment_stances.append(gold_stance)
            topic_comments = topic_comments.with_columns(pl.Series(name=f"gold_{stance}", values=comment_stances))

            assert topic_comments[f'gold_{stance}'].null_count() == 0

            print("Predicting stance for submissions")

            cols = ['all_text']
            if f'gold_{stance}' in topic_submissions.columns:
                cols.append(f'gold_{stance}')

            submission_rows = topic_submissions.select(pl.concat_list(pl.col(cols))).to_series(0).to_list()
            submission_stances = []
            for i in tqdm.tqdm(range(len(submission_rows))):
                batch_row = submission_rows[i]

                if len(batch_row) == 2:
                    try:
                        label = batch_row[-1]
                        assert label in ['-2', '-1', '0', '1', '2']
                        submission_stances.append(str(label))
                        continue
                    except:
                        pass

                batch_input = {'post': batch_row[0]}
                prompt = get_prompt_template(False, False)
                prompt = prompt.format(**batch_input, target=stance)
                gold_stance = input(prompt)
                submission_stances.append(gold_stance)
            topic_submissions = topic_submissions.with_columns(pl.Series(name=f"gold_{stance}", values=submission_stances))

            assert topic_submissions[f'gold_{stance}'].null_count() == 0

        topic_comments.write_parquet(gold_comment_path, compression='zstd')
        topic_submissions.write_parquet(gold_submission_path, compression='zstd')
    
if __name__ == '__main__':
    main()