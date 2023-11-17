import json
import os

import wandb
import polars as pl
import tqdm

from stance import StanceClassifier

def get_f1_score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn) if tp + fn > 0 else 1
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1

def get_stance_f1_score(df, gold_col, stance):

    if df[0][gold_col][0] in ['-2', '-1', '0', '1', '2']:
        # convert stance labels to favor, against, neutral
        df = df.with_columns(pl.when(pl.col(gold_col) == '-2')
                                .then(pl.lit('against'))
                                .when(pl.col(gold_col) == '-1')
                                .then(pl.lit('against'))
                                .when(pl.col(gold_col) == '0')
                                .then(pl.lit('neutral'))
                                .when(pl.col(gold_col) == '1')
                                .then(pl.lit('favor'))
                                .when(pl.col(gold_col) == '2')
                                .then(pl.lit('favor'))
                                .alias(gold_col))

    # calculate total F1 score as average of F1 scores for each stance
    # calculate f1 score for favor
    # calculate precision for favor
    num_f_tp = len(df.filter(pl.col(stance) == 'favor').filter(pl.col(gold_col) == 'favor'))
    num_f_fp = len(df.filter(pl.col(stance) == 'favor').filter(pl.col(gold_col) != 'favor'))
    num_f_fn = len(df.filter(pl.col(stance) != 'favor').filter(pl.col(gold_col) == 'favor'))

    num_a_tp = len(df.filter(pl.col(stance) == 'against').filter(pl.col(gold_col) == 'against'))
    num_a_fp = len(df.filter(pl.col(stance) == 'against').filter(pl.col(gold_col) != 'against'))
    num_a_fn = len(df.filter(pl.col(stance) != 'against').filter(pl.col(gold_col) == 'against'))

    if num_f_tp > 0:
        favor_f1 = get_f1_score(num_f_tp, num_f_fp, num_f_fn)

    if num_a_tp > 0:
        against_f1 = get_f1_score(num_a_tp, num_a_fp, num_a_fn)

    if num_f_tp > 0 and num_a_tp > 0:
        f1 = (favor_f1 + against_f1) / 2
    elif num_f_tp > 0:
        f1 = favor_f1
    elif num_a_tp > 0:
        f1 = against_f1
    else:
        f1 = 1

    return f1

def map_row(r):
    if r[1] is None:
        return {'comment': r[0], 'post': r[2]}
    else:
        return {'comment': r[0], 'parent_comment': r[1], 'post': r[2]}

def main():
    batch_size = 8

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    run_dir_path = os.path.join(data_dir_path, 'topics_minilm_0_2')

    with open(os.path.join(run_dir_path, 'topic_stances.json'), 'r') as f:
        all_topic_stances = json.load(f)

    model_name = 'zephyr'
    response_type = 'thought'
    classifier = StanceClassifier(model_name=model_name, response_type=response_type)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="reddit-stance-prompting",
        
        # track hyperparameters and run metadata
        config={
            "model_name": classifier.model_name,
            "response_type": classifier.regex,
            "reply_comment_prompt": classifier._get_prompt_template(True, True),
            "base_comment_prompt": classifier._get_prompt_template(True, False),
            "submission_prompt": classifier._get_prompt_template(False, False),
        }
    )

    comment_f1s = []
    submission_f1s = []

    for topics_stances in all_topic_stances['topic_stances']:

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        # print(f"Predicting stances for topics: {topics}")

        topic_comments = None
        comment_gold_path = os.path.join(run_dir_path, f'topic_{topics[0]}', 'sample_context_comments_gold_stance.parquet.zstd')
        if os.path.exists(comment_gold_path):
            topic_comments = pl.read_parquet(comment_gold_path)
            topic_comments = topic_comments.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('post_all_text'))

        topic_submissions = None
        submission_gold_path = os.path.join(run_dir_path, f'topic_{topics[0]}', 'sample_context_submissions_gold_stance.parquet.zstd')
        if os.path.exists(submission_gold_path):
            topic_submissions = pl.read_parquet(submission_gold_path)
            topic_submissions = topic_submissions.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('all_text'))
        
        # # sort idxs by length of comment
        # topic_comments = topic_comments.with_columns(pl.col('body').str.len_chars().alias('body_len'))
        # topic_comments = topic_comments.sort('body_len')

        # # sort idxs by length of submission
        
        # topic_submissions = topic_submissions.with_columns(pl.col('all_text').str.len_chars().alias('all_text_len'))
        # topic_submissions = topic_submissions.sort('all_text_len')

        for stance in stances:
            stance_slug = stance.replace(' ', '_')
            if topic_comments is not None and topic_submissions is not None:
                print(f"Predicting stance for {stance}")

            if topic_comments is not None:
                print("Predicting stance for comments")

                if not any(f'gold_{stance_slug}' in c for c in topic_comments.columns):
                    continue
                gold_col = [c for c in topic_comments.columns if f'gold_{stance_slug}' in c][0]
                agreement_method = gold_col.replace(f'gold_{stance_slug}_', '')
                wandb.run.config['agreement_method'] = agreement_method

                comment_rows = topic_comments.select(pl.concat_list(pl.col(['body', 'body_parent', 'post_all_text']))).to_series(0).to_list()
                comment_stances = []
                for i in tqdm.tqdm(range(0, len(comment_rows), batch_size)):
                    batch_rows = comment_rows[i:min(len(comment_rows), i+batch_size)]
                    batch_inputs = [map_row(r) for r in batch_rows]
                    batch_stances = classifier.predict_stances(batch_inputs, stance)
                    comment_stances.extend(batch_stances)
                topic_comments = topic_comments.with_columns(pl.Series(name=stance, values=comment_stances))

                comment_f1 = get_stance_f1_score(topic_comments, gold_col, stance)
                print(f"F1 score for {stance} comments: {round(comment_f1, 4)}")
                wandb.run.summary[f"{stance}_comment_f1"] = comment_f1

                comment_f1s.append(comment_f1)

            if topic_submissions is not None:
                print("Predicting stance for submissions")

                if not any(f'gold_{stance_slug}' in c for c in topic_submissions.columns):
                    continue
                gold_col = [c for c in topic_submissions.columns if f'gold_{stance_slug}' in c][0]

                submission_rows = topic_submissions.select(pl.concat_list(pl.col('all_text'))).to_series(0).to_list()
                submission_stances = []
                for i in tqdm.tqdm(range(0, len(submission_rows), batch_size)):
                    batch_rows = submission_rows[i:min(len(submission_rows), i+batch_size)]
                    batch_inputs = [{'post': r} for r in batch_rows]
                    batch_stances = classifier.predict_stances(batch_inputs, stance)
                    submission_stances.extend(batch_stances)
                topic_submissions = topic_submissions.with_columns(pl.Series(name=stance, values=submission_stances))

                submission_f1 = get_stance_f1_score(topic_submissions, gold_col, stance)
                print(f"F1 score for {stance} submissions: {round(submission_f1, 4)}")
                wandb.run.summary[f"{stance}_submission_f1"] = submission_f1

                submission_f1s.append(submission_f1)

    avg_comment_f1 = sum(comment_f1s) / len(comment_f1s)
    avg_submission_f1 = sum(submission_f1s) / len(submission_f1s)
    avg_f1 = (avg_comment_f1 + avg_submission_f1) / 2

    wandb.run.summary["avg_comment_f1"] = avg_comment_f1
    wandb.run.summary["avg_submission_f1"] = avg_submission_f1
    wandb.run.summary["avg_f1"] = avg_f1
    wandb.finish()

if __name__ == '__main__':
    main()