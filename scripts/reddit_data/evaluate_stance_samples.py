import json
import os

import dspy
import polars as pl
import tqdm
import wandb

from stance import StanceClassifier, StanceDataset, TARGET_EXPLANATIONS, TARGET_NAMES

def get_f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def get_stance_f1_score(gold_stances, stances):

    num_f_tp = 0
    num_f_fp = 0
    num_f_fn = 0
    num_a_tp = 0
    num_a_fp = 0
    num_a_fn = 0

    for gold_stance, stance in zip(gold_stances, stances):
        assert stance in ['favor', 'against', 'neutral']
        assert gold_stance in ['favor', 'against', 'neutral']
        if stance == 'favor' and gold_stance == 'favor':
            num_f_tp += 1
        if stance == 'favor' and gold_stance != 'favor':
            num_f_fp += 1
        if stance != 'favor' and gold_stance == 'favor':
            num_f_fn += 1
        if stance == 'against' and gold_stance == 'against':
            num_a_tp += 1
        if stance == 'against' and gold_stance != 'against':
            num_a_fp += 1
        if stance != 'against' and gold_stance == 'against':
            num_a_fn += 1

    # calculate total F1 score as average of F1 scores for each stance
    # calculate f1 score for favor
    # calculate precision for favor

    if (num_f_tp + num_f_fn) > 0:
        favor_precision, favor_recall, favor_f1 = get_f1_score(num_f_tp, num_f_fp, num_f_fn)

    if (num_a_tp + num_a_fn) > 0:
        against_precision, against_recall, against_f1 = get_f1_score(num_a_tp, num_a_fp, num_a_fn)

    if (num_f_tp + num_f_fn) > 0 and (num_a_tp + num_a_fn) > 0:
        f1 = (favor_f1 + against_f1) / 2
        precision = (favor_precision + against_precision) / 2
        recall = (favor_recall + against_recall) / 2
    elif (num_f_tp + num_a_fn) > 0:
        f1 = favor_f1
        precision = favor_precision
        recall = favor_recall
    elif (num_a_tp + num_a_fn) > 0:
        f1 = against_f1
        precision = against_precision
        recall = against_recall
    else:
        raise Exception("No true positives or true negatives")

    return precision, recall, f1

def do_stance_detection(stance, stance_slug, df, cols, classifier, batch_size, text_type, f1s, precisions, recalls, log_to_wandb, topic_path, train_num, val_num, dataset_strategy):
    if not any(f'gold_{stance_slug}' in c for c in df.columns):
        return
    gold_col = [c for c in df.columns if f'gold_{stance_slug}' in c][0]
    agreement_method = gold_col.replace(f'gold_{stance_slug}_', '')
    if log_to_wandb:
        wandb.run.config['agreement_method'] = agreement_method
        wandb.run.config[f'{stance_slug}_target_explanation'] = TARGET_EXPLANATIONS[stance]
        wandb.run.config[f'{stance_slug}_target_name'] = TARGET_NAMES[stance]

    inputs = df.select(pl.concat_list(pl.col(cols + [gold_col]))).to_series(0).to_list()

    dataset = StanceDataset(inputs, stance, train_num=train_num, val_num=val_num, strategy=dataset_strategy)

    if train_num + val_num > 0:
        train_dataset = dataset.get_train_data()
        val_dataset = dataset.get_dev_data()
        classifier.train(train_dataset, val_dataset)

        if log_to_wandb:
            wandb.run.config['teleprompter_settings'] = classifier.teleprompter_settings

    data = dataset.get_test_data()
    stances = []
    gold_stances = []
    incorrect_cases = []
    for i in tqdm.tqdm(range(0, len(data), batch_size)):
        batch_inputs = data[i:min(len(data), i+batch_size)]
        batch_stances = classifier.predict_stances(batch_inputs, stance)
        stances.extend(batch_stances)
        for input, stance_output in zip(batch_inputs, batch_stances):
            gold_stances.append(input.gold_stance)
            if stance_output != input.gold_stance:
                incorrect_cases.append({
                    'comment': input.comment,
                    'comment_parent': input.parent_comment,
                    'post': input.post,
                    'target': input.target_opinion,
                    'gold_stance': input.gold_stance,
                    'model_output': classifier._extra_responses,
                })

    precision, recall, f1 = get_stance_f1_score(gold_stances, stances)

    with open(os.path.join(topic_path, f'{stance_slug}_{text_type}_incorrect_cases.json'), 'w') as f:
        json.dump(incorrect_cases, f)

    print(f"{stance} {text_type}s: F1 score: {round(f1, 4)}, precision: {round(precision, 4)}, recall: {round(recall, 4)}")
    if log_to_wandb:
        wandb.run.summary[f"{stance}_{text_type}_f1"] = f1
        wandb.run.summary[f"{stance}_{text_type}_precision"] = precision
        wandb.run.summary[f"{stance}_{text_type}_recall"] = recall

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

def main():
    batch_size = 1

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    run_dir_path = os.path.join(data_dir_path, 'topics_minilm_0_2')

    with open(os.path.join(run_dir_path, 'topic_stances.json'), 'r') as f:
        all_topic_stances = json.load(f)

    use_baseline = False
    train_num = 10
    val_num = 10
    dataset_strategy = 'ratio:113'
    if not use_baseline:
        model_name = 'HuggingFaceH4/zephyr-7b-beta'
        prompting_method = 'predict'
        opinion_method = 'template'
        teleprompter = 'prompttune'
        classifier = StanceClassifier(model_name=model_name, prompting_method=prompting_method, opinion_method=opinion_method, backend="dspy", teleprompter=teleprompter)
    else:
        baseline_type = 'annotator'
        if baseline_type == 'neutral' or baseline_type == 'favor':
            class Classifier:
                def __init__(self):
                    self.model_name = 'baseline'
                    self.prompting_method = None
                    self._extra_responses = [None]

                def predict_stances(self, inputs, stance):
                    return [baseline_type] * len(inputs)
            classifier = Classifier()
        elif baseline_type == 'annotator':
            class Classifier:
                def __init__(self):
                    self.model_name = 'annotator'
                    self.prompting_method = None
                    self._extra_responses = [None]

                    self.dfs = {}
                    self.idxs = {}
                    for topic in all_topic_stances['topic_stances']:
                        for stance in topic['stances']:
                            stance_slug = stance.replace(' ', '_')
                            df_path = os.path.join(run_dir_path, f'topic_{topic["topics"][0]}', 'annotator_2', f'label_comments_{stance_slug}.csv')
                            if not os.path.exists(df_path):
                                continue
                            self.dfs[stance_slug] = pl.read_csv(df_path)
                            self.idxs[stance_slug] = 0

                def predict_stances(self, inputs, stance):
                    stance_slug = stance.replace(' ', '_')
                    annotator_df = self.dfs[stance_slug]
                    labels = annotator_df['stance'].to_list()
                    input_labels = [labels[self.idxs[stance_slug]] for _ in range(len(inputs))]
                    self.idxs[stance_slug] += len(inputs)
                    return input_labels

            classifier = Classifier()

    messages = ""

    # start a new wandb run to track this script
    log_to_wandb = True

    if log_to_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="reddit-stance-prompting",
            
            # track hyperparameters and run metadata
            config={
                "model_name": classifier.model_name,
                "prompt_method": classifier.prompting_method,
                "regex": getattr(classifier, 'regex', None),
                "comment_prompt": str(classifier._get_prompt_template(True, True)),
                "submission_prompt": str(classifier._get_prompt_template(False, False)),
                "train_num": train_num,
                "val_num": val_num,
                "dataset_strategy": dataset_strategy,
                "prompt_template": str(messages),
                "teleprompter": classifier.teleprompter,
                "extended_prompt": classifier.get_extended_prompt(),
            }
        )

    comment_f1s = []
    comment_precisions = []
    comment_recalls = []
    submission_f1s = []
    submission_precisions = []
    submission_recalls = []

    # just do the first topic for now
    # all_topic_stances['topic_stances'] = all_topic_stances['topic_stances'][:1]

    for topics_stances in all_topic_stances['topic_stances']:

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        # print(f"Predicting stances for topics: {topics}")

        topic_path = os.path.join(run_dir_path, f'topic_{topics[0]}')

        topic_comments = None
        comment_gold_path = os.path.join(topic_path, 'sample_context_comments_gold_stance.parquet.zstd')
        if os.path.exists(comment_gold_path):
            topic_comments = pl.read_parquet(comment_gold_path)
            topic_comments = topic_comments.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('post_all_text'))

        topic_submissions = None
        submission_gold_path = os.path.join(topic_path, 'sample_context_submissions_gold_stance.parquet.zstd')
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
                do_stance_detection(
                    stance, stance_slug, topic_comments, ['body', 'body_parent', 'post_all_text'], classifier, batch_size, 'comment', 
                    comment_f1s, comment_precisions, comment_recalls, log_to_wandb, topic_path, train_num, val_num, dataset_strategy
                )

                if teleprompter == 'prompttune':
                    classifier.remove_model()

            # if topic_submissions is not None:
            #     print("Predicting stance for submissions")
            #     do_stance_detection(stance, stance_slug, topic_submissions, ['all_text'], classifier, batch_size, 'submission', submission_f1s, log_to_wandb)

    avg_comment_f1 = sum(comment_f1s) / len(comment_f1s)
    avg_comment_precision = sum(comment_precisions) / len(comment_precisions)
    avg_comment_recall = sum(comment_recalls) / len(comment_recalls)

    # avg_submission_f1 = sum(submission_f1s) / len(submission_f1s)
    # avg_f1 = (avg_comment_f1 + avg_submission_f1) / 2
    print(f"Average comment F1 score: {round(avg_comment_f1, 4)}")
    print(f"Average comment precision: {round(avg_comment_precision, 4)}")
    print(f"Average comment recall: {round(avg_comment_recall, 4)}")
    # print(f"Average submission F1 score: {round(avg_submission_f1, 4)}")
    # print(f"Average F1 score: {round(avg_f1, 4)}")

    if log_to_wandb:
        wandb.run.summary["avg_comment_f1"] = avg_comment_f1
        wandb.run.summary["avg_comment_precision"] = avg_comment_precision
        wandb.run.summary["avg_comment_recall"] = avg_comment_recall
        # wandb.run.summary["avg_submission_f1"] = avg_submission_f1
        # wandb.run.summary["avg_f1"] = avg_f1
        wandb.finish()

if __name__ == '__main__':
    main()