import json
import os

import polars as pl
import tqdm
import wandb

from stance import StanceClassifier, StanceDataset, StancesDataset, TARGET_EXPLANATIONS, TARGET_NAMES, get_stance_f1_score, get_fbeta_score
from utils import process_quotes


def get_stance_dataset(stance, stance_slug, df, cols, train_num, val_num, dataset_strategy, log_to_wandb=False):
    if not any(f'gold_{stance_slug}' in c for c in df.columns):
        return
    gold_col = [c for c in df.columns if f'gold_{stance_slug}' in c][0]
    agreement_method = gold_col.replace(f'gold_{stance_slug}_', '')
    if log_to_wandb:
        wandb.run.config['agreement_method'] = agreement_method
        wandb.run.config[f'{stance_slug}_target_explanation'] = TARGET_EXPLANATIONS[stance]
        wandb.run.config[f'{stance_slug}_target_name'] = TARGET_NAMES[stance]

    inputs = df.select(pl.concat_list(pl.col(cols + [gold_col]))).to_series(0).to_list()

    inputs = process_quotes(inputs)

    dataset = StanceDataset(inputs, stance, train_num=train_num, val_num=val_num, strategy=dataset_strategy)
    return dataset

def do_stance_detection(stance, stance_slug, dataset, classifier, batch_size, text_type, f1s, precisions, recalls, log_to_wandb, topic_path, train_num, val_num, tune_general, general_checkpoint_path):
    if train_num + val_num > 0 and ("multitask" in classifier.teleprompter or not tune_general):
        teleprompter_settings = {}
        if "multitask" in classifier.teleprompter:
            teleprompter_settings['previous_checkpoint_path'] = general_checkpoint_path

        train_dataset = dataset.get_train_data()
        val_dataset = dataset.get_dev_data()
        classifier.train(train_dataset, val_dataset, teleprompter_settings=teleprompter_settings)

        if log_to_wandb:
            wandb.run.config[f'{stance_slug}_teleprompter_settings'] = classifier.teleprompter_settings

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

    metrics = get_stance_f1_score(gold_stances, stances, return_all=True)

    f1 = metrics['macro']['f1']
    precision = metrics['macro']['precision']
    recall = metrics['macro']['recall']

    with open(os.path.join(topic_path, f'{stance_slug}_{text_type}_incorrect_cases.json'), 'w') as f:
        json.dump(incorrect_cases, f)

    print(f"{stance} {text_type}s: F1 score: {round(f1, 4)}, precision: {round(precision, 4)}, recall: {round(recall, 4)}")
    if log_to_wandb:
        beta = 0.5
        wandb.run.summary[f"{stance}_{text_type}_f1"] = f1
        wandb.run.summary[f"{stance}_{text_type}_precision"] = precision
        wandb.run.summary[f"{stance}_{text_type}_recall"] = recall
        wandb.run.summary[f"{stance}_{text_type}_f{beta}"] = metrics['macro'][f'f{beta}']
        wandb.run.summary[f"{stance}_{text_type}_favor_f1"] = metrics['favor']['f1']
        wandb.run.summary[f"{stance}_{text_type}_against_f1"] = metrics['against']['f1']
        wandb.run.summary[f"{stance}_{text_type}_neutral_f1"] = metrics['neutral']['f1']
        wandb.run.summary[f"{stance}_{text_type}_favor_precision"] = metrics['favor']['precision']
        wandb.run.summary[f"{stance}_{text_type}_against_precision"] = metrics['against']['precision']
        wandb.run.summary[f"{stance}_{text_type}_neutral_precision"] = metrics['neutral']['precision']
        wandb.run.summary[f"{stance}_{text_type}_favor_recall"] = metrics['favor']['recall']
        wandb.run.summary[f"{stance}_{text_type}_against_recall"] = metrics['against']['recall']
        wandb.run.summary[f"{stance}_{text_type}_neutral_recall"] = metrics['neutral']['recall']
        wandb.run.summary[f"{stance}_{text_type}_favor_f{beta}"] = metrics['favor'][f'f{beta}']
        wandb.run.summary[f"{stance}_{text_type}_against_f{beta}"] = metrics['against'][f'f{beta}']
        wandb.run.summary[f"{stance}_{text_type}_neutral_f{beta}"] = metrics['neutral'][f'f{beta}']
        

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
    dataset_strategy = 'order'
    tune_general = False
    if not use_baseline:
        model_name = 'berkeley-nest/Starling-LM-7B-alpha'
        # model_name = 'mistralai/Mistral-7B-v0.1'
        model_prompt_template = "GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
        prompting_method = 'predict'
        opinion_method = 'template'
        teleprompter = 'finetune'
        teleprompter_settings = {}
        classifier = StanceClassifier(model_name=model_name, model_prompt_template=model_prompt_template, prompting_method=prompting_method, opinion_method=opinion_method, backend="dspy", teleprompter=teleprompter)
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
                "comment_prompt": str(classifier._get_prompt_template(True, True)) if not use_baseline else None,
                "submission_prompt": str(classifier._get_prompt_template(False, False)) if not use_baseline else None,
                "train_num": train_num,
                "val_num": val_num,
                "dataset_strategy": dataset_strategy,
                "model_prompt_template": classifier.model_prompt_template if not use_baseline else None,
                "teleprompter": classifier.teleprompter if not use_baseline else None,
                "extended_prompt": classifier.get_extended_prompt() if not use_baseline else None,
                "tune_general": tune_general,
            }
        )

    comment_f1s = []
    comment_precisions = []
    comment_recalls = []
    submission_f1s = []
    submission_precisions = []
    submission_recalls = []

    stance_datasets = {}

    for topics_stances in all_topic_stances['topic_stances']:

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        # print(f"Predicting stances for topics: {topics}")

        topic_path = os.path.join(run_dir_path, f'topic_{topics[0]}')

        for stance in stances:
            stance_slug = stance.replace(' ', '_')

            topic_comments = None
            comment_gold_path = os.path.join(topic_path, f'sample_context_{stance_slug}_comments_gold_stance.parquet.zstd')
            if os.path.exists(comment_gold_path):
                topic_comments = pl.read_parquet(comment_gold_path)
                topic_comments = topic_comments.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('post_all_text'))

            topic_submissions = None
            submission_gold_path = os.path.join(topic_path, f'sample_context_{stance_slug}_submissions_gold_stance.parquet.zstd')
            if os.path.exists(submission_gold_path):
                topic_submissions = pl.read_parquet(submission_gold_path)
                topic_submissions = topic_submissions.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('all_text'))
            
            if topic_comments is None:
                continue

            stance_dataset = get_stance_dataset(stance, stance_slug, topic_comments, ['body', 'body_parent', 'post_all_text'], train_num, val_num, dataset_strategy, log_to_wandb)
            stance_datasets[stance_slug] = stance_dataset

    general_checkpoint_path = None
    if (not use_baseline) and "tune" in teleprompter and train_num + val_num > 0 and tune_general:
        stances_dataset = StancesDataset(stance_datasets.values())
        trainset = stances_dataset.get_train_data()
        valset = stances_dataset.get_dev_data()
        classifier.train(trainset, valset, all_tasks=True, teleprompter_settings=teleprompter_settings)
        if "multitask" in teleprompter:
            classifier.remove_model()
            general_checkpoint_path = classifier.checkpoint_path

        if log_to_wandb:
            wandb.run.config['general_teleprompter_settings'] = classifier.teleprompter_settings

    for topics_stances in all_topic_stances['topic_stances']:

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        topic_path = os.path.join(run_dir_path, f'topic_{topics[0]}')

        for stance in stances:
            stance_slug = stance.replace(' ', '_')
            if stance_slug in stance_datasets:
                print("Predicting stance for comments")
                stance_dataset = stance_datasets[stance_slug]
                do_stance_detection(
                    stance, stance_slug, stance_dataset, classifier, batch_size, 'comment', 
                    comment_f1s, comment_precisions, comment_recalls, log_to_wandb, topic_path, train_num, val_num, tune_general, general_checkpoint_path
                )

                if (not use_baseline) and "tune" in teleprompter and train_num + val_num > 0 and ("multitask" in teleprompter or not tune_general):
                    classifier.remove_model()

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