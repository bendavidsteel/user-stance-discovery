import json
import os
import uuid

import dotenv
import numpy as np
import polars as pl
import tqdm
import wandb

from stance import StanceClassifier, StanceDataset, get_stance_f1_score, get_fbeta_score
import utils

from cluster_threads import filter_comments

def get_stance_dataset(stance, stance_slug, df, cols, train_num, val_num, dataset_strategy):
    if not any(f'gold_{stance_slug}' in c for c in df.columns):
        return
    gold_col = [c for c in df.columns if f'gold_{stance_slug}' in c][0]
    inputs = df.select(pl.concat_list(pl.col(cols + [gold_col]))).to_series(0).to_list()
    inputs = utils.process_quotes(inputs)
    dataset = StanceDataset(inputs, stance, train_num=train_num, val_num=val_num, strategy=dataset_strategy)
    return dataset

def eval_stance_detection(stance, dataset, classifier, batch_size, train_num, val_num, tune_general):
    if train_num  + val_num > 0 and not tune_general:
        train_dataset = dataset.get_train_data()
        val_dataset = dataset.get_dev_data()
        classifier.train(train_dataset, val_dataset)

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
                    'target': input.target_stance,
                    'gold_stance': input.gold_stance,
                    'model_output': classifier._extra_responses,
                })

    all_metrics = get_stance_f1_score(gold_stances, stances, return_all=True)

    return all_metrics

def do_stance_detection(stance, stance_slug, topic_path, topic_comments, topic_comments_df, batch_size, stance_focus, chosen_metric):
    output_file_name = f'predicted_{len(topic_comments_df)}_comments.parquet.zstd'
    
    previous_best_metrics = None
    previous_best_metric = 0
    for dir_name in os.listdir(os.path.join(topic_path, stance_slug)):
        # we only want to consider runs that have been evaluated on the same set of comments
        if os.path.exists(os.path.join(topic_path, stance_slug, dir_name, output_file_name)):
            with open(os.path.join(topic_path, stance_slug, dir_name, 'metrics.json'), 'r') as f:
                metrics = json.load(f)
            if chosen_metric not in metrics[stance_focus]:
                continue
            if previous_best_metrics is None or metrics[stance_focus][chosen_metric] > previous_best_metric:
                previous_best_metrics = metrics
                previous_best_metric = metrics[stance_focus][chosen_metric]

    api = wandb.Api(overrides={"project": "reddit-stance-prompting"})
    runs = api.runs(filters={"State": "finished"})

    best_eval_run = None
    best_eval_run_metric = 0
    for run in runs:
        if f"{stance}_comment_{stance_focus}_precision" not in run.summary or run.summary[f"{stance}_comment_{stance_focus}_precision"] <= 0.5:
            continue
        if run.config["prompt_method"] != "predict":
            continue # for now we won't consider chain of thought methods as they're slow
        if "teleprompter" not in run.config:
            continue
        if "tune" in run.config['teleprompter'] and "all_tasks_checkpoint_name" not in run.config:
            continue
        fbeta = run.summary[f'{stance}_comment_{stance_focus}_{chosen_metric}']
        if best_eval_run is None or fbeta > best_eval_run_metric:
            best_eval_run = run
            best_eval_run_metric = fbeta

    if best_eval_run is None or \
        (   
            previous_best_metrics is not None and \
            chosen_metric in previous_best_metrics[stance_focus] and \
            round(best_eval_run_metric, 4) <= round(previous_best_metric, 4) # round to avoid floating point errors
        ):
        return

    train_num = best_eval_run.config['train_num']
    val_num = best_eval_run.config['val_num']
    model_name = best_eval_run.config['model_name']
    model_prompt_template = best_eval_run.config['model_prompt_template'] if 'model_prompt_template' in best_eval_run.config else "{prompt}"
    prompting_method = best_eval_run.config['prompt_method']
    opinion_method = "onestep" if "target issue" in best_eval_run.config["comment_prompt"] else "template"
    teleprompter = best_eval_run.config['teleprompter'] if 'teleprompter' in best_eval_run.config else ""
    tune_general = "tune_general" in best_eval_run.config and best_eval_run.config['tune_general']
    dataset_strategy = best_eval_run.config['dataset_strategy'] if 'dataset_strategy' in best_eval_run.config else "order"

    stance_dataset = get_stance_dataset(stance, stance_slug, topic_comments, ['body', 'body_parent', 'post_all_text'], train_num, val_num, dataset_strategy)
    
    classifier = StanceClassifier(model_name=model_name, model_prompt_template=model_prompt_template, prompting_method=prompting_method, opinion_method=opinion_method, backend="dspy", optimizer=teleprompter)
    classifier.shot_num = train_num + val_num

    if "tune" in teleprompter:
        if "multitask" in teleprompter:
            task = stance_dataset.get_train_data()[0].target_stance.replace(' ', '_')
        else:
            task = "all_tasks"
        # check for saved model checkpoint
        checkpoint_name = best_eval_run.config[f'{task}_checkpoint_name']
        checkpoint_path = os.path.join(".", "model_checkpoints", checkpoint_name)
        if os.path.exists(checkpoint_path):
            classifier.load_model(model_name, checkpoint_path, stance_dataset.get_train_data())
        else:
            raise Exception("No checkpoint found")

    this_metrics = eval_stance_detection(
        stance, stance_dataset, classifier, batch_size, train_num, val_num, tune_general
    )

    if this_metrics[stance_focus]['precision'] > 0.5:
        # evaluate against previous runs again post validation
        best_so_far = True
        for dir_name in os.listdir(os.path.join(topic_path, stance_slug)):
            if os.path.exists(os.path.join(topic_path, stance_slug, dir_name, output_file_name)):
                with open(os.path.join(topic_path, stance_slug, dir_name, 'metrics.json'), 'r') as f:
                    metrics = json.load(f)

                if chosen_metric not in metrics[stance_focus]:
                    continue

                fbeta = metrics[stance_focus][chosen_metric]
                this_fbeta = this_metrics[stance_focus][chosen_metric]
                # round to avoid floating point errors
                if round(this_fbeta, 4) <= round(fbeta, 4):
                    best_so_far = False
                    break
        
        if not best_so_far:
            return

        cols = ['body', 'body_parent', 'post_all_text']
        topic_comments_df = topic_comments_df.clone()
        inputs = topic_comments_df.select(pl.concat_list(pl.col(cols))).to_series(0).to_list()

        dataset = StanceDataset(inputs, stance, train_num=0, val_num=0)
        data = dataset.get_test_data()
        stances = []
        for i in tqdm.tqdm(range(len(data))):
            batch_inputs = data[i:min(len(data), i+1)]
            batch_stances = classifier.predict_stances(batch_inputs, stance)
            stances.extend(batch_stances)

        topic_comments_df = topic_comments_df.with_columns([
            pl.Series(name='stance', values=stances)
        ])

        result_id = best_eval_run.id
        result_dir_path = os.path.join(topic_path, stance_slug, result_id)

        if not os.path.exists(result_dir_path):
            os.mkdir(result_dir_path)

        topic_comments_df.write_parquet(os.path.join(result_dir_path, output_file_name))
        
        with open(os.path.join(result_dir_path, 'metrics.json'), 'w') as f:
            json.dump(this_metrics, f, indent=2)


    classifier.remove_model()

def main():
    dotenv.load_dotenv()

    batch_size = 1

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit')

    experiment = '4sub_2year'

    if experiment == '1sub_1year':
        run_dir_path = os.path.join(data_dir_path, '1sub_1year', 'topics_minilm_0_2')
        subreddits = ['canada']
    elif experiment == '4sub_1year':
        run_dir_path = os.path.join(data_dir_path, '4sub_1year')
        subreddits = ['canada', 'ontario', 'toronto', 'vancouver']
    elif experiment == '4sub_2year':
        run_dir_path = os.path.join(data_dir_path, '4sub_2year', 'topics_minilm_0_0250')
        subreddits = ['canada', 'ontario', 'toronto', 'vancouver']

    with open(os.path.join(run_dir_path, 'topic_stances.json'), 'r') as f:
        all_topic_stances = json.load(f)

    gold_run_dir_path = os.path.join(data_dir_path, '1sub_1year', 'topics_minilm_0_2')
    with open(os.path.join(data_dir_path, '1sub_1year', 'topics_minilm_0_2', 'topic_stances.json'), 'r') as f:
        gold_topic_stances = json.load(f)

    if experiment in ['1sub_1year', '4sub_1year']:
        doc_topics_path = os.path.join(run_dir_path, 'topics.json')
        with open(doc_topics_path, 'r') as f:
            doc_topics = json.load(f)
    elif experiment == '4sub_2year':
        doc_topics_path = os.path.join(run_dir_path, 'topics.npy')
        doc_topics = np.load(doc_topics_path)

    if experiment in ['1sub_1year', '4sub_1year']:
        comments_df = utils.get_comment_df(return_pd=False, subreddits=subreddits)
        submissions_df = utils.get_submission_df(return_pd=False, subreddits=subreddits)

        comments_df, submissions_df = utils.get_parents(comments_df, submissions_df)
        comments_df = comments_df.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('post_all_text'))

        if experiment == '1sub_1year':
            assert len(doc_topics) == len(comments_df) + len(submissions_df)
        elif experiment == '4sub_1year':
            assert len(doc_topics) == len(comments_df)
        comments_df = comments_df.with_columns(pl.Series(name='topic', values=doc_topics[:len(comments_df)]))
    elif experiment == '4sub_2year':
        comment_dir_path = './data/reddit/processed_comments'
        submission_dir_path = './data/reddit/processed_submissions'
        comment_files = [f for f in os.listdir(comment_dir_path) if 'filtered' in f]
        comment_files = sorted(comment_files)
        cum_idx = 0
        comments_df = None
        relevant_topics = []
        for topics_stances in all_topic_stances['topic_stances']:
            relevant_topics.extend(topics_stances['topics'])
        for comment_file in tqdm.tqdm(comment_files, desc='Transforming data'):
            # find equivalent submission file
            submission_file = comment_file.replace('comments', 'submissions')
            if submission_file not in os.listdir(submission_dir_path):
                continue
            file_comment_df = pl.read_parquet(os.path.join(comment_dir_path, comment_file), columns=['permalink', 'id', 'parent_id', 'body', 'author'])
            file_submission_df = pl.read_parquet(os.path.join(submission_dir_path, submission_file), columns=['permalink', 'title', 'selftext'])
            file_comment_df, file_submission_df = utils.get_parents(file_comment_df, file_submission_df)
            file_comment_df = filter_comments(file_comment_df)
            file_comment_df = file_comment_df.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('post_all_text'))
            file_comment_df = file_comment_df.with_columns(pl.Series(name='topic', values=doc_topics[cum_idx:cum_idx+len(file_comment_df)]))
            cum_idx += len(file_comment_df)
            file_comment_df = file_comment_df.filter(pl.col('topic').is_in(relevant_topics))
            if comments_df is None:
                comments_df = file_comment_df
            else:
                comments_df = pl.concat([comments_df, file_comment_df])
        assert cum_idx == len(doc_topics)
        

    for topics_stances in all_topic_stances['topic_stances']:

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        topic_path = os.path.join(run_dir_path, f'topic_{topics[0]}')
        if not os.path.exists(topic_path):
            os.mkdir(topic_path)

        gold_topics = [t for t in gold_topic_stances['topic_stances'] if stances[0] in t['stances']][0]['topics']
        gold_topic_path = os.path.join(gold_run_dir_path, f'topic_{gold_topics[0]}')

        for stance in stances:
            stance_slug = stance.replace(' ', '_')

            topic_comments = None
            topic_submissions = None
            comment_gold_path = os.path.join(gold_topic_path, f'sample_context_{stance_slug}_comments_gold_stance.parquet.zstd')
            if os.path.exists(comment_gold_path):
                topic_comments = pl.read_parquet(comment_gold_path)
                topic_comments = topic_comments.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('post_all_text'))

            submission_gold_path = os.path.join(gold_topic_path, f'sample_context_{stance_slug}_submissions_gold_stance.parquet.zstd')
            if os.path.exists(submission_gold_path):
                topic_submissions = pl.read_parquet(submission_gold_path)
                topic_submissions = topic_submissions.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('all_text'))
            
            if topic_comments is None:
                continue

            if not os.path.exists(os.path.join(topic_path, stance_slug)):
                os.mkdir(os.path.join(topic_path, stance_slug))

            topic_comments_df = comments_df.filter(pl.col('topic').is_in(topics))

            # limit to users with more than 5 comments on this topic
            min_comments = 5
            topic_user_comments_df = topic_comments_df.group_by('author').agg(pl.count('author').alias('num_comments'))
            topic_user_comments_df = topic_user_comments_df.filter(pl.col('num_comments') > min_comments)
            topic_comments_df = topic_comments_df.filter(pl.col('author').is_in(topic_user_comments_df['author']))

            chosen_metric = 'f0.5'
            for stance_focus in ['favor', 'against', 'neutral']:
                do_stance_detection(stance, stance_slug, topic_path, topic_comments, topic_comments_df, batch_size, stance_focus, chosen_metric)


if __name__ == '__main__':
    main()