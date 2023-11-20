import json
import os

import numpy as np
import polars as pl
from statsmodels.stats.inter_rater import fleiss_kappa

def convert_to_cat_counts(examples):
    cat_counts = np.zeros((len(examples), 3))
    for i, annotations in enumerate(examples):
        for annotation in annotations:
            if annotation in ['favor', '2', '1']:
                cat_counts[i, 0] += 1
            elif annotation in ['neutral', '0']:
                cat_counts[i, 1] += 1
            elif annotation in ['against', '-2', '-1']:
                cat_counts[i, 2] += 1
            else:
                raise Exception("Invalid annotation")
    return cat_counts


def join_dfs(main_df, df, col_name):
    if 'id' in df:
        # create row count to keep order
        main_df = main_df.with_columns(pl.Series(name='row_count', values=range(len(main_df))))
        main_df = main_df.join(df[['id', col_name]], on='id')
        main_df = main_df.sort('row_count')
        return main_df.drop('row_count')
    elif 'prompt' in df:
        # these are already in order
        annotations = df[col_name].to_list()
        return main_df.with_columns(pl.Series(name=col_name, values=annotations))

def evaluate_annotations(df, stance, stance_slug, run_dir_path, agreement_method, type_name, topics):
    annotated_columns = [col for col in df.columns if 'annotator' in col and stance_slug in col]
    if len(annotated_columns) >= 2:
        annotations = df.select(pl.concat_list(pl.col(annotated_columns))).to_series(0).to_list()
        cat_counts = convert_to_cat_counts(annotations)
        comment_kappa = round(fleiss_kappa(cat_counts, method='fleiss'), 3)
        # get percentage of examples where all annotators agreed
        percent_unaninimous = np.sum(np.max(cat_counts, axis=1) == len(annotated_columns)) / len(cat_counts)
        percent_disagreement = np.sum(np.logical_and(np.max(cat_counts, axis=1) != len(annotated_columns), cat_counts[:, 1] == 0)) / len(cat_counts)
        print(f"{type_name} Fleiss' kappa for {stance}: {comment_kappa}, percent unanimous: {percent_unaninimous}, percent completely disagree: {percent_disagreement}")

        # save disagreements
        disagreement = np.max(cat_counts, axis=1) != len(annotated_columns)
        disagreement_df = df.filter(pl.Series(disagreement)).select(pl.col(['id', 'prompt'] + annotated_columns))
        disagreement_path = os.path.join(run_dir_path, f'topic_{topics[0]}', f'{stance_slug}_{type_name.lower()}_disagreements.csv')
        valid = False
        if os.path.exists(disagreement_path):
            current_disagreement_df = pl.read_csv(disagreement_path)
            valid = 'id' in current_disagreement_df.columns and set(current_disagreement_df['id'].to_list()) == set(disagreement_df['id'].to_list())
        
        if not valid:
            disagreement_df.write_csv(os.path.join(run_dir_path, f'topic_{topics[0]}', f'{stance_slug}_{type_name.lower()}_disagreements.csv'))
        
        if agreement_method == 'annotator_1':
            annotator_1_col = [col for col in annotated_columns if 'annotator_1' in col][0]
            gold_vals = df[annotator_1_col]

        elif agreement_method == 'adjudication':
            adjudicated_col = [col for col in current_disagreement_df.columns if 'adjudicator' in col][0]
            gold_vals = current_disagreement_df[adjudicated_col]

        df = df.with_columns(pl.Series(name=f"gold_{stance_slug}_{agreement_method}", values=gold_vals))
        
        df = df.drop(annotated_columns)
        df.write_parquet(os.path.join(run_dir_path, f'topic_{topics[0]}', f'sample_context_{type_name.lower()}s_gold_stance.parquet.zstd'))

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    run_dir_path = os.path.join(data_dir_path, 'topics_minilm_0_2')

    with open(os.path.join(run_dir_path, 'topic_stances.json'), 'r') as f:
        all_topic_stances = json.load(f)

    agreement_method = 'annotator_1'

    for topics_stances in all_topic_stances['topic_stances']:

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        filenames = os.listdir(os.path.join(run_dir_path, f'topic_{topics[0]}'))
        annotator_dirs = [f for f in filenames if f.startswith('annotator_')]
        if len(annotator_dirs) < 2:
            continue

        topic_comments = pl.read_parquet(os.path.join(run_dir_path, f'topic_{topics[0]}', 'sample_context_comments.parquet.zstd'))
        topic_submissions = pl.read_parquet(os.path.join(run_dir_path, f'topic_{topics[0]}', 'sample_context_submissions.parquet.zstd'))

        for annotator_dir in annotator_dirs:
            annotated_files = os.listdir(os.path.join(run_dir_path, f'topic_{topics[0]}', annotator_dir))
            if len(annotated_files) == 0:
                continue
            for annotated_file in annotated_files:
                if 'parquet' in annotated_file:
                    annotated_df = pl.read_parquet(os.path.join(run_dir_path, f'topic_{topics[0]}', annotator_dir, annotated_file))
                elif annotated_file.endswith('.csv'):
                    annotated_df = pl.read_csv(os.path.join(run_dir_path, f'topic_{topics[0]}', annotator_dir, annotated_file))

                assert len(annotated_df) == 100

                label_col_names = None
                if 'stance' in annotated_df.columns:
                    label_col_names = ['stance']
                elif any(f'gold_{stance}' in annotated_df.columns for stance in stances):
                    label_col_names = [f'gold_{stance}' for stance in stances if f'gold_{stance}' in annotated_df.columns]
                else:
                    raise Exception("Could figure out label column name")

                annotated_stances = None
                if any(f'gold_{stance}' in annotated_df.columns for stance in stances):
                    annotated_stances = [stance for stance in stances if f'gold_{stance}' in annotated_df.columns]
                elif any(stance.replace(' ', '_') in annotated_file for stance in stances):
                    annotated_stances = [stance for stance in stances if stance.replace(' ', '_') in annotated_file]
                else:
                    raise Exception("Could figure out stance")

                annotated_stance_slugs = [s.replace(' ', '_') for s in annotated_stances]
                if 'comments' in annotated_file:
                    for stance_slug, label_col_name in zip(annotated_stance_slugs, label_col_names):
                        annotated_df = annotated_df.with_columns(pl.col(label_col_name).alias(f'{stance_slug}_{annotator_dir}'))
                        topic_comments = join_dfs(topic_comments, annotated_df, f'{stance_slug}_{annotator_dir}', )

                    if 'prompt' in annotated_df.columns:
                        topic_comments = topic_comments.with_columns([
                            pl.Series(name='prompt', values=annotated_df['prompt'])
                        ])
                elif 'submissions' in annotated_file:
                    for stance_slug, label_col_name in zip(annotated_stance_slugs, label_col_names):
                        annotated_df = annotated_df.with_columns(pl.col(label_col_name).alias(f'{stance_slug}_{annotator_dir}'))
                        topic_submissions = join_dfs(topic_submissions, annotated_df, f'{stance_slug}_{annotator_dir}')
                    if 'prompt' in annotated_df.columns:
                        topic_submissions = topic_submissions.with_columns([
                            pl.Series(name='prompt', values=annotated_df['prompt'])
                        ])


        for stance in stances:
            stance_slug = stance.replace(' ', '_')
            evaluate_annotations(topic_comments, stance, stance_slug, run_dir_path, agreement_method, 'Comment', topics)
            evaluate_annotations(topic_submissions, stance, stance_slug, run_dir_path, agreement_method, 'Submission', topics)
      
        
if __name__ == '__main__':
    main()