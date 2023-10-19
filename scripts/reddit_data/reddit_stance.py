import json
import os

import numpy as np
import tqdm

import utils
import lm

class StanceClassifier:
    def __init__(self, model_name='mistral'):
        self.prompt = """What is the attitude of the sentence: "{text}" to the target "{target}". Response only with a selection from "favor", "against" or "neutral" """

        if model_name == 'mistral':
            self.model = lm.Mistral()
        elif model_name == 'zephyr':
            self.model = lm.Zephyr()

    def _get_model_response(self, prompts):
        response = self.model(prompts)
        return response

    def _predict_stance(self, texts, target):
        prompts = [self.prompt.format(text=text, target=target) for text in texts]
    
        responses = self._get_model_response(prompts)

        responses = [response.strip().lower() for response in responses]

        def parse_response(r):
            stances = ['favor', 'against', 'neutral']
            for stance in stances:
                if stance in r:
                    return stance
                
            return 'error'
        
        responses = [parse_response(r) for r in responses]

        return responses

    def predict_stances(self, texts, target):
        return self._predict_stance(texts, target)

def main():
    batch_size = 2

    topic_stances = [
        (1, 'vaccine mandates'),
        (2, 'government response to protests'),
        (3, 'government response to rising housing costs'),
        (4, 'banning firearms'),
        (5, 'government action on healthcare'),
        (6, 'supporting ukraine'),
        ([7, 13], ['support conservatives', 'support liberals', 'support NDP']),
        (10, 'trudeau'),
        (12, 'government action on inflation'),
        (14, 'drug legalization'),
        (16, 'lgbtq+ rights'),
        (17, 'french language mandates'),
        (20, 'police actions'),
        (21, 'mask mandates'),
        (22, 'electric vehicles'),
        (23, 'carbon tax'),
        (24, 'reproductive rights'),
        (25, 'government policy on china')
    ]

    sample_frac = None

    # Load the data.
    comment_df = utils.get_comment_df()
    submission_df = utils.get_submission_df()

    submission_df['all_text'] = submission_df['title'] + ' ' + submission_df['selftext']

    docs = list(comment_df['body'].values) + list(submission_df['all_text'].values)
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    run_dir_path = os.path.join(data_dir_path, 'topics_minilm_0_2')

    with open(os.path.join(run_dir_path, 'topics.json'), 'r') as f:
        doc_topics = json.load(f)

    stance_classifier = StanceClassifier()

    for topics, stances in topic_stances:

        print(f"Predicting stances for topics: {topics}")

        if isinstance(topics, int):
            topics = [topics]
        if isinstance(stances, str):
            stances = [stances]

        comment_topic_idxs = np.array([idx for idx, topic in enumerate(doc_topics[:len(comment_df)]) if topic in topics])
        # sort idxs by length of comment
        comment_topic_idxs = comment_topic_idxs[np.argsort(comment_df.iloc[comment_topic_idxs]['body'].apply(len).values)]

        submission_topic_idxs = np.array([idx for idx, topic in enumerate(doc_topics[len(comment_df):]) if topic in topics])
        # sort idxs by length of submission
        submission_topic_idxs = submission_topic_idxs[np.argsort(submission_df.iloc[submission_topic_idxs]['all_text'].apply(len).values)]

        for stance in stances:
            print(f"Predicting stance for {stance}")
            print("Predicting stance for comments")
            comment_df[stance] = 'unknown'
            for i in tqdm.tqdm(range(0, len(comment_topic_idxs), batch_size)):
                comment_batch_idxs = comment_topic_idxs[i:min(i+batch_size, len(comment_topic_idxs))]
                comment_batch = comment_df.iloc[comment_batch_idxs]
                stances_batch = stance_classifier.predict_stances(comment_batch['body'].values, stance)
                comment_df.iloc[comment_batch_idxs, comment_df.columns.get_loc(stance)] = stances_batch

            print("Predicting stance for submissions")
            submission_df[stance] = 'unknown'
            for i in tqdm.tqdm(range(0, len(submission_topic_idxs), batch_size)):
                submission_batch_idxs = submission_topic_idxs[i:min(i+batch_size, len(submission_topic_idxs))]
                submission_batch = submission_df.iloc[submission_batch_idxs]
                stances_batch = stance_classifier.predict_stances(submission_batch['all_text'].values, stance)
                submission_df.iloc[submission_batch_idxs, submission_df.columns.get_loc(stance)] = stances_batch

    comment_df.to_parquet(os.path.join(data_dir_path, 'processed_comments_stance.parquet.gzip'), compression='gzip', index=False)
    submission_df.to_parquet(os.path.join(data_dir_path, 'processed_submissions_stance.parquet.gzip'), compression='gzip', index=False)
    
if __name__ == '__main__':
    main()