import os

import utils
from topics import TopicModel

def main():
    sample_frac = 0.2

    # Load the data.
    comment_df = utils.get_comment_df()
    submission_df = utils.get_submission_df()

    submission_df['all_text'] = submission_df['title'] + ' ' + submission_df['selftext']

    docs = list(comment_df['body'].values) + list(submission_df['all_text'].values)

    # Train the model on the corpus.
    pretrained_model = 'sentence-transformers/all-MiniLM-L12-v2'
    model_name = 'minilm'
    
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    topic_model = TopicModel(
        pretrained_model, 
        model_name, 
        len(docs), 
        data_dir_path, 
        sample_frac=sample_frac,
        min_topic_frac=0.001,
        n_neighbours=30 # higher for less topics
    )
    topic_model.fit(docs)
    if sample_frac:
        topic_model.transform(docs)


if __name__ == '__main__':
    main()