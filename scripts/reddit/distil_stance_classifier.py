import json
import os

import datasets
import transformers

import utils

def main():

    batch_size = 1

    sample_frac = 0.01
    sample_per_topic = 10

    # Load the data.
    comment_df = utils.get_comment_df(return_pd=False)
    submission_df = utils.get_submission_df(return_pd=False)

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'reddit', '1sub_1year')

    run_dir_path = os.path.join(data_dir_path, 'topics_minilm_0_2')

    with open(os.path.join(run_dir_path, 'topic_stances.json'), 'r') as f:
        all_topic_stances = json.load(f)

    with open(os.path.join(run_dir_path, 'topics.json'), 'r') as f:
        doc_topics = json.load(f)

    comment_df['topic'] = doc_topics[:len(comment_df)]
    submission_df['topic'] = doc_topics[len(comment_df):]

    if sample_frac is not None:
        comment_df = comment_df.sample(frac=sample_frac, random_state=0)
        submission_df = submission_df.sample(frac=sample_frac, random_state=0)

    submission_df = submission_df.with_columns(pl.concat_str([pl.col('title'), pl.col('selftext')], separator=' ').alias('all_text'))

    stance_classifier = StanceClassifier(batch_size=batch_size)

    for topics_stances in all_topic_stances['topic_stances']:

        print(f"Predicting stances for topics: {topics}")

        topics = topics_stances['topics']
        stances = topics_stances['stances']

        topic_comments = comment_df.filter(pl.col('topic').is_in(topics))
        topic_submissions = submission_df.filter(pl.col('topic').is_in(topics))

        if sample_per_topic is not None:
            # we want a deterministic sample
            n_rows = topic_comments.select(pl.count()).collect().item()
            topic_comments = topic_comments.take_every(math.floor(n_rows / sample_per_topic))
            n_rows = topic_submissions.select(pl.count()).collect().item()
            topic_submissions = topic_submissions.take_every(math.floor(n_rows / sample_per_topic))

        # sort idxs by length of comment
        topic_comments = topic_comments.with_columns(pl.col('body').str.len_chars().alias('body_len'))
        topic_comments = topic_comments.sort('body_len')

        # sort idxs by length of submission
        topic_submissions = topic_submissions.with_columns(pl.col('all_text').str.len_chars().alias('all_text_len'))
        topic_submissions = topic_submissions.sort('all_text_len')

        for stance in stances:
            print(f"Predicting stance for {stance}")
            print("Predicting stance for comments")
            comment_df = comment_df.with_columns(pl.col(stance).alias(stance))
            for i in tqdm.tqdm(range(0, len(topic_comments), batch_size)):
                comment_batch = topic_comments.slice(i,min(i+batch_size, len(topic_comments)))
                stances_batch = stance_classifier.predict_stances(comment_batch['body'].values, stance)
                topic_comments.iloc[comment_batch.index, comment_df.columns.get_loc(stance)] = stances_batch

            print("Predicting stance for submissions")
            submission_df[stance] = 'unknown'
            for i in tqdm.tqdm(range(0, len(topic_submissions), batch_size)):
                submission_batch = topic_submissions.slice(i,min(i+batch_size, len(topic_submissions)))
                stances_batch = stance_classifier.predict_stances(submission_batch['all_text'].values, stance)
                topic_submissions.iloc[submission_batch.index, submission_df.columns.get_loc(stance)] = stances_batch

        

            ds = datasets.Dataset.from_generator()

            # use teacher to classify examples (not from the gold set)

            # use auto labelled dataset to train student
            tokenizer = transformers.AutoTokenizer.from_pretrained('distilroberta-base')
            model = transformers.AutoModelForSequenceClassification.from_pretrained('distilroberta-base')

            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True)

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=4,
                remove_columns=["text"],
            )

            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=tokenizer,
            )

            trainer.train()

            # load the gold set

            # evaluate on the gold set

        topic_comments.to_parquet(os.path.join(data_dir_path, 'topics_minilm_0_2', f'topic_{topics[0]}', 'context_comments_stance.parquet.gzip'), compression='gzip', index=False)
        topic_submissions.to_parquet(os.path.join(data_dir_path, 'topics_minilm_0_2', f'topic_{topics[0]}', 'context_submissions_stance.parquet.gzip'), compression='gzip', index=False)
    

if __name__ == '__main__':
    main()