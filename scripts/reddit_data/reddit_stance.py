import os

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils

class StanceClassifier:
    def __init__(self, targets):
        self.prompt = """<s>[INST] What is the attitude of the sentence: "{text}" to the target "{target}". Response only with a selection from "favor", "against" or "neutral" [/INST]</s>"""

        # TODO extend with CoT and 1 shot

        self.device = "cuda" # the device to load the model onto

        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        # convert to fp16
        self.model = self.model.half()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

        # example
        # text = "<s>[INST] What is your favourite condiment? [/INST]"
        # "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        # "[INST] Do you have mayonnaise recipes? [/INST]"

        self.targets = targets

    def _get_model_response(self, prompts):
        encodeds = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False)

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids[:,model_inputs['input_ids'].shape[1]:])
        return decoded

    def _predict_stance(self, texts, target):
        prompts = [self.prompt.format(text=text, target=target) for text in texts]
        if len(prompts) == 1:
            prompts = prompts[0]
        else:
            raise NotImplementedError("Batched inference not implemented yet")
    
        response = self._get_model_response(prompts)

        if len(response) == 1:
            response = response[0]

        response = response.strip().lower()

        if "favor" in response:
            response = "favor"
        elif "against" in response:
            response = "against"
        elif "neutral" in response:
            response = "neutral"

        return response

    def predict_stances(self, texts):
        return [self._predict_stance(texts, target) for target in self.targets]

def main():
    batch_size = 1

    targets = ['vaccine', 'covid', 'mask', 'lockdown', 'social distancing']
    stance_classifier = StanceClassifier(targets)

    data_dir_path = utils.get_data_dir_path()
    comment_df = utils.get_comment_df()

    for i in range(0, len(comment_df), batch_size):
        comment_batch = comment_df[i:min(i+batch_size, len(comment_df))]
        stances_batch = stance_classifier.predict_stances(comment_batch['body'].values)
        comment_df.loc[i:min(i+batch_size, len(comment_df)), targets] = stances_batch

    comment_df.to_parquet(os.path.join(data_dir_path, 'processed_comments_stance.parquet.gzip'), compression='gzip', index=False)

    submission_df = utils.get_submission_df()

    submission_df['all_text'] = submission_df['title'] + ' ' + submission_df['selftext']

    for i in range(0, len(submission_df), batch_size):
        submission_batch = submission_df[i:min(i+batch_size, len(submission_df))]
        stances_batch = stance_classifier.predict_stances(submission_batch['all_text'].values)
        submission_df.loc[i:min(i+batch_size, len(submission_df)), targets] = stances_batch

    submission_df.to_parquet(os.path.join(data_dir_path, 'processed_submissions_stance.parquet.gzip'), compression='gzip', index=False)
    
if __name__ == '__main__':
    main()