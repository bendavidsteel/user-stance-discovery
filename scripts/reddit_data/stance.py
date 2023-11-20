import dspy

import lm

PROMPT_RESPONSES = [
    {
        'name': 'choice',
        'regex': r"(favor|neutral|against)"
    },
    {
        'name': 'thought',
        'regex': r"Reason: [^\.]*\. Stance: (favor|neutral|against)"
    }
]

def get_prompt_template(is_comment, is_replying):
    if is_comment and is_replying:
        prompt = """Given a post: "{post}", and a parent comment replying to that post: "{parent_comment}", what is the attitude of this replying comment: "{comment}" to the target "{target}"?"""
    if is_comment and not is_replying:
        prompt = """Given a post: "{post}", what is the attitude of this comment: "{comment}" to the target "{target}"?"""
    if not is_comment:
        prompt = """What is the attitude of this post: "{post}" to the target "{target}"?"""

    # prompt += """ Respond only with one of the following: strongly against, against, neutral, favor, strongly favor: """
    prompt += """ Respond only with one of the following: against, neutral, favor: """
    return prompt

class StanceDataset(dspy.Dataset):
    def __init__(self, examples, target):
        self._dev = [dict(comment=ex['comment'], parent_comment=ex['parent_comment'], post=ex['post'], target=target) for ex in examples]

class StanceDetectionSignature(dspy.Signature):
    """Determine the stance of the comment towards the target."""

    post = dspy.InputField(desc="The post being commented on.")
    parent_comment = dspy.InputField(desc="The comment being replied to.")
    comment = dspy.InputField(desc="The comment to determine the stance of.")
    target = dspy.InputField(desc="The target of the stance detection.")
    stance = dspy.OutputField(desc="Choice between favor, neutral, and against.")

class StanceClassifier:
    def __init__(self, model_name='mistral', response_type='choice', prompt_type='base'):
        if model_name == 'mistral':
            # self.model = lm.Mistral()
            self.model = dspy.HFModel('mistralai/Mistral-7B-Instruct-v0.1')
        elif model_name == 'zephyr':
            # self.model = lm.Zephyr()
            self.model = dspy.HFModel('HuggingFaceH4/zephyr-7b-beta')

        self.classifier = dspy.Predict(StanceDetectionSignature)

        self.model_name = self.model.model_name

        self.prompt_type = prompt_type

        for response in PROMPT_RESPONSES:
            if response['name'] == response_type:
                self.regex = response['regex']
                break
        else:
            raise ValueError(f"Invalid response type: {response_type}")

    def _get_prompt_template(self, is_comment, is_replying):
        return get_prompt_template(is_comment, is_replying)

    def _format_text_to_prompt(self, text, target):
        is_comment, is_replying = False, False
        if "comment" in text:
            is_comment = True
        if "parent_comment" in text:
            is_replying = True
        prompt = self._get_prompt_template(is_comment, is_replying)
        return prompt.format(**text, target=target)

    def _get_model_response(self, prompts, regex):
        response = self.model(prompts, regex=regex)
        return response

    def _predict_stance(self, texts, target):
        prompts = [self._format_text_to_prompt(text, target) for text in texts]
    
        responses = self._get_model_response(prompts, self.regex)

        def get_choice(r):
            if r.endswith('favor'):
                return 'favor'
            elif r.endswith('against'):
                return 'against'
            elif r.endswith('neutral'):
                return 'neutral'

        responses = [get_choice(r) for r in responses]

        return responses

    def predict_stances(self, texts, target=None):
        if target is not None:
            self.target = target
        return self._predict_stance(texts, self.target)
    
    def set_target(self, target):
        self.target = target