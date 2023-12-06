import os
import re

import dspy
from dspy import dsp
from dspy.datasets.dataset import Dataset as DSPyDataset
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithOptuna, LabeledFewShot, BootstrapFinetune
import torch

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

TARGET_EXPLANATIONS = {
    'vaccine mandates': "Any laws or policies enforcing the use of vaccines, including in specific communities like companies or the army. Includes creating rewards for vaccine use, or creating disincentives for lack of vaccination. Not laws concerning the rest of COVID policy i.e. lockdowns. A personal opinion on vaccines in general counts as neutral on vaccine mandates. But an opinion that others should get vaccines indicates support for vaccine mandates.",
    'renter protections': "Opinion on laws protecting renters - rent control, eviction laws, etc.",
    'liberals': "Opinion on the Liberal Party of Canada. Opinion on an official party representative counts as an opinion on the party, i.e. opinions on Justin Trudeau count towards opinion for Liberals. Liking or disliking another party doesn't imply disliking or liking the party in question.",
    'conservatives': "Opinion on the Conservative Party of Canada. Opinion on an official party representative counts as an opinion on the party, i.e. opinions on Pierre Poilievre or Erin O'Toole count towards opinion for Conservatives. Liking or disliking another party doesn't imply disliking or liking the party in question.",
    'ndp': "Opinion on the NDP Party of Canada. Opinion on an official party representative counts as an opinion on the party, i.e. opinions on Jagmeet Singh count towards opinion for NDP. Liking or disliking another party doesn't imply disliking or liking the party in question.",
    'gun control': "Opinion on regulations on weapons, including any weapon ban - assault rifles, handguns, or any restrictions on gun ownership.",
    'drug decriminalization': "Includes opinions on harm reduction measures - i.e. safe spaces to use, government provided supply - anywhere where the government is adding legal situations to do normally illegal drugs",
    'liberal immigration policy': "Opinion on more open, liberal immigration policy, indicating there should be more immigrants, as opposed to a more restrictive immigration policy, implying there should be fewer immigrants. Views on individual immigrants don't count - just whether the government should allow more or restrict immigration.",
    'canadian aid to ukraine': "Opinion on any government aid to Ukraine from Canada, including financial, arms, or in the form of troops. Individual private citizen volunteers don't count.",
    'funding the cbc': "Opinion on any government funding to the CBC, including any funding cuts or increases.",
    'french language laws': "Favor/against laws requiring that people speak French/must attend French schooling/other laws that promote the use of French or limit the use of English in order to increase the use of French"
}

def map_row(r):
    d = {}
    if r[0] is not None:
        d['comment'] = r[0]

    if r[1] is not None:
        d['parent_comment'] = r[1]
    else:
        d['parent_comment'] = ""

    if r[2] is not None:
        d['post'] = r[2]
    else:
        d['post'] = ""

    d['gold_stance'] = r[3]

    return d

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


def _extract_example(field_names, example, raw_pred):
    example = dsp.Example(example)

    raw_pred = raw_pred.strip()

    fields = [
        {
            'input_variable': f,
            'output_variable': f,
            'name': f.replace('_', ' ').title() + ':'
        }
        for f in field_names
    ]

    idx = 0
    while idx < len(fields):
        if (
            fields[idx]['input_variable'] not in example
            or example[fields[idx]['input_variable']] is None
        ):
            break
        idx += 1

    idx = min(idx, len(fields) - 1)
    while raw_pred != "" and idx < len(fields):
        if idx < len(fields) - 1:
            next_field_name = "\n" + fields[idx + 1]['name']
            offset = raw_pred.find(next_field_name)

            if offset >= 0:
                example[fields[idx]['output_variable']] = raw_pred[:offset].strip().split('---')[0].strip()
                raw_pred = raw_pred[offset + len(next_field_name) :].strip().split('---')[0].strip()()

                idx += 1
            else:
                example[fields[idx]['output_variable']] = raw_pred.strip().split('---')[0].strip()

                raw_pred = ""
                idx += 1
                break

        else:
            assert idx == len(fields) - 1, (idx, len(fields))
            example[fields[idx]['output_variable']] = raw_pred.strip().split('---')[0].strip()

            break

    return example
    

class CommentStanceDetectionSignature(dspy.Signature):
    """Predict the stance of the comment towards the target issue.
    If the comment is directly or indirectly agreeing with the target issue, or opposing or critizing something opposed to the target issue, then the opinion should be agree.
    If the comment is directly or indirectly disagreeing with the target issue, or opposing or critizing something in favor of the target issue, then the opinion should be disagree.
    If the comment is discussing something irrelevant to the target, then the opinion should be neutral"""

    target_opinion = dspy.InputField(desc="""The target issue that is either being agreed with, neutral, or disagreed with.""", prefix="Target Issue: ")
    target_explanation = dspy.InputField(desc="""An explanation of the target issue, may be useful in determining the opinion of the comment.""")
    post = dspy.InputField(desc="""The post being commented on, may be useful in determining what the comment is discussing.""")
    parent_comment = dspy.InputField(desc="""The parent comment being replied to, may be useful in determining the context of the comment.""")
    comment = dspy.InputField(desc="""The comment to determine the opinion of.""")
    opinion = dspy.OutputField(desc="""${agree, disagree, or neutral}""", prefix="Opinion: Therefore the opinion of the comment is ")

    def extract(example, raw_pred):
        field_names = ['target_opinion', 'target_explanation', 'post', 'parent_comment', 'comment', 'opinion']
        return _extract_example(field_names, example, raw_pred)

class TwoStepCommentStanceDetectionSignature(dspy.Signature):
    """Determine the opinion of the comment towards the target issue.
    First determine if there is clearly an opinion specifically about the target present in the comment. 
    If the comment is clearly expressing an opinion specifically about the target, answer yes.
    If the comment is discussing something different from the target, or if it is not clear what the opinion is, answer no.
    Then determine if the opinion is agree or disagree.
    If the comment is directly or indirectly agreeing with the target, or opposing or critizing something opposed to the target, then the opinion should be agree.
    If the comment is directly or indirectly disagreeing with the target, or opposing or critizing something in favor of the target, then the opinion should be disagree."""

    post = dspy.InputField(desc="""The post being commented on, may be useful in determining the relevancy or opinion of the comment""")
    parent_comment = dspy.InputField(desc="""The parent comment being replied to, may be useful in determining the relevancy or opinion of the comment.""")
    comment = dspy.InputField(desc="""The comment to determine the opinion of.""")
    target_opinion = dspy.InputField(desc="""The target that is either being agreed with, disagreed with, or not discussed.""")
    target_explanation = dspy.InputField(desc="""An explanation of the target, may be useful in determining the opinion of the comment.""")
    presence_of_opinion = dspy.OutputField(desc="""Choice between yes and no""")
    opinion = dspy.OutputField(desc="""Choice between agree and disagree.""")

class YesNoCommentStanceDetectionSignature(dspy.Signature):
    """Is the comment agreeing with the target opinion?
    If the comment is directly or indirectly agreeing with the opinion, or opposing or critizing something opposed to the opinion, then the answer is yes.
    If the comment is not agreeing with the opinion, or if the comment is not discussing the opinion, then the answer is no."""

    target_opinion = dspy.InputField(desc="""The target opinion.""")
    target_explanation = dspy.InputField(desc="""An explanation of the target opinion, may be useful in determining the opinion of the comment.""")
    post = dspy.InputField(desc="""The post being commented on, may be useful in determining the opinion of the comment""")
    parent_comment = dspy.InputField(desc="""The parent comment being replied to, may be useful in determining the opinion of the comment.""")
    comment = dspy.InputField(desc="""The comment to determine the opinion of.""")
    answer = dspy.OutputField(desc="""Answer yes or no""")

    def extract(example, raw_pred):
        field_names = ['target_opinion', 'target_explanation', 'post', 'parent_comment', 'comment', 'answer']
        return _extract_example(field_names, example, raw_pred)

class PostStanceDetectionSignature(dspy.Signature):
    """Determine the stance of the post towards the target."""

    target = dspy.InputField(desc="""The target of the stance detection.""")
    target_explanation = dspy.InputField(desc="""An explanation of the target, may be useful in determining the stance of the post.""")
    post = dspy.InputField(desc="""The post being commented on.""")
    parent_comment = dspy.InputField(desc="""The comment being replied to.""")
    comment = dspy.InputField(desc="""The comment to determine the stance of.""")
    stance = dspy.OutputField(desc="""Choice between favor, neutral, and against.""")



class ChainOfThoughtForOneStepOpinion(dspy.Predict):
    def __init__(self, signature, **config):
        super().__init__(signature, **config)

        signature = self.signature
        *keys, last_key = signature.kwargs.keys()

        rationale = dsp.Type(desc="""${predict the stance}""", 
                             prefix="Reasoning: Take a deep breath, and work on this problem step-by-step. Let's consider first if the comment is clearly expressing a stance on the target issue. ")
        self.extended_prompt = rationale.prefix + rationale.desc

        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update({'rationale': rationale, last_key: signature.kwargs[last_key]})
        
        self.extended_signature = dsp.Template(signature.instructions, **extended_kwargs)
    
    def forward(self, **kwargs):
        signature = self.extended_signature
        
        return super().forward(signature=signature, **kwargs)


class DSPyStanceDataset(DSPyDataset):
    def __init__(self, examples, target, target_explanation, train_num, val_num):
        ds = []
        for ex in examples:
            d = ex.copy()
            d['target_opinion'] = target
            d['target_explanation'] = target_explanation
            ds.append(d)
        self._train = ds[:train_num]
        self._dev = ds[train_num:train_num + val_num]
        self._test = ds[train_num + val_num:]
        super().__init__(self, train_size=len(self._train), dev_size=len(self._dev), test_size=len(self._test))
        self.do_shuffle = False

class StanceDataset:
    def __init__(self, examples, target, train_num=None, val_num=None, backend='dspy'):
        self.backend = backend
        self.train_num = train_num
        self.val_num = val_num
        examples = [map_row(r) for r in examples]
        if self.backend == 'dspy':
            self.dataset = DSPyStanceDataset(examples, target, TARGET_EXPLANATIONS[target], train_num, val_num)
        elif self.backend == 'hf':
            self.dataset = examples

    def get_test_data(self):
        if self.backend == 'dspy':
            return [x.with_inputs('post', 'parent_comment', 'comment', 'target_opinion', 'target_explanation') for x in self.dataset.test]
        elif self.backend == 'hf':
            return self.dataset[self.train_num + self.val_num:]
        
    def get_train_data(self):
        if self.backend == 'dspy':
            return [x.with_inputs('post', 'parent_comment', 'comment', 'target_opinion', 'target_explanation') for x in self.dataset.train]
        elif self.backend == 'hf':
            return self.dataset[:self.train_num]

    def get_dev_data(self):
        if self.backend == 'dspy':
            return [x.with_inputs('post', 'parent_comment', 'comment', 'target_opinion', 'target_explanation') for x in self.dataset.dev]
        elif self.backend == 'hf':
            return self.dataset[self.train_num:self.train_num + self.val_num]


class StanceClassifier:
    def __init__(self, model_name='zephyr', prompting_method='predict', opinion_method='twostep', backend='dspy', teleprompter='bootstrap'):
        self.backend = backend
        if self.backend == 'hf':
            if model_name == 'mistral':
                self.model = lm.Mistral()
            elif model_name == 'zephyr':
                self.model = lm.Zephyr()
            self.model_name = self.model.model_name

            for response in PROMPT_RESPONSES:
                if response['name'] == prompting_method:
                    self.regex = response['regex']
                    break
            else:
                raise ValueError(f"Invalid response type: {prompting_method}")

        elif self.backend == 'dspy':
            self.model_name = model_name
            if "gpt" in model_name:
                self.model = dspy.OpenAI(self.model_name, os.environ['OPENAI_API_KEY'])
            else:
                self.model = dspy.HFModel(self.model_name, model_kwargs={'torch_dtype': torch.bfloat16, 'use_flash_attention_2': True, 'device_map': 'auto'})
            dspy.settings.configure(lm=self.model)
        
        self.teleprompter = teleprompter
        self.prompting_method = prompting_method
        self.opinion_method = opinion_method
        self.classifier = None
        self.shot_num = 0

    def _get_prompt_template(self, is_comment, is_replying):
        if self.backend == 'hf':
            return get_prompt_template(is_comment, is_replying)
        elif self.backend == 'dspy':
            if self.opinion_method == 'onestep':
                if is_comment:
                    return CommentStanceDetectionSignature({'demos':[]})
                else:
                    return PostStanceDetectionSignature({'demos':[]})
            elif self.opinion_method == 'twostep':
                TwoStepCommentStanceDetectionSignature({'demos': []})
            elif self.opinion_method == 'yesno':
                return YesNoCommentStanceDetectionSignature({'demos': []})

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
        self._extra_responses = []
        if self.backend == 'hf':
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
        elif self.backend == 'dspy':
            responses = []
            for input in texts:
                classifier, config = self._get_classifier()

                if self.opinion_method != 'yesno':
                    if self.shot_num == 0:
                        self.classifier = classifier

                    response = self.classifier(config=config, **input.inputs()._store)

                    extras = {'opinion': response.opinion}
                    if 'rationale' in response.keys():
                        extras['rationale'] = response.rationale
                        self._extra_responses.append(extras)

                    stance = _parse_opinion_answer(response.opinion)
                elif self.opinion_method == 'yesno':
                    if self.shot_num == 0:
                        self.agree_classifier = classifier
                        self.disagree_classifier = classifier

                    inputs = input.inputs()._store
                    agree_inputs = inputs.copy()
                    agree_inputs['target_opinion'] = f"Support for {inputs['target_opinion']}"
                    agree_prediction = self.agree_classifier(config=config, **agree_inputs)
                    disagree_inputs = inputs.copy()
                    disagree_inputs['target_opinion'] = f"Against {inputs['target_opinion']}"
                    disagree_prediction = self.disagree_classifier(config=config, **disagree_inputs)

                    extras = {
                        'agree_answer': agree_prediction.answer,
                        'disagree_answer': disagree_prediction.answer,
                    }

                    if 'rationale' in agree_prediction.keys():
                        extras['agree_rationale'] = agree_prediction.rationale
                        extras['disagree_rationale'] = disagree_prediction.rationale

                    self._extra_responses.append(extras)
                    
                    agree_response = agree_prediction.answer
                    disagree_response = disagree_prediction.answer

                    agree_answer = _parse_yesno_answer(agree_response)
                    disagree_answer = _parse_yesno_answer(disagree_response)

                    assert agree_answer == 'yes' or agree_answer == 'no'
                    assert disagree_answer == 'yes' or disagree_answer == 'no'

                    if agree_answer == 'yes' and disagree_answer == 'no':
                        stance = 'favor'
                    elif agree_answer == 'no' and disagree_answer == 'yes':
                        stance = 'against'
                    elif agree_answer == 'no' and disagree_answer == 'no':
                        stance = 'neutral'
                    else:
                        stance = 'neutral'

                assert stance in ['favor', 'against', 'neutral']

                responses.append(stance)

        return responses

    def predict_stances(self, texts, target=None):
        if target is not None:
            self.target = target
        return self._predict_stance(texts, self.target)
    
    def set_target(self, target):
        self.target = target

    def train(self, trainset, valset=None):

        self.shot_num = len(trainset)

        if self.opinion_method == 'yesno':
            # Validation logic: check that the predicted answer is correct.
            # Also check that the retrieved context does actually contain that answer.
            def validate_context_and_answer(example, pred, trace=None):
                pred_answer = _parse_yesno_answer(pred.answer)
                return example.answer == pred_answer
        else:
            # Validation logic: check that the predicted stance is correct.
            # Also check that the retrieved context does actually contain that stance.
            def validate_context_and_answer(example, pred, trace=None):
                pred_stance = _parse_opinion_answer(pred.opinion)
                return example.opinion == pred_stance

        # Compile!
        classifier, config = self._get_classifier()
        class StanceModule(dspy.Module):
            def __init__(self):
                self.classifier = classifier
                super().__init__()
            
            def forward(self, **kwargs):
                kwargs.update(config)
                return self.classifier(**kwargs)

        if self.opinion_method == 'yesno':

            def convert_inputs(ex, new_set, agree):
                store = ex._store.copy()
                store['target_opinion'] = f"Support for {store['target_opinion']}" if agree else f"Against {store['target_opinion']}"
                stance = 'favor' if agree else 'against'
                store['answer'] = 'yes' if ex._store['gold_stance'] == stance else 'no'
                new_ex = dspy.Example(**store)
                new_set.append(new_ex.with_inputs('post', 'parent_comment', 'comment', 'target_opinion', 'target_explanation'))

            agree_trainset = []
            disagree_trainset = []
            for ex in trainset:
                convert_inputs(ex, agree_trainset, agree=True)
                convert_inputs(ex, disagree_trainset, agree=False)

            agree_valset = []
            disagree_valset = []
            for ex in valset:
                convert_inputs(ex, agree_valset, agree=True)
                convert_inputs(ex, disagree_valset, agree=False)

            # Set up a basic teleprompter, which will compile our RAG program.
            if len(valset) == 0:
                if self.teleprompter == 'bootstrap':
                    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
                elif self.teleprompter == 'labelled':
                    teleprompter = LabeledFewShot(k=len(trainset))
                self.agree_classifier = teleprompter.compile(StanceModule(), trainset=agree_trainset)
                self.disagree_classifier = teleprompter.compile(StanceModule(), trainset=disagree_trainset)
            else:
                teleprompter = BootstrapFewShotWithOptuna(metric=validate_context_and_answer)
                self.agree_classifier = teleprompter.compile(StanceModule(), max_demos=len(agree_trainset), trainset=agree_trainset, valset=agree_valset)
                self.disagree_classifier = teleprompter.compile(StanceModule(), max_demos=len(disagree_trainset), trainset=disagree_trainset, valset=disagree_valset)

        else:
            for ex in trainset + valset:
                ex.opinion = ex.gold_stance

            if len(valset) == 0:
                if self.teleprompter == 'bootstrap':
                    teleprompter = BootstrapFewShot(metric=validate_context_and_answer, max_labeled_demos=len(trainset), max_bootstrapped_demos=len(trainset))
                    args = (StanceModule(),)
                    kwargs = {'trainset': trainset}
                elif self.teleprompter == 'finetune':
                    teleprompter = BootstrapFinetune(metric=validate_context_and_answer)
                    labelled_teleprompter = LabeledFewShot(k=len(trainset))
                    teleprompter.teleprompter = BootstrapFewShot(metric=validate_context_and_answer,
                                             max_bootstrapped_demos=999999,
                                             max_labeled_demos=len(trainset))
                    teacher = labelled_teleprompter.compile(StanceModule(), trainset=trainset)
                    args = (StanceModule(),)
                    kwargs = {'teacher': teacher, 'trainset': trainset, 'target': self.model_name, 'bsize': 2, 'bf16': True, 'peft': True}
                self.classifier = teleprompter.compile(*args, **kwargs)
            else:
                teleprompter = BootstrapFewShotWithOptuna(metric=validate_context_and_answer, max_labeled_demos=len(trainset), max_bootstrapped_demos=len(trainset))
                self.classifier = teleprompter.compile(StanceModule(), max_demos=len(trainset), trainset=trainset, valset=valset)

    def _get_classifier(self, comment=True):
        if self.opinion_method == 'onestep':
            if comment:
                signature = CommentStanceDetectionSignature
            else:
                signature = PostStanceDetectionSignature
        elif self.opinion_method == 'twostep':
            if comment:
                signature = TwoStepCommentStanceDetectionSignature
            else:
                signature = PostStanceDetectionSignature
        elif self.opinion_method == 'yesno':
            signature = YesNoCommentStanceDetectionSignature

        if self.prompting_method == 'predict':
            classifier = dspy.Predict(signature)
            if 'gpt' in self.model_name:
                config = {}
            else:
                config = {'max_tokens': 4}
        elif self.prompting_method == 'multicomparison':
            classifier = MultiComparison(signature)
            if 'gpt' in self.model_name:
                config = {}
            else:
                config = {'max_tokens': 4}
        elif self.prompting_method == 'chainofthought':
            classifier = dspy.ChainOfThought(signature)
            if 'gpt' in self.model_name:
                config = {}
            else:
                config = {'max_tokens': 400}
        elif self.prompting_method == 'chainofthoughtstance':
            classifier = ChainOfThoughtForOneStepOpinion(signature)
            if 'gpt' in self.model_name:
                config = {}
            else:
                config = {'max_tokens': 400}
        elif self.prompting_method == 'multichaincomparison':
            classifier = MultiChainComparison(signature)
            config = {}
        else:
            raise ValueError(f"Invalid prompting method: {self.prompting_method}")
        
        self.prompting_text = getattr(classifier, "extended_prompt", None)

        return classifier, config
    
    def get_extended_prompt(self):
        self._get_classifier()
        return self.prompting_text
    
def _parse_yesno_answer(response):
    response = response.split('\n')[0].lower()
    if 'yes' in response and not 'no' in response:
        return 'yes'
    elif 'no' in response and not 'yes' in response:
        return 'no'
    else:
        if any(a in response.lower() for a in ['favor', 'agree', 'support']):
            return 'yes'
        elif any(a in response.lower() for a in ['against', 'disagree', 'unclear']):
            return 'no'
        else:
            return 'no'
        
def _parse_opinion_answer(opinion):
    opinion = opinion.split('\n')[0].lower()
    words = re.findall(r"(\w+)", opinion)
    def get_stance(word):
        if any(a in word for a in ['favor', 'agree', 'support']) and not 'disagree' in word:
            return 'favor'
        elif any(a in word for a in ['against', 'disagree']):
            return 'against'
        elif 'neutral' in word:
            return 'neutral'
        else:
            return None
    for word in words:
        stance = get_stance(word)
        if stance is not None:
            return stance
    else:
        return 'neutral'


class MultiChainComparison(dspy.Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):
        super().__init__()

        self.M = M
        signature = dspy.Predict(signature).signature
        *keys, last_key = signature.kwargs.keys()

        extended_kwargs = {key: signature.kwargs[key] for key in keys}

        for idx in range(M):
            candidate_type = dsp.Type(prefix=f"Student Attempt #{idx+1}:", desc="${reasoning attempt}")
            extended_kwargs.update({f'reasoning_attempt_{idx+1}': candidate_type})
        
        rationale_type = dsp.Type(prefix="Accurate Reasoning: Thank you everyone. Let's now holistically", desc="${corrected reasoning}")
        extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})

        signature = dsp.Template(signature.instructions, **extended_kwargs)
        self.predict = dspy.Predict(signature, temperature=temperature, **config)
        self.last_key = last_key

        self.chainofthought = dspy.ChainOfThought(signature, temperature=temperature, **config)
    
    def forward(self, **kwargs):
        attempts = []

        for _ in range(self.M):
            c = self.chainofthought(**kwargs)
            rationale = c.rationale.strip().split('\n')[0].strip()
            answer = _parse_opinion_answer(c[self.last_key])
            attempts.append(f"«{rationale} I'm not sure but my prediction is {answer}»")

        assert len(attempts) == self.M, len(attempts)

        kwargs = {**{f'reasoning_attempt_{idx+1}': attempt for idx, attempt in enumerate(attempts)}, **kwargs}
        return self.predict(**kwargs)
    
class MultiComparison(dspy.Module):
    def __init__(self, signature, M=3, temperature=0.4, **config):
        super().__init__()

        self.M = M
        self.predict = dspy.Predict(signature, temperature=temperature, **config)
    
    def forward(self, **kwargs):
        stance_counts = {}
        completions = []
        for _ in range(self.M):
            c = self.predict(**kwargs)
            completions.append(c)
            stance = _parse_opinion_answer(c.opinion)
            stance_counts[stance] = stance_counts.get(stance, 0) + 1

        stance_counts = sorted(stance_counts.items(), key=lambda x: x[1], reverse=True)
        stance = stance_counts[0][0]
        return [c for c in completions if _parse_opinion_answer(c.opinion) == stance][0]
