from outlines import models as outline_models
import outlines.text.generate as outlines_generate
import transformers
import torch

class Mistral:
    def __init__(self):
        self.device = "cuda" # the device to load the model onto
        tokenizer_kwargs = {'use_fast': True, 'padding_side': 'left', 'add_eos_token': True}
        self.guided_model = outline_models.transformers(
            'mistralai/Mistral-7B-Instruct-v0.1', 
            device='cuda', 
            model_kwargs={'torch_dtype': torch.float16, 'device_map': 'auto'}, 
            tokenizer_kwargs=tokenizer_kwargs
        )
        
        # self.tokenizer.pad_token = self.tokenizer.bos_token
        # self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        # self.model.config.pad_token_id = self.model.config.bos_token_id

        # example
        # text = "<s>[INST] What is your favourite condiment? [/INST]"
        # "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        # "[INST] Do you have mayonnaise recipes? [/INST]"


    def _format(self, prompt):
        return f"<s>[INST] {prompt} [/INST]"

    def __call__(self, prompts, regex=None):
        prompts = [self._format(prompt) for prompt in prompts]

        if regex:
            decoded = []
            for prompt in prompts:
                decoded.append(outlines_generate.regex(self.guided_model, regex)(prompt))

        else:
            tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length':1024}

            encodeds = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False, **tokenizer_kwargs)

            model_inputs = encodeds.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            decoded = [d.split("[/INST]")[1] for d in decoded]

        return decoded
    
class Zephyr:
    def __init__(self):
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
        # self.model = transformers.AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")#, use_flash_attention_2=True)

        self.guided_model = outline_models.transformers('HuggingFaceH4/zephyr-7b-alpha', device='cuda', model_kwargs={'torch_dtype': torch.bfloat16, 'device_map': 'auto'})

    def _format(self, prompt):
        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": "You are a helpful chatbot",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self.guided_model.tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def __call__(self, prompts, regex=None):
        prompts = [self._format(prompt) for prompt in prompts]
        if regex:
            decoded = []
            regex_model = outlines_generate.regex(self.guided_model, regex)
            for prompt in prompts:
                decoded.append(regex_model(prompt))

        else:
            encodes = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=max([len(prompt) for prompt in prompts]))
            model_inputs = encodes.to(self.model.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, pad_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded = [d[len(p):] for d, p in zip(decoded, prompts)]
        return decoded
        
def main():
    zephyr = Zephyr()
    res = zephyr(["What is your favourite condiment?"])
    print(res)

    zephyr = Zephyr()
    res = zephyr(["What is your favourite condiment?", "What is your favourite food?"])
    print(res)
    zephyr = None

    mistral = Mistral()
    res = mistral(["What is your favourite condiment?"])
    print(res)

    mistral = Mistral()
    res = mistral(["What is your favourite condiment?", "What is your favourite food?"])
    print(res)

    mistral = None


if __name__ == '__main__':
    main()
