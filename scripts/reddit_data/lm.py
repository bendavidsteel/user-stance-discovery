import transformers
import torch

class Mistral:
    def __init__(self):
        self.device = "cuda" # the device to load the model onto

        self.model = transformers.AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16, use_flash_attention_2=True, device_map="auto")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.tokenizer.padding_side  = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # example
        # text = "<s>[INST] What is your favourite condiment? [/INST]"
        # "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        # "[INST] Do you have mayonnaise recipes? [/INST]"


    def _format(self, prompt):
        return f"<s>[INST] {prompt} [/INST]"

    def __call__(self, prompts):
        encodeds = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=1024, pad_token=self.tokenizer.eos_token)

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids[:,model_inputs['input_ids'].shape[1]:])
        return decoded
    
class Zephyr:
    def __init__(self):
        self.pipe = transformers.pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto", use_flash_attention_2=True)

    def _format(self, prompt):
        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": "You are a helpful chatbot",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def __call__(self, prompts):
        for prompt in prompts:
            prompt = self._format(prompt)
            outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, pad_token_id=self.pipe.tokenizer.eos_token_id)
            return outputs[0]["generated_text"]
