from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftConfig, PeftModel
import torch
from tqdm import tqdm
import dsp
from dsp import HFModel
from torch.utils.data import DataLoader

class Prompttune:

    def compile(self, student, model_name, trainset, valset, lr=3e-2, num_epochs=50, batch_size=1):
        
        signature = student.predictors()[0].signature
        prompt = signature(dsp.Example(demos=[], **{'target_opinion': trainset[0]['target_opinion'], 'target_explanation': trainset[0]['target_explanation']})).strip()
        dataset_name = trainset[0].target_opinion

        def ex_to_dict(ex):
            ex = dict(ex)
            completion = ex.pop(signature.fields[-1].output_variable)
            prompt = signature.query(dsp.Example(demos=[], **ex)).strip()
            return dict(prompt=prompt, completion=completion)

        train = [ex_to_dict(ex) for ex in trainset]
        val = [ex_to_dict(ex) for ex in valset]
        dataset = Dataset.from_list(train + val).train_test_split(test_size=len(val))

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        num_virtual_tokens = len(tokenizer.encode(prompt))

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text=prompt,
            tokenizer_name_or_path=model_name,
        )
        checkpoint_name = f"{dataset_name}_{model_name}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
            "/", "_"
        )
        text_column = "prompt"
        label_column = "completion"

        batch_size = 1

        def preprocess_function(examples):
            batch_size = len(examples[text_column])
            inputs = examples[text_column]
            targets = examples[label_column]
            model_inputs = tokenizer(inputs)
            labels = tokenizer(targets)
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i] # + [tokenizer.pad_token_id]
                # print(i, sample_input_ids, label_input_ids)
                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
            
            max_length = max(len(x) for x in model_inputs["input_ids"])
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                    "attention_mask"
                ][i]
                labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
                model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )


        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=True)
        model = get_peft_model(model, peft_config)

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["test"]

        # train_dataloader = DataLoader(
        #     train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        # )
        # eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=(len(train_dataloader) * num_epochs),
        # )

        device = 'cuda'
        # model = model.to(device)

        # for epoch in range(num_epochs):
        #     model.train()
        #     total_loss = 0
        #     for step, batch in enumerate(tqdm(train_dataloader)):
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         outputs = model(**batch)
        #         loss = outputs.loss
        #         total_loss += loss.detach().float()
        #         loss.backward()
        #         optimizer.step()
        #         lr_scheduler.step()
        #         optimizer.zero_grad()

        #     model.eval()
        #     eval_loss = 0
        #     eval_preds = []
        #     for step, batch in enumerate(tqdm(eval_dataloader)):
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         with torch.no_grad():
        #             outputs = model(**batch)
        #         loss = outputs.loss
        #         eval_loss += loss.detach().float()
        #         get_accuracy = True
        #         if get_accuracy:
        #             outputs = model.generate(
        #                 input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=10, eos_token_id=3
        #             )
        #             output = tokenizer.batch_decode(outputs[:,batch['input_ids'].shape[1]-1:].detach().cpu().numpy(), skip_special_tokens=True)[0]
        #             eval_preds.append(output)

        #     eval_epoch_loss = eval_loss / len(eval_dataloader)
        #     eval_ppl = torch.exp(eval_epoch_loss)
        #     train_epoch_loss = total_loss / len(train_dataloader)
        #     train_ppl = torch.exp(train_epoch_loss)
        #     print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        gradient_accumulation_steps = batch_size
        output_dir = "./model_checkpoints"
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=1,
            lr_scheduler_type="linear",
            learning_rate=lr,
            optim="adamw_torch",
            num_train_epochs=num_epochs,
            # logging & evaluation strategies
            log_level="error",
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=500,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            bf16=True,
            bf16_full_eval=True,
        )

        # Create trainer instance
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )

        trainer.train()

        sanity_check = True
        if sanity_check:
            for i in range(len(valset)):
                inputs = tokenizer(val[0]['prompt'], return_tensors='pt')
                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
                    )
                    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                    print(output)

        signature.instructions = ""
        dsp.settings.show_guidelines = False

        lm = HFModel(model_name, is_client=True)
        lm.model = model
        lm.tokenizer = tokenizer
        lm.drop_prompt_from_output = True
        lm.kwargs['temperature'] = 0.0
        lm.is_client = False

        student.predictors()[0].lm = lm
        return student