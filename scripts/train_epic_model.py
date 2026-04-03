import os
import argparse
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from datasets import load_dataset, concatenate_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Train SanskritGPT-Epic (Hyper-Precision)")
    parser.add_argument("--epochs", type=int, default=30, help="Saturation epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=3e-5, help="Hyper-precision learning rate")
    parser.add_argument("--base_path", type=str, default="./", help="Base path for data and outputs")
    return parser.parse_args()

class DetailedLogCallback(TrainerCallback):
    def __init__(self, tokenizer, log_file, device):
        self.tokenizer = tokenizer
        self.log_file = log_file
        self.device = device

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs and "loss" in logs:
            kwargs['model'].eval()
            prompts = ["<MBH> ", "<RAM> "]
            samples = []
            for p in prompts:
                inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = kwargs['model'].generate(
                        inputs.input_ids, max_length=50, do_sample=True, temperature=0.8, top_p=0.95,
                        eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id
                    )
                samples.append(self.tokenizer.decode(outputs[0], skip_special_tokens=False))
            kwargs['model'].train()
            
            log_entry = f"Step {state.global_step} | Loss: {logs['loss']:.4f} | LR: {logs.get('learning_rate', 0):.2e}\n"
            log_entry += f"MBH: {samples[0]}\nRAM: {samples[1]}\n{'='*40}\n"
            print(log_entry)
            with open(self.log_file, "a", encoding="utf-8") as f: f.write(log_entry)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_file = os.path.join(args.base_path, "data/processed/sanskrit_epic_dataset.txt")
    output_dir = os.path.join(args.base_path, "model_output")
    tokenizer_dir = os.path.join(args.base_path, "tokenizer")
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    log_file = os.path.join(args.base_path, "training_log.txt")

    if not os.path.exists(data_file): raise FileNotFoundError(f"Dataset not found at {data_file}")

    if not os.path.exists(tokenizer_file):
        tokenizer_obj = Tokenizer(models.Unigram())
        tokenizer_obj.normalizer = normalizers.Sequence([NFKC()])
        tokenizer_obj.pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask" + ">", "<MBH>", "<RAM>", "<eos>"]
        trainer = trainers.UnigramTrainer(vocab_size=32000, special_tokens=special_tokens, unk_token="<unk>")
        tokenizer_obj.train(files=[data_file], trainer=trainer)
        tokenizer_obj.save(tokenizer_file)
    else:
        tokenizer_obj = Tokenizer.from_file(tokenizer_file)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj, bos_token="<s>", eos_token="<eos>", 
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask" + ">"
    )

    dataset = load_dataset("text", data_files={"train": data_file})
    # Context window upgraded to 512
    def tokenize_function(examples): return tokenizer(examples["text"], truncation=True, max_length=512)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)

    # 2x Data Augmentation
    train_original = split_dataset["train"].shuffle(seed=42)
    split_dataset["train"] = concatenate_datasets([train_original, train_original]).shuffle(seed=42)
    print(f"Augmented Training set size: {len(split_dataset['train'])}")

    # 512/8/8 Architecture with 512 Context
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=512, n_ctx=512, n_embd=512, n_layer=8, n_head=8, 
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
    )
    model = GPT2LMHeadModel(config)
    
    total_steps = (len(split_dataset["train"]) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        save_steps=10000, save_total_limit=1, evaluation_strategy="steps", eval_steps=10000, logging_steps=500,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup_steps, weight_decay=0.1, max_grad_norm=1.0, fp16=torch.cuda.is_available(),
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, report_to="none"
    )

    trainer = Trainer(
        model=model, args=training_args, 
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), 
        train_dataset=split_dataset["train"], eval_dataset=split_dataset["test"], 
        callbacks=[DetailedLogCallback(tokenizer, log_file, device), EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print(f"Starting Hyper-Precision training. Total steps: {total_steps}")
    trainer.train()

    final_output = os.path.join(args.base_path, "sanskrit-gpt-epic-hyper")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

if __name__ == "__main__": main()
