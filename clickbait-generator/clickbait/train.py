import os
from dataclasses import dataclass
from typing import Optional

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


@dataclass
class TrainConfig:
    model_name: str = "facebook/bart-base"
    output_dir: str = "models/bart-clickbait"
    epochs: float = 3
    train_bs: int = 8
    eval_bs: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    fp16: bool = True  # set False if CPU-only
    gradient_accumulation_steps: int = 1


def train_model(
    dataset: DatasetDict,
    cfg: TrainConfig = TrainConfig(),
    tokenizer: Optional[AutoTokenizer] = None,
):
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./clickbait_model",
        do_train=True,
        do_eval=True,

        # core
        learning_rate=4e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        predict_with_generate=True,
        save_total_limit=2,

        # logging (force first-step log and visible tqdm)
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False,        

        # evaluation (your version uses eval_steps directly)
        eval_steps=500,

        # safeguards that also fix several zero-loss logging cases
        remove_unused_columns=False,
        label_names=["labels"],

        # MPS nicety
        dataloader_pin_memory=False,

        # misc
        push_to_hub=False,
        seed=42,
    )


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
