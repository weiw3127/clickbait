from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


def train_model(dataset, output_dir="models/bart-clickbait", epochs=3, batch_size=8):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

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
        disable_tqdm=False,          # <-- show live loss from tqdm

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
    model.save_pretrained(output_dir)
