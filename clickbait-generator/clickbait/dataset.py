from typing import Optional
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


class ClickbaitDataset:
    """
    Prepares article/headline pairs for seq2seq fine-tuning.
    - Input  : df['article'] (optionally prefixed with instruction)
    - Target : df['headline']
    """

    def __init__(
        self,
        tokenizer_name: str = "facebook/bart-base",
        source_max_len: int = 512,
        target_max_len: int = 64,
        use_prefix: bool = True,
        prefix: str = "generate clickbait headline: ",
        pad_to_max_length: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.use_prefix = use_prefix
        self.prefix = prefix
        self.pad_to_max_length = pad_to_max_length

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -------- Loading -------- #

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "article" not in df.columns or "headline" not in df.columns:
            raise ValueError("CSV must contain 'article' and 'headline' columns.")
        df["article"] = df["article"].astype(str).str.strip()
        df["headline"] = df["headline"].astype(str).str.strip()
        df = df[(df["article"] != "") & (df["headline"] != "")]
        df = df.dropna(subset=["article", "headline"]).reset_index(drop=True)
        return df

    # -------- Tokenization -------- #

    def _tokenize_batch(self, batch):
        sources = batch["article"]
        targets = batch["headline"]
        if self.use_prefix and self.prefix:
            sources = [self.prefix + s for s in sources]

        src = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            padding="max_length" if self.pad_to_max_length else False,
        )

        with self.tokenizer.as_target_tokenizer():
            tgt = self.tokenizer(
                targets,
                max_length=self.target_max_len,
                truncation=True,
                padding="max_length" if self.pad_to_max_length else False,
            )

        labels = tgt["input_ids"]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        labels = [[(tok if tok != pad_id else -100) for tok in seq] for seq in labels]

        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels": labels,
        }

    def tokenize(self, df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["article", "headline"]], preserve_index=False)
        ds = ds.map(self._tokenize_batch, batched=True, remove_columns=ds.column_names)
        return ds

    def build_from_csv(
        self, path: str, val_size: float = 0.1, seed: int = 42
    ) -> DatasetDict:
        df = self.load_csv(path)
        ds = Dataset.from_pandas(df[["article", "headline"]], preserve_index=False)
        split = ds.train_test_split(test_size=val_size, seed=seed)
        train_ds = split["train"].map(
            self._tokenize_batch, batched=True, remove_columns=split["train"].column_names
        )
        val_ds = split["test"].map(
            self._tokenize_batch, batched=True, remove_columns=split["test"].column_names
        )
        return DatasetDict(train=train_ds, validation=val_ds)
