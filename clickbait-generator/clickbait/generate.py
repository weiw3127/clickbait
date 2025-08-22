from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch


class TitleGenerator:
    def __init__(self, model_dir: str = "data/clickbait_model", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def generate(
        self,
        article: str,
        max_length: int = 24,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        prefix: str = "generate clickbait headline: ",
    ) -> str:
        input_text = prefix + article
        enc = self.tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.generate(
            **enc,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=True,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def batch_generate(
        self,
        articles: List[str],
        max_length: int = 24,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        prefix: str = "generate clickbait headline: ",
    ) -> List[str]:
        inputs = [prefix + a for a in articles]
        enc = self.tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.generate(
            **enc,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)
