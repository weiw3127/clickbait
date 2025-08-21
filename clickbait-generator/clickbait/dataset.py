import pandas as pd
from dataset import dataset
from transformers import AutoTokenizer


class ClickbaitDataset: 
    """
    Prepares DailyMail article/headline pairs for seq2seq fine-tuning.
    - Source text:  df['article']
    - Target text:  df['headline']
    - Optional instruction prefix is prepended to source text.
    """

    # other good options: "google/pegasus-xsum", "t5-base"
    def __init__(
        self, 
        tokenizer_name="facebook/bart-base", 
        source_max_len=512,
        target_max_len=64, 
        use_prefix=True, 
        prefix="generate clickbait headline: ",
    ): 
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.use_prefix = use_prefix
        self.prefix = prefix
    
    def load_csv(self, path): 
        return pd.read_csv(path)

    def _tokenize_batch(self, batch): 
        '''
        Tokenize a batch (dict of list) from datasets.Dataset.map.
        Returns model-ready field: input_ids, attention_mask, labels
        '''
        articles = batch["article"]
        headlines = batch["headline"]

        if self.use_prefix and self.prefix: 
            articles = [self.prefix + a for a in articles]
        
        # articles encoding
        src_enc = self.tokenizer(
            articles, 
            max_length = self.source_max_len,
            truncation=True,
        )

        # target encodling
        tgt_enc = self.tokenizer(
            headlines, 
            max_length = self.target_max_len,
            truncation=True,
        )

        labels = tgt_enc["input_ids"]

        return {
            "input_ids": src_enc["input_ids"],
            "attention_mask": src_enc["attention_mask"],
            "labels": labels,
        }

    def tokenize(self, df: pd.DataFrame) -> Dataset: 
        '''
        Convert a pandas Dataframe to a tokenized dataset. 
        expects columns: 'article' headline. 
        '''
        ds = Dataset.from_pandas(df[["article", "headline"]], preserve_index=False)
        ds = ds.map(self._tokenize_batch, batched=True, remove_columns=ds.column_names)
        return ds 

    
