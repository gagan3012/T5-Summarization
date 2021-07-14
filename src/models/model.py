import time
import torch
import numpy as np
import pandas as pd
from datasets import load_metric
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule

torch.cuda.empty_cache()
pl.seed_everything(42)


class DataModule(Dataset):
    """
    Data Module for pytorch
    """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5Tokenizer,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
    ):
        """
        :param data:
        :param tokenizer:
        :param source_max_token_len:
        :param target_max_token_len:
        """
        self.data = data
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        input_encoding = self.tokenizer(
            data_row["input_text"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        output_encoding = self.tokenizer(
            data_row["output_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = output_encoding["input_ids"]
        labels[
            labels == 0
            ] = -100

        return dict(
            keywords=data_row["keywords"],
            text=data_row["text"],
            keywords_input_ids=input_encoding["input_ids"].flatten(),
            keywords_attention_mask=input_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=output_encoding["attention_mask"].flatten(),
        )


class PLDataModule(LightningDataModule):
