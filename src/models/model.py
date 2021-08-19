import torch
import pandas as pd
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dagshub.pytorch_lightning import DAGsHubLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from datasets import load_metric
from tqdm.auto import tqdm

# from dagshub.pytorch_lightning import DAGsHubLogger

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
        labels[labels == 0] = -100

        return dict(
            keywords=data_row["input_text"],
            text=data_row["output_text"],
            keywords_input_ids=input_encoding["input_ids"].flatten(),
            keywords_attention_mask=input_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=output_encoding["attention_mask"].flatten(),
        )


class PLDataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 4,
        split: float = 0.1,
        num_workers: int = 2,
    ):
        """
        :param data_df:
        :param tokenizer:
        :param source_max_token_len:
        :param target_max_token_len:
        :param batch_size:
        :param split:
        """
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.split = split
        self.batch_size = batch_size
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = DataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = DataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """validation dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LightningModel(LightningModule):
    """PyTorch Lightning Model class"""

    def __init__(
        self,
        tokenizer,
        model,
        learning_rate,
        adam_epsilon,
        weight_decay,
        output: str = "outputs",
    ):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5 tokenizer
            model : T5 model
            output (str, optional): output directory to save model checkpoints. Defaults to "outputs".
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.output = output
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """forward step"""
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """training step"""
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_size):
        """validation step"""
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
        """test step"""
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """configure optimizers"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon
        )
        self.opt = optimizer
        return [optimizer]


class Summarization:
    """Custom Summarization class"""

    def __init__(self) -> None:
        """initiates Summarization class"""
        pass

    def from_pretrained(self, model_type="t5", model_name="t5-base") -> None:
        """
        loads T5/MT5 Model model for training/finetuning
        Args:
            model_name (str, optional): exact model architecture name, "t5-base" or "t5-large". Defaults to "t5-base".
            :param model_type:
        """
        if model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        max_epochs: int = 5,
        use_gpu: bool = True,
        outputdir: str = "models",
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        learning_rate: float = 0.0001,
        adam_epsilon: float = 0.01,
        num_workers: int = 2,
        weight_decay: float = 0.0001,
    ):
        """
        trains T5/MT5 model on custom dataset
        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "input_text" and "output_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "input_text" and
            "output_text"
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training,
            if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping.
            Defaults to 0 (disabled)
            :param learning_rate:
            :param adam_epsilon:
        """
        self.target_max_token_len = target_max_token_len
        self.data_module = PLDataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=num_workers,
        )

        self.T5Model = LightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            output=outputdir,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
        )

        MLlogger = MLFlowLogger(
            experiment_name="Summarization",
            tracking_uri="https://dagshub.com/gagan3012/summarization.mlflow",
        )

        WandLogger = WandbLogger(project="summarization-dagshub")

        # logger = DAGsHubLogger(metrics_path='reports/training_metrics.txt')

        early_stop_callback = (
            [
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00,
                    patience=early_stopping_patience_epochs,
                    verbose=True,
                    mode="min",
                )
            ]
            if early_stopping_patience_epochs > 0
            else None
        )

        gpus = -1 if use_gpu and torch.cuda.is_available() else 0

        trainer = Trainer(
            logger=[WandLogger, MLlogger],
            callbacks=early_stop_callback,
            max_epochs=max_epochs,
            gpus=gpus,
            progress_bar_refresh_rate=5,
        )

        trainer.fit(self.T5Model, self.data_module)

    def load_model(
        self, model_type: str = "t5", model_dir: str = "models", use_gpu: bool = False
    ):
        """
        loads a checkpoint for inferencing/prediction
        Args:
            model_type (str, optional): "t5" or "mt5". Defaults to "t5".
            model_dir (str, optional): path to model directory. Defaults to "outputs".
            use_gpu (bool, optional): if True, model uses gpu for inferencing/prediction. Defaults to True.
        """
        if model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_dir}", return_dict=True
            )
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_dir}")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"{model_dir}", return_dict=True
            )
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_dir}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_dir}", return_dict=True
            )

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise Exception(
                    "exception ---> no gpu found. set use_gpu=False, to use CPU"
                )
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def save_model(self, model_dir="models"):
        """
        Save model to dir
        :param model_dir:
        :return: model is saved
        """
        path = f"{model_dir}"
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )

        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return preds

    def evaluate(self, test_df: pd.DataFrame, metrics: str = "rouge"):
        metric = load_metric(metrics)
        input_text = test_df["input_text"]
        references = test_df["output_text"]
        references = references.to_list()

        predictions = [self.predict(x) for x in tqdm(input_text)]

        results = metric.compute(predictions=predictions, references=references)

        output = {
            "Rouge_1 Low Precision": results["rouge1"].low.precision,
            "Rouge_1 Low recall": results["rouge1"].low.recall,
            "Rouge_1 Low F1": results["rouge1"].low.fmeasure,
            "Rouge_1 Mid Precision": results["rouge1"].mid.precision,
            "Rouge_1 Mid recall": results["rouge1"].mid.recall,
            "Rouge_1 Mid F1": results["rouge1"].mid.fmeasure,
            "Rouge_1 High Precision": results["rouge1"].high.precision,
            "Rouge_1 High recall": results["rouge1"].high.recall,
            "Rouge_1 High F1": results["rouge1"].high.fmeasure,
            "Rouge_2 Low Precision": results["rouge2"].low.precision,
            "Rouge_2 Low recall": results["rouge2"].low.recall,
            "Rouge_2 Low F1": results["rouge2"].low.fmeasure,
            "Rouge_2 Mid Precision": results["rouge2"].mid.precision,
            "Rouge_2 Mid recall": results["rouge2"].mid.recall,
            "Rouge_2 Mid F1": results["rouge2"].mid.fmeasure,
            "Rouge_2 High Precision": results["rouge2"].high.precision,
            "Rouge_2 High recall": results["rouge2"].high.recall,
            "Rouge_2 High F1": results["rouge2"].high.fmeasure,
            "Rouge_L Low Precision": results["rougeL"].low.precision,
            "Rouge_L Low recall": results["rougeL"].low.recall,
            "Rouge_L Low F1": results["rougeL"].low.fmeasure,
            "Rouge_L Mid Precision": results["rougeL"].mid.precision,
            "Rouge_L Mid recall": results["rougeL"].mid.recall,
            "Rouge_L Mid F1": results["rougeL"].mid.fmeasure,
            "Rouge_L High Precision": results["rougeL"].high.precision,
            "Rouge_L High recall": results["rougeL"].high.recall,
            "Rouge_L High F1": results["rougeL"].high.fmeasure,
            "rougeLsum Low Precision": results["rougeLsum"].low.precision,
            "rougeLsum Low recall": results["rougeLsum"].low.recall,
            "rougeLsum Low F1": results["rougeLsum"].low.fmeasure,
            "rougeLsum Mid Precision": results["rougeLsum"].mid.precision,
            "rougeLsum Mid recall": results["rougeLsum"].mid.recall,
            "rougeLsum Mid F1": results["rougeLsum"].mid.fmeasure,
            "rougeLsum High Precision": results["rougeLsum"].high.precision,
            "rougeLsum High recall": results["rougeLsum"].high.recall,
            "rougeLsum High F1": results["rougeLsum"].high.fmeasure,
        }
        return output

    def upload(self, hf_username, model_name):
        hf_password = getpass("Enter your HuggingFace password")
        if Path("./models").exists():
            shutil.rmtree("./models")
        token = HfApi().login(username=hf_username, password=hf_password)
        del hf_password
        model_url = HfApi().create_repo(token=token, name=model_name, exist_ok=True)
        model_repo = Repository(
            "./model",
            clone_from=model_url,
            use_auth_token=token,
            git_email=f"{hf_username}@users.noreply.huggingface.co",
            git_user=hf_username,
        )

        readme_txt = f"""
            ---
            Summarisation model {model_name}
            """.strip()

        (Path(model_repo.local_dir) / "README.md").write_text(readme_txt)
        self.save_model()
        commit_url = model_repo.push_to_hub()

        print("Check out your model at:")
        print(commit_url)
        print(f"https://huggingface.co/{hf_username}/{model_name}")
