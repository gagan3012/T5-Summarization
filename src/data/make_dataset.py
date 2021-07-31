import yaml
from datasets import load_dataset
import pandas as pd
import os
import pprint


def make_dataset(dataset="cnn_dailymail", split="train"):
    """make dataset for summarisation"""
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
    dataset = load_dataset(dataset, "3.0.0", split=split)
    df = pd.DataFrame()
    df["article"] = dataset["article"]
    df["highlights"] = dataset["highlights"]
    df.to_csv("data/raw/{}.csv".format(split))


if __name__ == "__main__":
    with open("params.yml") as f:
        params = yaml.safe_load(f)
    pprint.pprint(params)
    make_dataset(dataset=params["data"], split="train")
    make_dataset(dataset=params["data"], split="test")
    make_dataset(dataset=params["data"], split="validation")
