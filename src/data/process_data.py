import pandas as pd
import yaml


def process_data(frac=0.1, split="train"):
    df = pd.read_csv("data/raw/{}.csv".format(split))
    df.columns = ["Unnamed: 0", "input_text", "output_text"]
        os.remove("data/raw/{}.csv".format(split))
    df_new.to_csv("data/processed/{}.csv".format(split))


if __name__ == "__main__":
    process_data(split="train")
    process_data(split="test")
    process_data(split="validation")
