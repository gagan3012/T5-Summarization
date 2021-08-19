import pandas as pd
import yaml


    df = pd.read_csv("data/raw/{}.csv".format(split))
    df.columns = ["Unnamed: 0", "input_text", "output_text"]
    df = df.sample(frac=params["split"], replace=True, random_state=1)
    if os.path.exists("data/raw/{}.csv".format(split)):
        os.remove("data/raw/{}.csv".format(split))
    df.to_csv("data/processed/{}.csv".format(split))


if __name__ == "__main__":
    process_data(split="train")
    process_data(split="test")
    process_data(split="validation")
