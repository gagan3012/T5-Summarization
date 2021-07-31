import pandas as pd
import yaml
import os


def process_data(split="train"):

    with open("params.yml") as f:
        params = yaml.safe_load(f)

    df = pd.read_csv("data/raw/{}.csv".format(split))
    if os.path.exists("data/raw/{}.csv".format(split)):
        os.remove("data/raw/{}.csv".format(split))
    df.to_csv('data/processed/{}.csv'.format(split))


if __name__ == '__main__':
    process_data(split='train')
    process_data(split='test')
    process_data(split='validation')
