from datasets import load_dataset
import pandas as pd


def make_dataset(dataset='cnn_dailymail', split='train'):
    """make dataset for summarisation"""
    dataset = load_dataset(dataset, '3.0.0', split=split)
    df = pd.DataFrame()
    df['article'] = dataset['article']
    df['highlights'] = dataset['highlights']
    df.to_csv('C:/Users/gbhat/Documents/GitHub/summarization/data/raw/{}.csv'.format(split))


if __name__ == '__main__':
