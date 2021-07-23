from datasets import load_dataset
import pandas as pd


def make_dataset(dataset='cnn_dailymail', split='train'):
    """make dataset for summarisation"""
    dataset = load_dataset(dataset, '3.0.0', split=split)
    df = pd.DataFrame()
    df['output_text'] = dataset['target']
    return df
    df['article'] = dataset['article']

if __name__ == '__main__':
    make_dataset(dataset='cnn_dailymail', split='train', version="3.0.0")