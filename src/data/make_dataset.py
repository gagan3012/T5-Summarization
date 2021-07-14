from datasets import load_dataset
import pandas as pd


def make_dataset(dataset='cnn_dailymail', split='train', version="3.0.0"):
    """make dataset for summarisation"""
    dataset = load_dataset(dataset, split=split, script_version=version)
    df = pd.DataFrame()
    df['input_text'] = dataset['concepts']
    df['output_text'] = dataset['target']
    return df

