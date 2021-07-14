from datasets import load_dataset
import pandas as pd


def make_dataset(dataset='cnn_dailymail', split='train', version="3.0.0"):
    """make dataset for summarisation"""
