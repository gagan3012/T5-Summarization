import pandas as pd


def process_data(split='train'):
    df = pd.read_csv('C:/Users/gbhat/Documents/GitHub/summarization/data/raw/{}.csv'.format(split))
