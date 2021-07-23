import pandas as pd


def process_data(split='train'):
    df = pd.read_csv('C:/Users/gbhat/Documents/GitHub/summarization/data/raw/{}.csv'.format(split))
    df.columns = ['Unnamed: 0', 'input_text', 'output_text']
    print(df.columns)
    df.to_csv('C:/Users/gbhat/Documents/GitHub/summarization/data/processed/{}.csv'.format(split))
