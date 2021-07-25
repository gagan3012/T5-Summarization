import pandas as pd


def process_data(split='train'):

    df.columns = ['Unnamed: 0', 'input_text', 'output_text']
    print(df.columns)
    df.to_csv('C:/Users/gbhat/Documents/GitHub/summarization/data/processed/{}.csv'.format(split))


if __name__ == '__main__':
    process_data(split='train')
    process_data(split='test')
    process_data(split='validation')
