import pandas as pd


def process_data(split='train'):

    df = pd.read_csv('data/raw/{}.csv'.format(split))
    df.columns = ['Unnamed: 0', 'input_text', 'output_text']
    df.to_csv('data/processed/{}.csv'.format(split))


if __name__ == '__main__':
    process_data(split='train')
    process_data(split='test')
    process_data(split='validation')
