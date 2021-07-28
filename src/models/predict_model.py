import yaml

from model import Summarization
import pandas as pd


def predict_model(text):
    """
    Predict the summary of the given text.
    """
    with open("params.yml") as f:
        params = yaml.safe_load(f)


    model = Summarization()
    model.load_model(model_type=params['model_type'], model_dir=params['model_dir'])
    pre_summary = model.predict(text)
    return pre_summary


if __name__ == '__main__':
    text = pd.load_csv('data/processed/test.csv')['input_text'][0]
    pre_summary = predict_model(text)
    print(pre_summary)
