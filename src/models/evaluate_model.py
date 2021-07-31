import dagshub
import yaml

from model import Summarization
import pandas as pd
import json


def evaluate_model():
    """
    Evaluate model using rouge measure
    """
    with open("params.yml") as f:
        params = yaml.safe_load(f)

    test_df = pd.read_csv("data/processed/test.csv")[:25]
    model = Summarization()
    results = model.evaluate(test_df=test_df, metrics=params['metric'])

    with open('reports/metrics.txt', 'w') as fp:
        json.dump(results, fp)


if __name__ == '__main__':
    evaluate_model()
