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

    model = Summarization()
    model.load_model(model_type=params['model_type'], model_dir=params['model_dir'])
    results = model.evaluate(test_df=test_df, metrics=params['metric'])

    with dagshub.dagshub_logger() as logger:
        logger.log_metrics(results)
    return results
