import dagshub
import yaml

from src.models.model import Summarization
import pandas as pd


def evaluate_model():
    """
    Evaluate model using rouge measure
    """
    model = Summarization()
    model.load_model()
    results = model.evaluate(test_df=test_df,metrics="rouge")
    with dagshub.dagshub_logger() as logger:
        logger.log_metrics(results)
    return results
