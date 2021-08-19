from dagshub import dagshub_logger
import yaml

from model import Summarization
import pandas as pd


def evaluate_model():
    """
    Evaluate model using rouge measure
    """
    with open("model_params.yml") as f:
        params = yaml.safe_load(f)

    test_df = pd.read_csv("data/processed/test.csv")
    test_df = test_df.sample(n=25, random_state=42)
    model = Summarization()
    model.load_model(model_type=params["model_type"], model_dir=params["model_dir"])
    results = model.evaluate(test_df=test_df, metrics=params["metric"])

    with dagshub_logger(
        metrics_path="reports/evaluation_metrics.csv", should_log_hparams=False
    ) as logger:
        logger.log_metrics(results)


if __name__ == "__main__":
    evaluate_model()
