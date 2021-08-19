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
    model = Summarization()
    model.load_model(model_type=params["model_type"], model_dir=params["model_dir"])
    results = model.evaluate(test_df=test_df, metrics=params["metric"])

    with open("reports/metrics.csv", "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    evaluate_model()
