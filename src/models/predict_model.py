import yaml

from .model import Summarization


def predict_model(text: str):
    """
    Predict the summary of the given text.
    """
    with open("model_params.yml") as f:
        params = yaml.safe_load(f)

    model = Summarization()
    model.load_model(model_type=params["model_type"], model_dir=params["model_dir"])
    pre_summary = model.predict(text)
    return pre_summary
