from src.data.make_dataset import make_dataset
from .model import Summarization

def predict_model(text):
    """
    Predict the summary of the given text.
    """
    model = Summarization()
    model.load_model()
    pre_summary = model.predict(text)
    return pre_summary
