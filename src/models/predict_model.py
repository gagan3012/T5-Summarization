import yaml

from .model import Summarization
import pandas as pd

def predict_model(text):
    """
    Predict the summary of the given text.
    """
    model = Summarization()
    model.load_model()
    pre_summary = model.predict(text)
    return pre_summary

    
if __name__ == '__main__':
    text = make_dataset(split="test")['input_text'][0]
    pre_summary = predict_model(text)
    print(pre_summary)  
