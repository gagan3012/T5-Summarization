import dagshub

from src.models.model import Summarization
import pandas as pd

def evaluate_model():
    """
    Evaluate model using rouge measure
    """
    test_df =  pd.load_csv('../../data/processed/test.csv')
