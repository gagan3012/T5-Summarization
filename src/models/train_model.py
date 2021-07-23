from src.models.model import Summarization
import pandas as pd


def train_model():
    """
    Train the model
    """
    # Load the data
    train_df = pd.read_csv('../../data/processed/train.csv')
    eval_df = pd.read_csv('../../data/processed/validation.csv')

    model = Summarization()
    model.from_pretrained('t5','t5-base')
    model.train(train_df=train_df, eval_df=eval_df, batch_size=4, max_epochs=3, use_gpu=True)
    model.save_model()


if __name__ == '__main__':
    train_model()
