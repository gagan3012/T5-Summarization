from .model import Summarization
from data.make_dataset import make_dataset

def train_model():
    """
    Train the model
    """
    # Load the data
    train_df = make_dataset(split = 'train')
    eval_df = make_dataset(split = 'test')

    model = Summarization()
    model.from_pretrained('t5-base')
    model.train(train_df=train_df, eval_df=eval_df, batch_size=4, max_epochs=3, use_gpu=True)
    model.save_model()