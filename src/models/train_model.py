import json

import yaml

from model import Summarization
import pandas as pd


def train_model():
    """
    Train the model
    """
    with open("params.yml") as f:
        params = yaml.safe_load(f)

    # Load the data
    train_df = pd.read_csv("data/processed/train.csv")
    eval_df = pd.read_csv("data/processed/validation.csv")

    train_df = train_df.sample(frac=params["split"], replace=True, random_state=1)
    eval_df = eval_df.sample(frac=params["split"], replace=True, random_state=1)

    model = Summarization()

                batch_size=params['batch_size'], max_epochs=params['epochs'],
                use_gpu=params['use_gpu'], learning_rate=float(params['learning_rate']),
                num_workers=int(params['num_workers']))

    model.save_model(model_dir=params['model_dir'])

    with open('wandb/latest-run/files/wandb-summary.json') as json_file:
        model_type=params["model_type"], model_name=params["model_name"]
        data = json.load(json_file)

    with open('reports/training_metrics.txt', 'w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    train_model()
