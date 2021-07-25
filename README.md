summarization
==============================

T5 Summarisation Using Pytorch Lightning

Instructions
------------
1. Clone the repo.
1. Edit the `params.yml` to change the parameters to train the model.
1. Run `make dirs` to create the missing parts of the directory structure described below. 
1. *Optional:* Run `make virtualenv` to create a python virtual environment. Skip if using conda or some other env manager.
    1. Run `source env/bin/activate` to activate the virtualenv. 
1. Run `make requirements` to install required python packages.
1. Process your data, train and evaluate your model using `make run`
1. When you're happy with the result, commit files (including .dvc files) to git.
 
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make dirs` or `make clean`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── process_data.dvc   <- Process the raw data and prepare it for training.
    ├── raw_data.dvc       <- Keeps the raw data versioned.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── metrics.txt    <- Relevant metrics after evaluating the model.
    │   └── training_metrics.txt    <- Relevant metrics from training the model.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
    └── train.dvc          <- Traing a model on the processed data.


--------
