summarization
==============================

T5 Summarisation Using Pytorch Lightning

Instructions
------------
1. Clone the repo.
1. Run `make dirs` to create the missing parts of the directory structure described below. 
1. *Optional:* Run `make virtualenv` to create a python virtual environment. Skip if using conda or some other env manager.
    1. Run `source env/bin/activate` to activate the virtualenv. 
1. Run `make requirements` to install required python packages.
1. Put the raw data in `data/raw`.
1. To save the raw data to the DVC cache, run `dvc commit raw_data.dvc`
1. Edit the code files to your heart's desire.
1. Process your data, train and evaluate your model using `dvc repro eval.dvc` or `make reproduce`
1. When you're happy with the result, commit files (including .dvc files) to git.
 
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make dirs` or `make clean`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
