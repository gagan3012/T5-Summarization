---
title: t5s
emoji: 💯
colorFrom: yellow
colorTo: red
sdk: streamlit
app_file: src/visualization/visualize.py
pinned: false
---



<h1 align="center">t5s</h1>

[![pypi Version](https://img.shields.io/pypi/v/t5s.svg?logo=pypi&logoColor=white)](https://pypi.org/project/t5s/)
[![Downloads](https://static.pepy.tech/personalized-badge/t5s?period=total&units=none&left_color=grey&right_color=orange&left_text=Pip%20Downloads)](https://pepy.tech/project/t5s)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/gagan3012/summarization)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gagan3012/summarization/blob/master/notebooks/t5s.ipynb)
[![DAGSHub](https://img.shields.io/badge/%F0%9F%90%B6-Pipeline%20on%20DAGsHub-green)](https://dagshub.com/gagan3012/summarization)

T5 Summarisation Using Pytorch Lightning, DVC, DagsHub and HuggingFace Spaces

Here you will find the code for the project, but also the data, models, pipelines and experiments. This means that the project is easily reproducible on any machine, but also that you can contribute data, models, and code to it.

Have a great idea for how to improve the model? Want to add data and metrics to make it more explainable/fair? We'd love to get your help.


## Installation

To use and run the DVC pipeline install the `t5s` package

```
pip install t5s
```

## Usage

![carbon (7)](https://user-images.githubusercontent.com/49101362/129279588-17271a4c-7258-4208-a94d-89e5b97b6cd0.png)

Firstly we need to clone the repo containing the code so we can do that using:

```
t5s clone 
```

We would then have to create the required directories to run the pipeline

```
t5s dirs
``` 

Then we need to pull the models from DVC

```
t5s pull
```

Now to run the training pipeline we can run:

```
t5s run
```

Finally to push the model to DVC

```
t5s push
```

To push this model to HuggingFace Hub for inference you can run:

```
t5s upload
```

Next if we would like to test the model and visualise the results we can run:

```
t5s visualize
```
And this would create a streamlit app for testing

