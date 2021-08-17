# T5S

Natural Language Processing is one of the key areas where Machine Learning has been very effective. In fact, whereas NLP traditionally required a lot of human intervention, today, this is no longer true. Specifically, Deep Learning technology can be used for learning tasks related to language, such as translation, classification, entity recognition or in this case, summarization.

Summarization is the task of condensing a piece of text to a shorter version, reducing the size of the initial text while at the same time preserving key informational elements and the meaning of content. Since manual text summarization is a time expensive and generally laborious task, the automatization of the task is gaining increasing popularity and therefore constitutes a strong motivation for academic research.

There are important applications for text summarization in various NLP related tasks such as text classification, question answering, legal texts summarization, news summarization, and headline generation. Moreover, the generation of summaries can be integrated into these systems as an intermediate stage which helps to reduce the length of the document.

Our goal here was to create a reproducible pipeline for Text summarisation. We wanted to train the model, visualise the results and upload the model. This project is built using the DVC cookiecutter template provided by DAGsHub. 

The package for text summarization is available to be downloaded as package 

```
pip install t5s
```

## Pipeline

Once we download the package we can use the training pipeline. Before we describe how the package works we will explain what each stage of the pipeline is.

![image](https://user-images.githubusercontent.com/49101362/129772732-438e700b-b0f0-4a74-832e-27628d8c2da3.png)

The first stage of our pipeline is to download data from the hugging face hub. Here for training we have used the CNN_dailymail dataset. In order to download the dataset we use the parameter files that is data_params.yml which defines the datasets and the split that we would like to train our data on. We run the download_data stage which downloads the data and then stores it as raw data which we will then process. 

Once the raw data is saved we move on to processing the data using our script to process the raw data. We change the column names and modify the data to work with our training script. Now the data is also split into three different files: train.csv, validation.csv and test.csv. 

Now we can move on to training the model. The code for training the model has been written in pytorch lightning. The script allows us to train T5, mT5 and byT5 models as well. All the script parameters can be controlled using the model_params.yml file. The training stage returns the model that can be saved and also the training metrics which are logged using MLflow and DAGsHub. 

Next we need to evaluate the model that has been created and to do so we need to use the rouge metric which uses the test datasets to evaluate the model. The evaluation metrics are also saved using DAGsHub. Once we commit all the models to git we can evaluate our models from the DAGsHub repo. 

![image](https://user-images.githubusercontent.com/49101362/129772801-063ec2fd-feb2-401b-ab9c-0d9250447d1a.png)


We can also visualise and test the results of the model using a streamlit app which can be accessed using Hugging Face spaces. We also have the option of running the upload script and uploading the model to Hugging Face Hub too.

![image](https://user-images.githubusercontent.com/49101362/129772845-8a93b3ce-ad6b-44ce-aa41-0b6da65a8ac4.png)

## T5S CLI

In order to run the pipeline we have setup a CLI application that will help us run the pipeline 

To install the pipeline we need to first install t5s as 

```
pip install t5s
```

Firstly we need to clone the repo containing the code so we can do that using before cloning make sure you have forked the code from the main repo so it would be faster to push and pull
```
t5s clone [-h] [-u USERNAME]
```

We would then have to create the required directories to run the pipeline

```
t5s dirs
``` 

Now to define the parameters for the run we have to run:
```
t5s start [-h] [-d DATASET] [-s SPLIT] [-n NAME] [-mt MODEL_TYPE]
                 [-m MODEL_NAME] [-e EPOCHS] [-lr LEARNING_RATE]
                 [-b BATCH_SIZE]
```
Then we need to pull the models from DVC

```
t5s pull
```

Now to run the training pipeline we can run:

```
t5s run
```

Before pushing make sure that the DVC remote is setup correctly:

```

dvc remote modify origin url https://dagshub.com/{user_name}/summarization.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user {user_name}
dvc remote modify origin --local password {your_token}

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



