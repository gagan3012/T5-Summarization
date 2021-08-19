import shutil
from getpass import getpass
from os.path import join, dirname
from pathlib import Path
import yaml

from model import Summarization
from huggingface_hub import HfApi, Repository


def upload(model_to_upload, model_name):
    hf_username = input("Enter your HuggingFace username:")
    hf_token = getpass("Enter your HuggingFace token:")
