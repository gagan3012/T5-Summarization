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
    model_url = HfApi().create_repo(token=hf_token, name=model_name, exist_ok=True)
    model_repo = Repository(
        "./hf_model",
        clone_from=model_url,
        use_auth_token=hf_token,
        git_email=f"{hf_username}@users.noreply.huggingface.co",
        git_user=hf_username,
    )

    del hf_token
    try:
        readme_txt = open(join(dirname(__file__), "README.md"), encoding="utf8").read()
    except Exception:
        readme_txt = None

    (Path(model_repo.local_dir) / "README.md").write_text(readme_txt)
    model_to_upload.save_model(Path(model_repo.local_dir))
    commit_url = model_repo.push_to_hub()
