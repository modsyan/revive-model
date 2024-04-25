import os


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory '{path}' does not exist.")
