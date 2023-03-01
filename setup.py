# TODO: Insert weights download link
import os
from configs import config
from utils import ValidatedInput
from urllib.request import urlretrieve

import zipfile
import functools


# Decorator for verbosing download
def logger(action):
    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            filename = args[0]
            if action == 'download':
                print(f'Downloading {filename}')
            elif action == 'extract':
                print(f'Extracting {filename}')
            func_val = func(*args, **kwargs)
            if action == 'download':
                print(f'Download complete')
            elif action == 'extract':
                print(f'Extraction complete')
            return func_val

        return wrapper

    return decorator


@logger('download')
def download(filename, url):
    urlretrieve(url, filename)


@logger('extract')
def extract(filename):
    file_dir = os.path.join(filename, os.path.pardir)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(file_dir)
    os.remove(filename)


if __name__ == '__main__':
    # Create directories
    dataset_dir = config.dataset_dir
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.images_dir, exist_ok=True)
    os.makedirs(config.weights_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)

    # Download dataset if requested
    # Prompt user for re-download if the dataset is already downloaded.
    ans_validator = lambda x: x in ['y', 'n']
    download_data = ValidatedInput('Download training dataset? (y/n)', ans_validator)
    if download_data == 'y' and os.path.exists(dataset_dir):
        download_data = ValidatedInput('The training dataset already exists. Re-download dataset? (y/n)', ans_validator)

    # Download and extract dataset
    if download_data == 'y':
        dataset_url = 'https://drive.google.com/uc?export=download&id=10Aqv57jU1RPsf2duOPHJe2asc6HYHOGc'
        dataset_zip_filename = os.path.join(config.data_dir, 'science_2022.zip')
        download(dataset_zip_filename, dataset_url)

        # Extract zip
        extract(dataset_zip_filename)

    # Download weights if requested
    # Prompt user for re-download if the weights are already downloaded.
    download_data = ValidatedInput('Download network weights? (y/n)', ans_validator)
    weights_filepath = os.path.join(config.weights_dir, 'gen_dis_default.pth')
    if download_data == 'y' and os.path.exists(weights_filepath):
        download_data = ValidatedInput('The weights file already exists. Re-download weights? (y/n)', ans_validator)

    # Download and extract dataset
    if download_data == 'y':
        weights_url = 'https://drive.google.com/uc?export=download&id=12oYbsfjyvYR_MosDfzhPjuEwZcF5LrF3'

        weights_zip_filename = os.path.join(config.weights_dir, 'weights_default.zip')
        download(weights_zip_filename, weights_url)

        # Extract zip
        extract(weights_zip_filename)

    print('Setup complete')
