"""
Docstring for get_pathogenkg

get_pathogenkg.py
    - contains functions to download preprocessed PathogenKG datasets from huggingface

    
https://huggingface.co/datasets/johndef64/PathogenKG/tree/main

Datasets available for download:

https://huggingface.co/datasets/johndef64/PathogenKG/resolve/main/pathogenkg_pathogens.zip?download=true

save to data/pathogenkg/
create in not exists

unpack zip files after download in the same directory


"""
import os
import requests
import zipfile
from pathlib import Path
from typing import Optional


DATASETS = {
    "drkg_extended": "https://huggingface.co/datasets/johndef64/PathogenKG/resolve/main/drkg_extended.zip?download=true",
    "pathogens": "https://huggingface.co/datasets/johndef64/PathogenKG/resolve/main/pathogenkg_pathogens.zip?download=true",    
}

SOURCE_DATA = {"source_datasets": "https://huggingface.co/datasets/johndef64/PathogenKG/resolve/main/source_datasets.zip?download=true"}

DATA_DIR = Path("dataset/pathogenkg")
SOURCE_DATA_DIR =  Path("dataset")


def download_file(url: str, output_path: Path) -> None:
    """Download a file from a URL to the specified path."""
    print(f"Downloading {output_path.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {output_path.name}")


def extract_zip(zip_path: Path) -> None:
    """Extract a zip file to the same directory."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent)
    print(f"Extracted {zip_path.name}")


def download_pathogenkg_dataset(dataset_name: str, extract: bool = True) -> Path:
    """
    Download a specific  dataset.
    
    Args:
        dataset_name: Name of the dataset ('pathogens')
        extract: Whether to extract the zip file after download
        
    Returns:
        Path to the downloaded (and extracted) file
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    # Create directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download file
    url = DATASETS[dataset_name]
    filename = f"{dataset_name}.zip"
    output_path = DATA_DIR / filename
    
    if not output_path.exists():
        download_file(url, output_path)
    else:
        print(f"{filename} already exists, skipping download")
    
    # Extract if requested
    if extract:
        extract_zip(output_path)
    
    return output_path


def download_all_datasets(extract: bool = True) -> None:
    """Download all available PathogenKG datasets."""
    for dataset_name in DATASETS.keys():
        download_pathogenkg_dataset(dataset_name, extract)

    for source_name, url in SOURCE_DATA.items():
        # simple download without extraction
        filename = f"{source_name}.zip"
        output_path = SOURCE_DATA_DIR / filename
        if not output_path.exists():
            download_file(url, output_path)
        else:
            print(f"{filename} already exists, skipping download")



if __name__ == "__main__":
    # Download all datasets
    download_all_datasets()