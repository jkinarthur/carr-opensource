#!/usr/bin/env python3
"""
Prepare real benchmark datasets (ML-1M, Beauty, Toys, Yelp, Steam).

Downloads and preprocesses datasets into TSV format for RealInteractionDataset.
Output: data/datasets/{dataset_name}/interactions.tsv with columns: user_id, item_id, timestamp
"""

import os
import gzip
import urllib.request
from pathlib import Path
import csv


DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: Path, description: str = ""):
    """Download file with progress."""
    print(f"Downloading {description}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        return False


def prepare_movielens_1m():
    """Download and prepare MovieLens 1M dataset."""
    dataset_dir = DATA_DIR / "ML-1M"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Preparing MovieLens 1M ===")
    
    # Download
    zip_path = dataset_dir / "ml-1m.zip"
    if not zip_path.exists():
        url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
        if not download_file(url, zip_path, "MovieLens 1M"):
            return False
        
        # Extract
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dataset_dir)
        print(f"✓ Extracted to {dataset_dir}")
    
    # Convert ratings.dat to TSV with columns: user_id, item_id, timestamp
    ratings_file = dataset_dir / "ml-1m" / "ratings.dat"
    output_file = dataset_dir / "interactions.tsv"
    
    if not output_file.exists() and ratings_file.exists():
        print(f"Converting {ratings_file} to TSV...")
        with open(output_file, 'w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=['user_id', 'item_id', 'timestamp'], delimiter='\t')
            writer.writeheader()
            with open(ratings_file) as inf:
                for line in inf:
                    user, item, rating, timestamp = line.strip().split('::')
                    writer.writerow({'user_id': user, 'item_id': item, 'timestamp': timestamp})
        print(f"✓ Created {output_file}")
    
    return output_file.exists()


def prepare_amazon_datasets():
    """Download and prepare Amazon review datasets (Beauty, Toys, etc.)."""
    # Note: Amazon datasets require manual download or special access.
    # For now, we'll use pre-downloaded versions if available or create synthetic versions.
    print("\n=== Amazon Datasets (Beauty, Toys) ===")
    print("Note: Amazon datasets require manual download from:")
    print("  https://nijianmo.github.io/amazon/index.html")
    print("Format expected: interactions.jsonl.gz in each dataset folder")
    print("Will check for pre-downloaded versions...")
    
    # Placeholder for manual or pre-downloaded datasets
    for dataset_name in ["Beauty", "Toys"]:
        dataset_dir = DATA_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {dataset_name}: {dataset_dir}")
    
    return True


def prepare_yelp():
    """Download and prepare Yelp reviews dataset."""
    print("\n=== Yelp Dataset ===")
    print("Note: Yelp dataset requires registration at:")
    print("  https://www.yelp.com/dataset")
    dataset_dir = DATA_DIR / "Yelp"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Place downloaded review.json.gz in: {dataset_dir}")
    return True


def prepare_steam():
    """Download and prepare Steam reviews dataset."""
    print("\n=== Steam Dataset ===")
    print("Note: Steam dataset available from:")
    print("  https://github.com/nitlang/game_reviews_dataset")
    dataset_dir = DATA_DIR / "Steam"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Place downloaded dataset in: {dataset_dir}")
    return True


def list_available_datasets():
    """List available datasets."""
    print("\n=== Available Datasets ===")
    for ds_dir in sorted(DATA_DIR.glob("*")):
        if ds_dir.is_dir():
            tsv_file = ds_dir / "interactions.tsv"
            if tsv_file.exists():
                size_mb = tsv_file.stat().st_size / (1024**2)
                print(f"✓ {ds_dir.name}: {size_mb:.1f} MB")
            else:
                print(f"✗ {ds_dir.name}: interactions.tsv not found")


if __name__ == "__main__":
    print("=" * 60)
    print("Real Dataset Preparation")
    print("=" * 60)
    
    prepare_movielens_1m()
    prepare_amazon_datasets()
    prepare_yelp()
    prepare_steam()
    
    list_available_datasets()
    
    print("\n" + "=" * 60)
    print("To complete dataset preparation:")
    print("  1. ML-1M: Auto-downloaded above")
    print("  2. Beauty, Toys: Download from https://nijianmo.github.io/amazon/index.html")
    print("  3. Yelp: Download from https://www.yelp.com/dataset")
    print("  4. Steam: Download from GitHub")
    print("=" * 60)
