import os
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download


def load_parquet_data():
    local_path = "train-00000-of-00015.parquet"
    file_is_valid = False

    if os.path.exists(local_path):
        try:
            pd.read_parquet(local_path, columns=["article"], nrows=1)
            file_is_valid = True
        except Exception:
            try:
                os.remove(local_path)
            except OSError:
                pass

    if not file_is_valid:
        try:
            cached_path = hf_hub_download(
                repo_id="ccdv/arxiv-summarization",
                filename="document/train-00000-of-00015.parquet",
                repo_type="dataset"
            )
            shutil.copy(cached_path, local_path)
        except Exception as e:
            print(f"Failed to download the dataset: {e}")
            raise

    df = pd.read_parquet(local_path, columns=["article", "abstract"])
    df = df.rename(columns={"article": "document"})
    return df
