import os
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download

LABEL_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


def load_parquet_data():
    local_path = "train-00000-of-00001.parquet"
    file_is_valid = False

    if os.path.exists(local_path):
        try:
            pd.read_parquet(local_path, nrows=1)
            file_is_valid = True
        except Exception:
            try:
                os.remove(local_path)
            except OSError:
                pass

    if not file_is_valid:
        try:
            cached_path = hf_hub_download(
                repo_id="fancyzhx/ag_news",
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset"
            )
            shutil.copy(cached_path, local_path)
        except Exception as e:
            print(f"Failed to download the dataset: {e}")
            raise

    df = pd.read_parquet(local_path)
    df['group_index'] = df.groupby('label').cumcount()
    df = df.sort_values(by=['group_index', 'label']).reset_index(drop=True)
    df = df.drop(columns=['group_index'])
    df['label'] = df['label'].map(LABEL_MAPPING)
    return df


if __name__ == "__main__":
    df = load_parquet_data()
    print(df.head())
