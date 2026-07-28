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
arxiv_categories = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control"
}

data_path = "src/functionalities/theme_reconginition/data/cs_papers_api.csv"


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


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['title', 'abstract', 'primary_category'])
    df['text'] = df['title'] + " " + df['abstract']
    df['label'] = df['primary_category'].map(arxiv_categories)
    
    df = df.dropna(subset=['label'])
    
    return df[['text', 'label']].reset_index(drop=True)

def load_data_raw(path, nrows=1000):
    return pd.read_csv(path).head(nrows)

if __name__ == "__main__":
    df = load_data(data_path)
    print("data info:", df.info())

    print(df.head())
