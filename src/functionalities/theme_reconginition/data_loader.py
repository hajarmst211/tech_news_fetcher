import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

LABEL_MAPPING = {
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


def load_arxiv_data():
    dataset_handle = "devintheai/arxiv-cs-papers-multi-label-classification-200k-v1"

    try:
        download_path = kagglehub.dataset_download(dataset_handle)
        
        supported_extensions = ('.csv', '.tsv', '.json', '.jsonl', '.parquet', '.feather')
        all_files = os.listdir(download_path)
        data_files = [f for f in all_files if f.lower().endswith(supported_extensions)]

        if not data_files:
            raise FileNotFoundError(f"No supported data files found in the dataset directory: {download_path}")
        
        selected_file = data_files[0]
        
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            dataset_handle,
            selected_file,
        )
    except Exception as e:
        print(f"Failed to load the dataset via kagglehub: {e}")
        raise

    one_hot_cols = [col for col in df.columns if col in LABEL_MAPPING]

    if one_hot_cols:
        def get_first_active_label(row):
            for col in one_hot_cols:
                if row[col] == 1:
                    return col
            return "Unknown"

        df['primary_label'] = df.apply(get_first_active_label, axis=1)

        df['group_index'] = df.groupby('primary_label').cumcount()
        df = df.sort_values(by=['group_index', 'primary_label']).reset_index(drop=True)
        df = df.drop(columns=['group_index', 'primary_label'])

        df = df.rename(columns=LABEL_MAPPING)

    else:
        label_col = None
        for candidate in ['terms', 'categories', 'label', 'subcategories']:
            if candidate in df.columns:
                label_col = candidate
                break

        if label_col:
            def extract_primary_label(val):
                if isinstance(val, list) and len(val) > 0:
                    return val[0]
                elif isinstance(val, str):
                    cleaned = val.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
                    return parts[0] if parts else "Unknown"
                return "Unknown"

            df['primary_label'] = df[label_col].apply(extract_primary_label)

            df['group_index'] = df.groupby('primary_label').cumcount()
            df = df.sort_values(by=['group_index', 'primary_label']).reset_index(drop=True)
            df = df.drop(columns=['group_index'])

            df['primary_label'] = df['primary_label'].map(LABEL_MAPPING).fillna(df['primary_label'])
        else:
            print("Warning: Could not identify target label columns. Returning unsorted DataFrame.")

    return df


if __name__ == "__main__":
    df = load_arxiv_data()
    print("First 5 records:")
    print(df.head())
1