import json
import os
from pathlib import Path
import shutil
from typing import List, Dict
import csv
import json
import os
import sys
import kagglehub
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from embeddings import CATEGORY_DESCRIPTIONS, USER_FRIENDLY_NAMES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'cs_papers_api.csv')

sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
from data_loader import load_data

class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device=None):
        """
        - all-MiniLM-L6-v2       -> N = 384  (Fast & lightweight, recommended default)
        - BAAI/bge-small-en-v1.5 -> N = 384  (High accuracy)
        - all-mpnet-base-v2      -> N = 768  (Higher quality, slightly slower)
        """
        print(f"Loading embedding model: '{model_name}'...")
        self.model = SentenceTransformer(model_name, device=device)
        self.vector_size = self.model.get_embedding_dimension()
        print(f"Model loaded. Output vector size (N) = {self.vector_size}\n")

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embeds a list of texts in batches and returns a 2D NumPy array (M, N)."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


def download_data(path: str = "./data/", verbose=False) -> Path:
    target_path = Path(path)
    expected_file = target_path / "arxiv-metadata-oai-snapshot.json"

    if expected_file.exists():
        if verbose:
            print(
                f"Data already exists at: {target_path.resolve()}. Skipping download."
            )
        return target_path

    target_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Downloading dataset via kagglehub...")
    downloaded_cache_path = kagglehub.dataset_download("Cornell-University/arxiv")

    if verbose:
        print(f"Copying files to: {target_path.resolve()}")
    for item in os.listdir(downloaded_cache_path):
        src_item = os.path.join(downloaded_cache_path, item)
        dst_item = target_path / item

        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy(src_item, dst_item)

    if verbose:
        print(f"Data successfully loaded into: {target_path.resolve()}")
    return target_path


def load_json(
    path: str = "./data/arxiv-metadata-oai-snapshot.json",
    number_of_elements=None,
) -> list:
    data = []

    with open(path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if number_of_elements is not None and i >= number_of_elements:
                break

            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def prepare_data(
    sample_data: list,
    path: str = "./data/prepared/",
    include_title: bool = True,
    include_abstract: bool = True,
):
    base_path = Path(path)
    base_path.mkdir(parents=True, exist_ok=True)

    for paper in tqdm(
        sample_data, desc="Preparing files", unit="file", total=len(sample_data)
    ):
        paper_id = paper.get("id", "unknown_id")
        safe_paper_id = paper_id.replace("/", "_")

        raw_categories = paper.get("categories")
        categories = raw_categories.split() if raw_categories else ["uncategorized"]

        title = " ".join(paper.get("title", "").split())
        abstract = " ".join(paper.get("abstract", "").split())

        content_parts = []
        if include_title and title:
            content_parts.append(f"Title: {title}")
        if include_abstract and abstract:
            content_parts.append(f"Abstract: {abstract}")

        content_str = (
            "\n\n".join(content_parts) if content_parts else f"Paper ID: {paper_id}"
        )
        for cat in categories:
            cat_dir = base_path / cat
            cat_dir.mkdir(parents=True, exist_ok=True)

            file_path = cat_dir / f"{safe_paper_id}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content_str)

    print(f"\nData successfully prepared under: {base_path.resolve()}")


def compute_vectorized_cosine_similarity(
    embeddings_matrix: np.ndarray, category_vector: np.ndarray
) -> np.ndarray:
    """Computes cosine similarity between a 2D matrix of paper embeddings (M, N)

    and a 1D category embedding vector (N) in a single vectorized BLAS operation.
    """
    matrix = np.asarray(embeddings_matrix, dtype=np.float32)
    vector = np.asarray(category_vector, dtype=np.float32).flatten()

    dot_products = matrix @ vector

    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)

    denominators = matrix_norms * vector_norm
    denominators[denominators == 0] = 1e-10

    return dot_products / denominators


def init_embder(model_name: str = "all-MiniLM-L6-v2") -> TextEmbedder:
    """Initializes and returns the TextEmbedder instance."""
    return TextEmbedder(model_name=model_name)


def pre_compute_category_embeddings(
    embedder: TextEmbedder, category_descriptions: dict, batch_size: int = 64
) -> dict:
    """Pre-computes embeddings for category descriptions using batched inference."""
    print("Embedding category descriptions in batch...")
    clean_keys = [cat_key.rstrip("/") for cat_key in category_descriptions.keys()]
    descriptions = list(category_descriptions.values())

    embeddings = embedder.embed_batch(descriptions, batch_size=batch_size)

    return {key: emb.tolist() for key, emb in zip(clean_keys, embeddings)}


def load_or_compute_category_embeddings(
    embedder: TextEmbedder,
    category_descriptions: dict,
    storage_dir: str = "./data/precomputation/embeddings",
) -> dict:
    """Loads cached category embeddings if available, otherwise computes them in batch."""
    dir_path = Path(storage_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    embeddings_file = dir_path / "category_embeddings.json"

    needed_count = len(category_descriptions)

    if embeddings_file.exists():
        try:
            with open(embeddings_file, "r", encoding="utf-8") as f:
                category_embeddings = json.load(f)

            existing_count = len(category_embeddings)

            if existing_count == needed_count:
                print(
                    f"Loaded {existing_count} category embeddings from cache: '{embeddings_file}'."
                )
                return category_embeddings
            else:
                print(
                    f"\n[Notice] Existing embeddings file found at '{embeddings_file}'."
                )
                print(f"Categories needed: {needed_count}")
                print(f"Categories found in file: {existing_count}")

                response = (
                    input(
                        "Do you want to overwrite the existing embeddings file? (y/n): "
                    )
                    .strip()
                    .lower()
                )

                if response in ["y", "yes"]:
                    print("Overwriting existing category embeddings file...")
                else:
                    print("Using existing cached embeddings without overwriting...\n")
                    return category_embeddings
        except Exception as e:
            print(f"Failed to read existing embeddings file ({e}). Recomputing...")

    category_embeddings = pre_compute_category_embeddings(
        embedder, category_descriptions
    )

    with open(embeddings_file, "w", encoding="utf-8") as f:
        json.dump(category_embeddings, f)
    print(f"Precomputed category embeddings saved to '{embeddings_file}'.\n")

    return category_embeddings


def chunked_iterable(iterable: list, chunk_size: int):
    """Yields successive chunks from a list for batch loading."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def process_papers_in_each_category(
    embedder: TextEmbedder,
    category_embeddings: dict,
    prepared_dir: str = "./data/prepared/",
    batch_size: int = 64,
    file_chunk_size: int = 512,
) -> list:
    """Processes papers per category using chunked I/O batching and matrix similarity."""
    base_dir = Path(prepared_dir)
    if not base_dir.exists():
        print(f"Error: Prepared data path '{base_dir}' does not exist.")
        return []

    category_folders = [f for f in base_dir.iterdir() if f.is_dir()]
    total_files = sum(len(list(cat_dir.glob("*.txt"))) for cat_dir in category_folders)
    print(
        f"Found {len(category_folders)} category folders containing {total_files} total paper files in '{base_dir}'.\n"
    )

    results = []

    cat_pbar = tqdm(
        sorted(category_folders), desc="Processing Categories", unit="category"
    )
    for cat_dir in cat_pbar:
        cat_name = cat_dir.name
        paper_files = list(cat_dir.glob("*.txt"))

        if not paper_files:
            continue

        cat_pbar.set_postfix({"current_cat": cat_name, "files": len(paper_files)})

        if cat_name in category_embeddings:
            cat_embedding = np.array(category_embeddings[cat_name], dtype=np.float32)
        else:
            cat_desc = f"{cat_name} scientific papers and research"
            cat_embedding = embedder.embed(cat_desc)

        all_file_names = []
        all_similarities = []

        for file_chunk in chunked_iterable(paper_files, file_chunk_size):
            paper_texts = []
            chunk_file_names = []

            for file_path in file_chunk:
                with open(file_path, "r", encoding="utf-8") as f:
                    paper_texts.append(f.read())
                    chunk_file_names.append(file_path.name)

            paper_embeddings = embedder.embed_batch(paper_texts, batch_size=batch_size)

            chunk_similarities = compute_vectorized_cosine_similarity(
                paper_embeddings, cat_embedding
            )

            all_file_names.extend(chunk_file_names)
            all_similarities.extend(chunk_similarities.tolist())

        avg_similarity = float(np.mean(all_similarities)) if all_similarities else 0.0
        std_similarity = float(np.std(all_similarities)) if all_similarities else 0.0

        results.append(
            {
                "category": cat_name,
                "file_names": all_file_names,
                "similarities": all_similarities,
                "avg_similarity": avg_similarity,
                "std_similarity": std_similarity,
            }
        )

    return results


def print_results(results: list):
    """Prints formatted similarity statistics and individual paper scores."""
    for res in results:
        cat_name = res["category"]
        file_names = res["file_names"]
        similarities = res["similarities"]
        avg_similarity = res["avg_similarity"]
        std_similarity = res["std_similarity"]

        print("=" * 60)
        print(f"Category: {cat_name}")
        print(f"Total Papers Evaluated: {len(similarities)}")
        print(f"Average Cosine Similarity : {avg_similarity:.4f}")
        print(f"Standard Deviation        : {std_similarity:.4f}")
        print("=" * 60 + "\n")


def pre_train():
    sample_data = load_csv(DATA_PATH, number_of_elements=1000)
    prepare_data(sample_data)


def train():
    embedder = init_embder()
    category_embeddings = load_or_compute_category_embeddings(
        embedder, CATEGORY_DESCRIPTIONS
    )
    results = process_papers_in_each_category(
        embedder, category_embeddings, batch_size=64, file_chunk_size=512
    )
    print_results(results)


def predict(
    text,
    embedder=None,
    category_embeddings=None,
    top_k=None,
    use_softmax: bool = False,
    temperature: float = 0.1,
) -> Dict[str, float]:
    if not text or not text.strip():
        raise ValueError("Input text for prediction cannot be empty.")

    if embedder is None:
        embedder = init_embder()

    if category_embeddings is None:
        category_embeddings = load_or_compute_category_embeddings(
            embedder, CATEGORY_DESCRIPTIONS
        )

    text_embedding = embedder.embed(text)

    categories = list(category_embeddings.keys())
    cat_matrix = np.array(
        [category_embeddings[cat] for cat in categories], dtype=np.float32
    )

    text_norm = np.linalg.norm(text_embedding)
    text_vec_norm = text_embedding / text_norm if text_norm > 0 else text_embedding

    cat_norms = np.linalg.norm(cat_matrix, axis=1, keepdims=True)
    cat_norms[cat_norms == 0] = 1e-10
    cat_matrix_norm = cat_matrix / cat_norms

    similarities = cat_matrix_norm @ text_vec_norm

    if use_softmax:
        scaled_sims = similarities / temperature
        exp_sims = np.exp(scaled_sims - np.max(scaled_sims))
        scores = exp_sims / np.sum(exp_sims)
    else:
        scores = similarities

    category_scores = dict(
        sorted(
            zip(categories, map(float, scores)),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    if top_k is not None and top_k > 0:
        category_scores = dict(list(category_scores.items())[:top_k])

    return category_scores


def print_predictions(
    predictions: dict, max_categories: int = 5, temperature: float = 0.05
):
    if not predictions:
        print("No predictions available.")
        return

    max_score = max(predictions.values())
    exp_scores = {
        cat: np.exp((score - max_score) / temperature)
        for cat, score in predictions.items()
    }
    total_exp = sum(exp_scores.values())

    probabilities = {cat: (exp / total_exp) * 100 for cat, exp in exp_scores.items()}

    sorted_preds = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_preds = sorted_preds[:max_categories]

    header = f"\nTop {len(top_preds)} Predicted Categories (Probabilities):"
    print(header)
    print("-" * 55)

    for category_code, probability in top_preds:
        name = USER_FRIENDLY_NAMES.get(category_code, category_code)
        print(f"  {name:<42} : {probability:6.2f}%")

    print("-" * 55)


def load_csv(
    path: str = "./data/cs_papers_api.csv",
    number_of_elements=None,
) -> list:
    data = []
    path_obj = Path(path)

    # Resolve path if running the script from inside the 'cosin_similarity' directory
    if not path_obj.exists():
        alternative_path = Path("../data/cs_papers_api.csv")
        if alternative_path.exists():
            path_obj = alternative_path
        else:
            raise FileNotFoundError(
                f"Could not find CSV file at '{path}' or '{alternative_path.resolve()}'"
            )

    print(f"Loading data from: {path_obj.resolve()}")
    with open(path_obj, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if number_of_elements is not None and i >= number_of_elements:
                break
            
            # Map standard CSV column headers to the structure expected by prepare_data
            paper = {
                "id": row.get("id", row.get("paper_id", f"paper_{i}")),
                "categories": row.get("categories", row.get("category", "uncategorized")),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", row.get("summary", ""))
            }
            data.append(paper)

    return data

if __name__ == "__main__":
    '''
    prediction = predict(
        "We present a novel deep learning architecture based on Vision Transformers (ViTs) "
        "for automated segmentation of brain tumors from multi-modal MRI scans. "
        "By incorporating multi-scale spatial attention mechanisms, our model captures "
        "fine-grained boundaries while significantly reducing false positives in low-contrast regions."
    )
    print_predictions(prediction, max_categories=5)
    '''
    pre_train()
    embedder = init_embder()
    category_embeddings = load_or_compute_category_embeddings(
        embedder, CATEGORY_DESCRIPTIONS
    )

    prepared_base = Path("./data/prepared/")
    all_files = list(prepared_base.glob("**/*.txt"))

    if all_files:
        for file_path in all_files[:10]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            print(f"\nEvaluating File: {file_path.name}")
            print(f"True Category  : {file_path.parent.name}")
            
            predictions = predict(
                content, 
                embedder=embedder, 
                category_embeddings=category_embeddings
            )
            print_predictions(predictions, max_categories=3)
    else:
        print("No prepared files found to predict.")