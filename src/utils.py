import numpy as np
import torch

from pathlib import Path


def load_yaml(path):
    import yaml

    with open(path) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    return data


def save_txt(text, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)

    with open(path, 'w') as f:
        f.write(text)


def save_json(data, path):
    import json

    Path(path).parent.mkdir(exist_ok=True, parents=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent='\t', ensure_ascii=False)

def load_json(path):
    import json

    def keystoint(pairs):
        return {int(k) if k.isdigit() else k: v for k, v in pairs}

    if Path(path).is_file():
        with open(path, 'r') as f:
            return json.load(f, object_pairs_hook=keystoint)

    return None

def save_tensor(tensor, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)

    torch.save(tensor, path)


def load_tensor(path):
    if Path(path).is_file():
        return torch.load(path)

    return None

def kmeans_sample(embeddings, k):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings.numpy())
            
    sampled_indices = []
    for center in kmeans.cluster_centers_:
        dist = np.linalg.norm(embeddings - center, axis=1)
        sampled_indices.append(np.argmin(dist))

    sampled_indices = list(set(sampled_indices))

    return torch.tensor(sampled_indices)


def get_embeddings(texts, prompt, model=None, model_name='Qwen/Qwen3-VL-Embedding-2B'):
    import gc
    import torch

    from sentence_transformers import SentenceTransformer
    
    load_from_name = model is None
    if load_from_name:
        model = SentenceTransformer(model_name)

    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, prompt=prompt).to(torch.float).cpu().clone()

    if load_from_name:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return embeddings


def append_report_row(report, file_name='results_log.csv', flush_file=False, extra_cols={}):
    import pandas as pd
    from datetime import datetime

    time_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    row_data = {
        'time': time_string,
        **extra_cols
    }
    
    for key, metrics in report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            row_data[f'f1_class_{key}'] = metrics['f1-score']
            
    row_data['f1-macro'] = report['macro avg']['f1-score']
    
    df_row = pd.DataFrame([row_data])
    
    print(file_name)
    file_exists = Path(file_name).is_file()
    Path(file_name).parent.mkdir(exist_ok=True, parents=True)
    
    mode = 'a' if not flush_file else 'w'
    header = (not file_exists) or flush_file
    print(file_exists, flush_file)
    df_row.to_csv(file_name, mode=mode, index=False, header=header)