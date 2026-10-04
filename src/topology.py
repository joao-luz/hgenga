import gc
import torch

from utils import get_embeddings

from torch_geometric.data import Data, HeteroData


def build_knn_graph(texts, embeddings=None, model_name='Qwen/Qwen3-VL-Embedding-2B', k=3, prompt=None):
    if embeddings is None:
        embeddings = get_embeddings(texts, prompt, model_name=model_name)

    sim_matrix = torch.matmul(embeddings, embeddings.T)

    n = len(texts)
    _,indices = torch.topk(sim_matrix, k=k+1, dim=1)

    edge_index_list = []
    for i in range(n):
        for j in indices[i][1:]:
            edge_index_list.append([i, j.item()])

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    return Data(x=embeddings, edge_index=edge_index)


import torch
from torch_geometric.data import HeteroData


def build_knn_heterogenous_graph(
    texts,
    descriptions,
    text_embeddings=None,
    description_embeddings=None,
    model_name='Qwen/Qwen3-VL-Embedding-2B',
    text_k=3,
    desc_k=5,
    prompt=None
):

    if text_embeddings is None:
        text_embeddings = get_embeddings(texts, prompt, model_name=model_name)

    if description_embeddings is None:
        description_embeddings = get_embeddings(descriptions, prompt, model_name=model_name)

    n_text = text_embeddings.size(0)
    n_desc = description_embeddings.size(0)

    text_sim = torch.matmul(text_embeddings, text_embeddings.T)
    _, text_indices = torch.topk(text_sim, k=text_k + 1, dim=1)

    text_text_edges = []
    for i in range(n_text):
        for j in text_indices[i][1:]:
            text_text_edges.append([i, j.item()])

    if len(text_text_edges) == 0:
        text_text_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        text_text_edge_index = torch.tensor(text_text_edges, dtype=torch.long).t().contiguous()

    cross_sim = torch.matmul(text_embeddings, description_embeddings.T)
    _, desc_indices = torch.topk(cross_sim, k=desc_k, dim=1)

    text_desc_edges = []
    for i in range(n_text):
        for j in desc_indices[i]:
            text_desc_edges.append([i, j.item()])

    if len(text_desc_edges) == 0:
        text_desc_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        text_desc_edge_index = torch.tensor(text_desc_edges, dtype=torch.long).t().contiguous()

    data = HeteroData()

    data["text"].x = text_embeddings
    data["description"].x = description_embeddings

    data["text", "similar", "text"].edge_index = text_text_edge_index
    data["text", "matches", "description"].edge_index = text_desc_edge_index
    data['description', 'rev_matches', 'text'].edge_index = text_desc_edge_index[[1, 0]]

    return data


def extend_graph_with_new_texts(data, new_embeddings, new_true_labels=None, new_pseudo_labels=None, k=10, rebuild_knn=True, device=None):
    if device is None:
        device = data.x.device

    data = data.to(device)
    new_embeddings = new_embeddings.to(device)

    old_N = data.num_nodes
    M = new_embeddings.size(0)

    data.gen_mask = torch.cat([data.gen_mask, torch.ones(M, dtype=torch.bool).to(device)], dim=0)

    data.x = torch.cat([data.x, new_embeddings], dim=0)

    if hasattr(data, "y") and new_true_labels is not None:
        new_true_labels = new_true_labels.to(device)
        data.y = torch.cat([data.y, new_true_labels], dim=0)

    if hasattr(data, "pseudo_y") and new_pseudo_labels is not None:
        new_pseudo_labels = new_pseudo_labels.to(device)
        data.pseudo_y = torch.cat([data.pseudo_y, new_pseudo_labels], dim=0)

    if rebuild_knn:
        sim_matrix = torch.matmul(data.x, data.x.T)

        n = data.x.shape[0]
        _,indices = torch.topk(sim_matrix, k=k+1, dim=1)

        edge_index_list = []
        for i in range(n):
            for j in indices[i][1:]:
                edge_index_list.append([i, j.item()])

        if len(edge_index_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        data.edge_index = edge_index

    data.num_nodes = old_N + M

    return data
