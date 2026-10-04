from llm import LLM
from topology import build_knn_graph, extend_graph_with_new_texts
from utils import get_embeddings, load_yaml, save_txt, save_tensor, save_json, append_report_row
from agents.description_generator import DescriptionGenerator
from agents.labeler import Labeler

import torch
import torch.nn.functional as F
import gc
import logging

from utils import kmeans_sample
from gnn import GNN, train_model, evaluate_model
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import k_hop_subgraph
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['ohsumed-single', 'agnews', 'r8', 'r52'])
    parser.add_argument('--sentence_model', type=str, default='Qwen/Qwen3-VL-Embedding-2B')
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('--llm', type=str, default='AxionML/Qwen3.5-9B-NVFP4')
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument('--log_dir', type=str, default='../log')
    parser.add_argument('--output_dir', type=str, default='../outputs')
    parser.add_argument('--max_iter', type=int, default=3)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--configs_dir', type=str, default='../configs')

    parser.add_argument('--dont_use_artificial_nodes', action='store_true', default=False)
    parser.add_argument('--dont_use_class_similariy', action='store_true', default=False)
    parser.add_argument('--dont_use_sampled_examples', action='store_true', default=False)
    
    return parser.parse_args()


def topk_consensus_semantic_nodes(data, labels, class_embeddings, k=5, hops=2, min_neighbors=5, alpha=0.5, normalize_embeddings=True):
    edge_index = data.edge_index
    x = data.x
    N = data.num_nodes
    num_classes = class_embeddings.size(0)

    if normalize_embeddings:
        x = F.normalize(x, dim=1)
        class_embeddings = F.normalize(class_embeddings, dim=1)

    class_scores = defaultdict(list)

    for node in range(N):
        if data.gen_mask[node]:
            continue
        
        node_label = labels[node].item()

        subset, _, _, _ = k_hop_subgraph(node, hops, edge_index, relabel_nodes=False)

        subset = subset[subset != node]

        if len(subset) < min_neighbors:
            continue

        neighbor_labels = labels[subset]
        same_label_count = (neighbor_labels == node_label).sum().item()
        purity = same_label_count / len(subset)

        similarity = torch.dot(x[node], class_embeddings[node_label]).item()

        score = alpha * purity + (1 - alpha) * similarity

        class_scores[node_label].append((node, score))

    result = {}

    for c in range(num_classes):
        if c not in class_scores:
            result[c] = []
            continue

        sorted_nodes = sorted(class_scores[c], key=lambda x: x[1], reverse=True)

        result[c] = [node for node, _ in sorted_nodes[:k]]

    return result


def get_examples(data, embeddings, propagated_y, texts, id2label):
    consensus = topk_consensus_semantic_nodes(data.cpu(), propagated_y.detach().cpu(), embeddings)

    positive = defaultdict(list)
    negative = defaultdict(list)
    for label, nodes in consensus.items():
        for node in nodes:
            true_label = data.y[node]

            if label != true_label:
                negative[label].append({'idx': node, 'text': texts[node], 'pred': id2label[label], 'true': id2label[true_label.item()]})
            else:
                positive[label].append({'idx': node, 'text': texts[node], 'pred': id2label[label], 'true': id2label[true_label.item()]})

    return positive, negative


def compute_class_similarity_matrix(embeddings, labels, normalize_embeddings=True):
    if embeddings.dim() != 2:
        raise ValueError("Embeddings must be [N, D]")

    if len(embeddings) != len(labels):
        raise ValueError("Embeddings and labels must have same length")

    embeddings = embeddings.float()

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    class_dict = defaultdict(list)
    for emb, label in zip(embeddings, labels):
        class_dict[int(label)].append(emb)

    class_ids = sorted(class_dict.keys())
    centroids = []

    for class_id in class_ids:
        class_embs = torch.stack(class_dict[class_id])
        centroid = class_embs.mean(dim=0)
        centroids.append(centroid)

    class_centroids = torch.stack(centroids)
    class_centroids = F.normalize(class_centroids, p=2, dim=1)

    sim_matrix = class_centroids @ class_centroids.T

    return sim_matrix, class_ids, class_centroids


def get_top_k_similarities(matrix, k=3):
    matrix.fill_diagonal_(float('-inf'))
    _, top_k_indices = torch.topk(matrix, k, dim=1)
    
    return top_k_indices


def main(args):
    log_path = f'{args.log_dir}/{args.dataset}/{args.exp_name}/run{args.run}.log'
    Path(log_path).parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f'{log_path}', filemode='w', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_name = args.dataset
    dataset_config = load_yaml(f'{args.configs_dir}/datasets/{dataset_name}.yaml')

    logging.info(f'Loading dataset "{dataset_config['path']}"')

    dataset = load_dataset(dataset_config['path'])
    texts = list(dataset['test']['text'])
    labels = list(dataset['test'][dataset_config['label_feature']])
    id2label = dataset_config['id2label']
    class_task_desc = dataset_config['labeling_task_description']
    sentece_embed_prompt = dataset_config['sentence_embed_prompt']

    sanitized_llm_name = args.llm.replace('/', '--')
    
    logging.info(f'Building kNN graph with k={args.k} and model "{args.sentence_model}"')

    embs = None
    embeddings_file = f'{args.cache_dir}/{dataset_name}/embeddings_{args.sentence_model.replace('/', '--')}.pt'
    if Path(embeddings_file).is_file():
        logging.info(f'Loading embeddings from "{embeddings_file}"')
        embs = torch.load(embeddings_file)

    data = build_knn_graph(texts, embeddings=embs, k=args.k, model_name=args.sentence_model, prompt=sentece_embed_prompt)
    embs = data.x

    if not Path(embeddings_file).is_file():
        save_tensor(embs, embeddings_file)

    data.y = torch.tensor(labels)
    data.id2label = id2label

    logging.info(f'Loading LLM "{args.llm}"')

    llm = LLM(args.llm, load_model=True, gpu_memory_utilization=0.8, max_model_len=6000)
    chat_kwargs = {
        'chat_template_kwargs': {
            'enable_thinking': False
        }
    }
    llm.set_default_chat_kwargs(chat_kwargs)

    output_dir = f'{args.output_dir}/{dataset_name}/{args.exp_name}/{args.run}'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    logging.info(f'Sampling {args.sample_size} nodes')

    sample_indices = kmeans_sample(embs.numpy(), args.sample_size)
    sampled_texts = [texts[i] for i in sample_indices]

    # Save indices and texts
    logging.info(f'Saving indices and texts')
    save_txt('\n'.join([str(i) for i in sample_indices]), f'{output_dir}/sample/ids.txt')
    for i,text in zip(sample_indices, sampled_texts):
        save_txt(text, f'{output_dir}/sample/{i}.txt')

    pos_examples = defaultdict(list)
    neg_examples = defaultdict(list)
    extended_data = data.clone()
    extended_data.gen_mask = torch.zeros(extended_data.num_nodes, dtype=torch.bool)
    extended_texts = texts
    extended_labels = labels
    similar_classes = None
    prev_labels = None
    prev_descs = None

    desc_gen_agent = DescriptionGenerator(class_task_desc, id2label, llm)
    labeler_agent = Labeler(class_task_desc, id2label, llm)

    for iter in range(args.max_iter):
        logging.info(f'Starting iteration {iter}')

        logging.info(f'Obtaining descriptions and generating examples')
        descriptions, gen_examples, (prompts, responses) = desc_gen_agent.get_descriptions(similar_classes=similar_classes,
                                                                                           positive_examples=pos_examples,
                                                                                           negative_examples=neg_examples)
        prev_descs = descriptions
        flat_gen_examples = [e for es in gen_examples for e in es]
        extended_texts += flat_gen_examples
        gen_labels = torch.tensor([l for label,examples in enumerate(gen_examples) for l in [label]*len(examples)])

        # Save LLM outputs
        logging.info(f'Saving LLM ouputs')
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            save_txt(prompt, f'{output_dir}/iters/{iter}/desc_gen/{sanitized_llm_name}_ouputs/prompts/{i}.txt')
            save_txt(response, f'{output_dir}/iters/{iter}/desc_gen/{sanitized_llm_name}_ouputs/responses/{i}.txt')

        # Save descriptions and examples
        logging.info(f'Saving descriptions and examples')
        for i,desc in enumerate(descriptions):
            class_name = '_'.join(id2label[i].split())
            save_txt(desc, f'{output_dir}/iters/{iter}/descriptions/{i}-{class_name}.txt')

        for i,examples in enumerate(gen_examples):
            for j,example in enumerate(examples):
                save_txt(example, f'{output_dir}/iters/{iter}/examples/{i}/{j}.txt')

        logging.info(f'Obtaining pseudo labels')
        label_dicts, (prompts, responses) = labeler_agent.get_labels(descriptions, sampled_texts, prev_descriptions=prev_descs, prev_labels=prev_labels)

        # Save LLM outputs
        logging.info(f'Saving LLM ouputs')
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            save_txt(prompt, f'{output_dir}/iters/{iter}/labeler/{sanitized_llm_name}_ouputs/prompts/{i}.txt')
            save_txt(response, f'{output_dir}/iters/{iter}/labeler/{sanitized_llm_name}_ouputs/responses/{i}.txt')

        pseudo_labels = torch.tensor([d['label'] for d in label_dicts])
        prev_labels = pseudo_labels.tolist()

        pseudo_y = torch.full((extended_data.num_nodes,), fill_value=-1)
        pseudo_y[sample_indices] = torch.tensor(pseudo_labels)
        extended_data.pseudo_y = pseudo_y

        # Save labels
        logging.info(f'Saving pseudo labels')
        save_tensor(pseudo_labels, f'{output_dir}/iters/{iter}/labels/sample_pseudo_labels.pt')
        save_tensor(pseudo_y, f'{output_dir}/iters/{iter}/labels/pseudo_labels.pt')
        
        # Extract generated example's embeddings in case we'll need it
        if not args.dont_use_artificial_nodes or not args.dont_use_class_similariy:
            logging.info(f'Unloading LLM "{args.llm}"')
            llm.unload_model()

            logging.info(f'Extracting generated examples embeddings')
            examples_embeddings = get_embeddings(flat_gen_examples, prompt=sentece_embed_prompt, model_name=args.sentence_model)

        if not args.dont_use_artificial_nodes:
            logging.info(f'Extending graph with generated nodes')
            extended_data = extend_graph_with_new_texts(extended_data, examples_embeddings, gen_labels, gen_labels, k=args.k)

        logging.info(f'Training GNN on pseudo labels')
        model = GNN(extended_data.x.shape[1], hidden_channels=extended_data.x.shape[1]//2, out_channels=int(extended_data.y.max() + 1))
        train_model(model, extended_data, 200)

        logging.info(f'Evaluating GNN')
        report, _, propagated_y = evaluate_model(model, extended_data, test_mask=~extended_data.gen_mask, output_report_dict=True)

        results_path = f'{output_dir}/results.csv'

        logging.info(f'Saving GNN results to "{results_path}" with iter={iter}')
        append_report_row(report, args.exp_name, args.run, iter, results_path, flush_file=iter == 0)

        if not args.dont_use_sampled_examples:
            if llm.loaded:
                logging.info(f'Unloading LLM "{args.llm}"')
                llm.unload_model()

            logging.info(f'Sampling positive and negative examples')

            logging.info(f'Extracting description embeddings')
            desc_embeddings = get_embeddings(descriptions, prompt=sentece_embed_prompt, model_name=args.sentence_model)

            pos_examples, neg_examples = get_examples(extended_data, desc_embeddings, propagated_y, extended_texts, id2label)

            # Saved sampled examples
            merged = {c: pos_examples[c] + neg_examples[c] for c in pos_examples}
            logging.info(f'Saving positive and negative examples')
            save_json(merged, f'{output_dir}/iters/{iter}/sampled_examples/examples.json')

        if not args.dont_use_class_similariy:
            logging.info(f'Obtaining similarity matrix')
            sim_matrix, _, _ = compute_class_similarity_matrix(examples_embeddings, gen_labels)
            similar_classes = get_top_k_similarities(sim_matrix, k=5)

    logging.info(f'Finished running. Outputs saved to "{output_dir}"')


if __name__ == '__main__':
    args = parse_args()
    main(args)