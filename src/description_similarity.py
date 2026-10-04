# For this test we generate multiple descriptions for the different classes
# using the LLM and classify texts via K nearest description embeddings. We
# select only texts that had high agreement (e.g. all K nearest descriptions
# are of the same class) to act as pseudo-labeled samples and propagate their
# labels to the rest of the graph

from utils import load_yaml, get_embeddings, save_json, load_json, append_report_row, save_tensor, load_tensor, kmeans_sample
from topology import build_knn_graph, build_knn_heterogenous_graph
from agents.multi_description_generator import MultiDescriptionsGenerator
from agents.example_generator import ExampleGenerator
from agents.description_generator import DescriptionGenerator
from agents.labeler import Labeler
from llm import LLM
from gnn import GNN, HeteroGNN, train_model, train_hetero_model, evaluate_model, evaluate_hetero_model

import torch
import logging
import random

from argparse import ArgumentParser
from datasets import load_dataset
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('--min_k', type=int, default=5)
    parser.add_argument('--dataset', type=str, choices=['ohsumed-single', 'agnews', 'r8', 'r52'])
    parser.add_argument('--configs_dir', type=str, default='../configs')
    parser.add_argument('--sentence_model', type=str, default='Qwen/Qwen3-VL-Embedding-2B')
    parser.add_argument('--llm', type=str, default='AxionML/Qwen3.5-9B-NVFP4')
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument('--setting', type=str, default='base')
    parser.add_argument('--examples', type=str, choices=['single', 'mult'], default='single')
    parser.add_argument('--llm_sample_size', type=int, default=0)
    parser.add_argument('--gt_proportion', type=float, default=0.0)
    parser.add_argument('--gt_descs', action='store_true', default=False)
    parser.add_argument('--knn_labels', type=str, default=None)
    parser.add_argument('--agreement_only', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='../outputs')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.5)

    return parser.parse_args()


def get_knn_labels(embeddings, description_embeddings, description_labels, k=5, min_k=None):
    min_k = min_k or k

    sims = (embeddings @ description_embeddings.T)
    _, indices = torch.topk(sims, k=k, dim=-1)
    top_labels = description_labels[indices]
    modes,_ = top_labels.mode()
    frequencies = (top_labels == modes.unsqueeze(1)).sum(dim=1)
    knn_labels = torch.where(frequencies >= min_k, modes, torch.tensor(-1))

    return knn_labels


def show_report(knn_labels, true):
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    agreement_mask = knn_labels != -1

    agreement = agreement_mask.sum()
    total = len(true)
    print(f'Agreement: {agreement}/{total} ({100*agreement/total:.2f}%)')

    print(classification_report(true[agreement_mask], knn_labels[agreement_mask], zero_division=0))
    cm = confusion_matrix(true[agreement_mask], knn_labels[agreement_mask])
    ConfusionMatrixDisplay(cm).plot(include_values=False)

    return classification_report(true[agreement_mask], knn_labels[agreement_mask], zero_division=0, output_dict=True)


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_name = args.dataset
    dataset_config = load_yaml(f'{args.configs_dir}/datasets/{dataset_name}.yaml')

    sanitized_llm_name = args.llm.replace('/', '--')
    sanitized_sentence_model_name = args.sentence_model.replace('/', '--')

    exp_name = 'desc_sim_'
    exp_name += args.setting
    exp_name += ('_mult' if args.examples == 'mult' and 'examples' in args.setting else '')
    exp_name += (f'_llm_labels_{args.llm_sample_size}' if args.llm_sample_size else '')
    exp_name += (f'_gt_descs' if args.gt_descs else '')
    exp_name += (f'_gt_prop_{args.gt_proportion:.1f}' if args.gt_proportion > 0 else '')
    exp_name += (f'_agreement_only' if args.agreement_only else '')
    exp_name += (f'_examples_knn_labels' if args.knn_labels == 'examples' else '_no_knn_labels' if args.knn_labels is None else '')
    exp_name = args.exp_name or exp_name

    output_dir = f'{args.output_dir}/{sanitized_llm_name}+{sanitized_sentence_model_name}/{dataset_name}/{exp_name}/{args.run}'
    cache_dir = f'{args.cache_dir}/{dataset_name}'
    llm_cache_dir = f'{args.cache_dir}/{dataset_name}/{sanitized_llm_name}/{args.run}'
    gt_llm_cache_dir = f'{args.cache_dir}/{dataset_name}/{sanitized_llm_name}/gt'

    logging.info(f'output_dir={output_dir}')

    logging.info(f'Loading dataset from "{dataset_config["path"]}"')
    dataset = load_dataset(dataset_config['path'])
    texts = list(dataset['test']['text'])
    true = torch.tensor(dataset['test'][dataset_config['label_feature']])
    id2label = dataset_config['id2label']
    class_task_desc = dataset_config['labeling_task_description']
    sentece_embed_prompt = dataset_config['sentence_embed_prompt']

    embeddings_file = f'{cache_dir}/embeddings_{sanitized_sentence_model_name}.pt'
    embs = load_tensor(embeddings_file)
    if embs is None:
        embs = get_embeddings(texts, sentece_embed_prompt, model_name=args.sentence_model)
        save_tensor(embs, embeddings_file)

    llm = LLM(args.llm, gpu_memory_utilization=0.8, max_model_len=6000)
    chat_kwargs = {
        'chat_template_kwargs': {
            'enable_thinking': False
        }
    }
    llm.set_default_chat_kwargs(chat_kwargs)

    multi_descs_agent = MultiDescriptionsGenerator(class_task_desc, id2label, llm)

    descriptions_file = f'{gt_llm_cache_dir if args.gt_descs else llm_cache_dir}/descriptions.json'
    descriptions = load_json(descriptions_file)

    if descriptions is None:
        logging.info(f'Obtaining descriptions')
        descriptions = multi_descs_agent.get_descriptions()
        save_json(descriptions, descriptions_file)
    else:
        logging.info(f'Loaded descriptions from "{descriptions_file}"')

    if 'examples' in args.setting or args.llm_sample_size > 0 or args.knn_labels == 'examples' or args.setting == 'llm_only':
        sample_texts = random.sample(texts, 5)

        desc_gen_agent = DescriptionGenerator(class_task_desc, id2label, llm)
        single_descriptions_file = f'{gt_llm_cache_dir if args.gt_descs else llm_cache_dir}/single_descriptions.json'
        single_descriptions = load_json(single_descriptions_file)

        if single_descriptions is None:
            logging.info(f'Obtaining single descriptions')
            single_descriptions,_ = desc_gen_agent.get_descriptions(sample_texts, return_llm_ouputs=False)
            save_json(single_descriptions, single_descriptions_file)
        else:
            logging.info(f'Loaded single descriptions from "{single_descriptions_file}"')

        example_gen_agent = ExampleGenerator(class_task_desc, id2label, llm)
        examples_file = f'{llm_cache_dir}/examples' + ('_mult' if args.examples == 'mult' else '') + '.json'
        examples = load_json(examples_file)

        if examples is None:
            logging.info(f'Obtaining examples')

            if args.examples == 'mult':
                examples = example_gen_agent.get_examples(descriptions, texts, 2)
            else:
                examples = example_gen_agent.get_examples(single_descriptions, texts, 10)

            save_json(examples, examples_file)
        else:
            logging.info(f'Loaded examples from "{examples_file}"')

    llm_labels = torch.full((len(texts),), -1)
    if args.llm_sample_size > 0 or args.setting == 'llm_only':
        labeler_agent = Labeler(class_task_desc, id2label, llm)

        if args.setting == 'llm_only':
            sample_ids = torch.arange(0, len(texts))
        else:
            sample_ids_file = f'{cache_dir}/kmeans_sample_{args.llm_sample_size}.pt'
            sample_ids = load_tensor(sample_ids_file)
            if sample_ids is None:
                sample_ids = kmeans_sample(embs, args.llm_sample_size)
                save_tensor(sample_ids, sample_ids_file)

        llm_labels_file = f'{llm_cache_dir}/labels.pt'
        llm_labels = load_tensor(llm_labels_file)
        if llm_labels is None:
            llm_labels = torch.full((len(texts),), -1)

        # Need to process only indices with no labels yet
        existing = (llm_labels != -1).nonzero().flatten()
        mask = ~torch.isin(sample_ids, existing)
        sample_ids_to_gen = sample_ids[mask]

        logging.info(f'Got {args.llm_sample_size - len(sample_ids_to_gen)}/{args.llm_sample_size} labels from "{llm_labels_file}"')

        sample_texts = [texts[i] for i in sample_ids_to_gen]

        if len(sample_texts) > 0:
            labels,_ = labeler_agent.get_labels(single_descriptions, sample_texts, return_llm_output=False)
            llm_labels[sample_ids_to_gen] = labels

            save_tensor(llm_labels, llm_labels_file)

        # Keep only the ones sampled
        curr_llm_labels = torch.full((len(texts),), -1)
        curr_llm_labels[sample_ids] = llm_labels[sample_ids]
        llm_labels = curr_llm_labels

    llm.unload_model()

    logging.info('Getting descriptions\' embeddings')
    descriptions_embeddings_file = f'{gt_llm_cache_dir if args.gt_descs else llm_cache_dir}/descriptions_embeddings_{sanitized_sentence_model_name}.pt'
    descriptions_labels = torch.tensor([i for i,descs in enumerate(descriptions.values()) for _ in descs])
    descriptions_flat = [d for descs in descriptions.values() for d in descs]
    descriptions_embeddings = load_tensor(descriptions_embeddings_file)
    if descriptions_embeddings is None:
        descriptions_embeddings = get_embeddings(descriptions_flat, sentece_embed_prompt)
        save_tensor(descriptions_embeddings, descriptions_embeddings_file)

    if 'examples' in args.setting or args.knn_labels == 'examples':
        examples_embeddings_file = f'{llm_cache_dir}/examples_embeddings{("_mult" if args.examples == "mult" else "")}_{sanitized_sentence_model_name}.pt'
        examples_labels = torch.tensor([i for i,exes in enumerate(examples.values()) for _ in exes])
        examples_flat = [e for exes in examples.values() for e in exes]
        examples_embeddings = load_tensor(examples_embeddings_file)
        if examples_embeddings is None:
            examples_embeddings = get_embeddings(examples_flat, sentece_embed_prompt)
            save_tensor(examples_embeddings, examples_embeddings_file)

    if args.knn_labels == 'descriptions':
        knn_labels = get_knn_labels(embs, descriptions_embeddings, descriptions_labels, args.k, args.min_k)
    elif args.knn_labels == 'examples':
        knn_labels = get_knn_labels(embs, examples_embeddings, examples_labels, args.k, args.min_k)
    elif args.knn_labels is None or args.setting == 'llm_only':
        knn_labels = torch.full((len(texts),), -1)

    if args.agreement_only:
        pseudo_labels = torch.full((len(texts),), -1)
        mask = (llm_labels != -1) & (llm_labels == knn_labels)
        pseudo_labels[mask] = llm_labels[mask]
    else:
        pseudo_labels = knn_labels.clone()
        pseudo_labels[llm_labels != -1] = llm_labels[llm_labels != -1]

    # Populate proportion of labels with ground truth
    if args.gt_proportion > 0:
        pseudo_labels_indices = (pseudo_labels != -1).nonzero().flatten()
        num_samples = int(pseudo_labels_indices.size(0) * args.gt_proportion)

        sample_indices = torch.randperm(pseudo_labels_indices.size(0))[:num_samples]
        sampled_pseudo_label_indices = pseudo_labels_indices[sample_indices]
        pseudo_labels[sampled_pseudo_label_indices] = true[sampled_pseudo_label_indices]

    save_tensor(pseudo_labels, f'{output_dir}/pseudo_labels.pt')
    show_report(pseudo_labels, true)

    # Base case, kNN graph of documents only
    if args.setting == 'base':
        data = build_knn_graph(texts, embeddings=embs, k=args.k)

        data.y = true
        data.id2label = id2label
        data.pseudo_y = pseudo_labels
        data.test_mask = torch.ones(data.pseudo_y.shape[0], dtype=torch.bool)
    
    # Generate examples via the LLM and use them as artificial nodes in the kNN graph
    elif args.setting == 'examples':
        data = build_knn_graph(texts + examples_flat, 
                               embeddings=torch.concat((embs, examples_embeddings)), k=args.k)
        
        data.y = torch.concat((true, examples_labels))
        data.id2label = id2label
        data.pseudo_y = torch.concat((pseudo_labels, examples_labels))
        data.test_mask = torch.ones(data.pseudo_y.shape[0], dtype=torch.bool)
        data.test_mask[-len(examples_flat):] = False

    # Use class descriptions as heterogenous nodes connected to similar documents
    elif args.setting == 'hetero':
        data = build_knn_heterogenous_graph(texts, descriptions_flat, 
                                            text_embeddings=embs, 
                                            description_embeddings=descriptions_embeddings,
                                            text_k=args.k, desc_k=args.k)
        
        data['text'].y = true
        data['text'].id2label = id2label
        data['text'].pseudo_y = pseudo_labels
        data.test_mask = torch.ones(data['text'].pseudo_y.shape[0], dtype=torch.bool)

    # Use class descriptions as heterogenous nodes connected to similar documents
    elif args.setting == 'examples_as_hetero':
        data = build_knn_heterogenous_graph(texts, examples_flat, 
                                            text_embeddings=embs, 
                                            description_embeddings=examples_embeddings,
                                            text_k=args.k, desc_k=args.k)
        
        data['text'].y = true
        data['text'].id2label = id2label
        data['text'].pseudo_y = pseudo_labels
        data.test_mask = torch.ones(data['text'].pseudo_y.shape[0], dtype=torch.bool)
    
    # Use both artificial document nodes and heterogenous description nodes
    elif args.setting == 'hetero_examples':
        data = build_knn_heterogenous_graph(texts + examples_flat, descriptions_flat, 
                                            text_embeddings=torch.concat((embs, examples_embeddings)), 
                                            description_embeddings=descriptions_embeddings,
                                            text_k=args.k, desc_k=args.k)
        
        data['text'].y = torch.concat((true, examples_labels))
        data['text'].id2label = id2label
        data['text'].pseudo_y = torch.concat((pseudo_labels, examples_labels))
        data.test_mask = torch.ones(data['text'].pseudo_y.shape[0], dtype=torch.bool)
        data.test_mask[-len(examples_flat):] = False

    llm_label_mask = torch.concat((llm_labels != -1, torch.zeros(data.test_mask.shape[0] - llm_labels.shape[0], dtype=torch.bool)))
    if args.setting in ['base', 'examples']:
        model = GNN(data.x.size(-1), data.x.size(-1)//2, len(id2label))
        train_model(model, data, lr=1e-3)
        _, _, propagated_labels, graph_embs = evaluate_model(model, data, test_mask=data.test_mask)
    
    elif args.setting in ['hetero', 'hetero_examples', 'examples_as_hetero']:
        model = HeteroGNN(data['text'].x.size(-1)//2, len(id2label))
        train_hetero_model(model, data, epochs=200, lr=5e-4, llm_labels=llm_label_mask, alpha=args.alpha)
        _, _, propagated_labels, graph_embs = evaluate_hetero_model(model, data, test_mask=data.test_mask)

    elif args.setting == 'llm_only':
        propagated_labels = pseudo_labels

    report = show_report(propagated_labels, true)
    save_tensor(propagated_labels, f'{output_dir}/propagated_labels.pt')

    save_tensor(data, f'{output_dir}/data.pt')
    save_tensor(graph_embs, f'{output_dir}/graph_embs.pt')

    results_path = f'{output_dir}/../results.csv'
    extra_cols = {
        'experiment': exp_name,
        'run': args.run,
    }
    append_report_row(report, results_path, extra_cols=extra_cols)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)