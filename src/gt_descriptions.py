import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from agents.gt_description_generator import ClassDescriptionGenerator
from agents.multi_description_generator import MultiDescriptionsGenerator
from llm import LLM
from utils import load_yaml, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Generate class descriptions using an LLM.")

    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--split", default="test")

    parser.add_argument("--llm", default="AxionML/Qwen3.5-9B-NVFP4")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=38000)

    parser.add_argument("--gt-num-descriptions", type=int, default=100)
    parser.add_argument("--gt-temperature", type=float, default=0.0)

    parser.add_argument("--multi-sample-size", type=int, default=100)

    parser.add_argument("--generate-gt", action="store_true")
    parser.add_argument("--generate-multi", action="store_true")

    parser.add_argument("--output-dir", default="../cache")

    return parser.parse_args()


def build_examples_by_class(texts, labels, id2label):
    examples_by_class = {}

    for c in id2label:
        indices = (labels == c).nonzero().flatten()
        examples_by_class[c] = [texts[i.item()] for i in indices]

    return examples_by_class


def load_llm(llm_name, gpu_memory_utilization, max_model_len):
    llm = LLM(
        llm_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        load_model=True
    )

    llm.set_default_chat_kwargs({
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    })

    return llm


def process_dataset(args, dataset_name, llm):
    print(f"\nProcessing dataset: {dataset_name}")

    dataset_config = load_yaml(f"../configs/datasets/{dataset_name}.yaml")
    id2label = dataset_config["id2label"]

    dataset = load_dataset(dataset_config["path"])

    texts = list(dataset[args.split]["text"])
    labels = torch.tensor(dataset[args.split][dataset_config['label_feature']])

    class_task_prompt = dataset_config["sentence_embed_prompt"]

    examples_by_class = build_examples_by_class(
        texts,
        labels,
        id2label,
    )

    output_dir = (
        Path(args.output_dir)
        / dataset_name
        / args.llm.replace("/", "--")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_gt:
        print(f"[{dataset_name}] Generating GT descriptions")

        generator = ClassDescriptionGenerator(
            class_task_prompt,
            id2label,
            llm,
        )

        descriptions = generator.get_descriptions(
            examples_by_class,
            100,
            temperature=args.gt_temperature,
        )

        save_json(descriptions, output_dir / "gt_single_descriptions.json")

    if args.generate_multi:
        print(f"[{dataset_name}] Generating multi descriptions")

        generator = MultiDescriptionsGenerator(
            class_task_prompt,
            id2label,
            llm,
        )

        descriptions = generator.get_descriptions(
            examples_by_class,
            100
        )

        save_json(descriptions, output_dir / "gt_descriptions.json")


def main():
    args = parse_args()

    if not args.generate_gt and not args.generate_multi:
        args.generate_gt = True
        args.generate_multi = True

    print(f"Loading LLM: {args.llm}")

    llm = load_llm(
        args.llm,
        args.gpu_memory_utilization,
        args.max_model_len,
    )

    for dataset_name in args.datasets:
        process_dataset(args, dataset_name, llm)


if __name__ == "__main__":
    main()