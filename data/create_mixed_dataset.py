import sys
import os
import argparse
import json
import random
from collections import defaultdict


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from project_config.config import cfg


def load_and_filter_data(input_file, target_datasets):
    dataset_samples = {dataset: [] for dataset in target_datasets}
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            required_keys = ["table", "question", "category", "answer", "question_id"]
            if not all(key in item for key in required_keys):
                continue
            category = item["category"]
            dataset_prefix = category.split("_")[0] if "_" in category else category
            if dataset_prefix in target_datasets:
                dataset_samples[dataset_prefix].append(item)
    return dataset_samples


def sample_and_split_data(dataset_samples, samples_per_dataset, val_ratio=0.15):
    random.seed(42)
    train_data = []
    val_data = []
    for samples in dataset_samples.values():
        if not samples:
            continue
        available = len(samples)
        train_planned = min(available, samples_per_dataset)
        val_planned = max(1, int(train_planned * val_ratio))
        needed = train_planned + val_planned
        if available < needed:
            train_count = int(available * train_planned / needed)
            val_count = available - train_count
        else:
            train_count = train_planned
            val_count = val_planned
        total_required = train_count + val_count
        if available > total_required:
            selected = random.sample(samples, total_required)
        else:
            selected = samples.copy()
            random.shuffle(selected)
        train_data.extend(selected[:train_count])
        val_data.extend(selected[train_count:train_count + val_count])
    random.shuffle(train_data)
    random.shuffle(val_data)
    return train_data, val_data


def save_dataset(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_data_summary(train_data, val_data, target_datasets, output_dir):
    summary = {
        "created_by": "Mixed Dataset Creator",
        "total_samples": len(train_data) + len(val_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "val_ratio": (len(val_data) / (len(train_data) + len(val_data))) if (len(train_data) + len(val_data)) else 0.0,
        "target_datasets": target_datasets,
        "dataset_distribution": {"train": defaultdict(int), "val": defaultdict(int)},
    }
    for item in train_data:
        category = item["category"]
        dataset_prefix = category.split("_")[0] if "_" in category else category
        summary["dataset_distribution"]["train"][dataset_prefix] += 1
    for item in val_data:
        category = item["category"]
        dataset_prefix = category.split("_")[0] if "_" in category else category
        summary["dataset_distribution"]["val"][dataset_prefix] += 1
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Create mixed training and validation datasets")
    parser.add_argument("--input_file", type=str, default="data/your_data_path/train_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/your_data_path")
    parser.add_argument("--samples_per_dataset", type=int, default=2000)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()
    target_datasets = cfg["DATA"]["MIXED_DATASETS"]
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    dataset_samples = load_and_filter_data(args.input_file, target_datasets)
    train_data, val_data = sample_and_split_data(
        dataset_samples, args.samples_per_dataset, args.val_ratio
    )
    if not train_data:
        raise RuntimeError("No training data generated")
    train_file = os.path.join(args.output_dir, "mixed_train.jsonl")
    val_file = os.path.join(args.output_dir, "mixed_val.jsonl")
    save_dataset(train_data, train_file)
    save_dataset(val_data, val_file)
    create_data_summary(train_data, val_data, target_datasets, args.output_dir)


if __name__ == "__main__":
    main()
