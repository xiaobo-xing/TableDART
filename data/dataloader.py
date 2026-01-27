import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from models.prompt.markdown_prompt import markdown_template_mapping, DEFAULT_MARKDOWN_PROMPT_BODY
from models.prompt.image_prompt import image_template_mapping, DEFAULT_IMAGE_PROMPT_TEMPLATE
from project_config.config import cfg

class ImageNotFoundError(Exception):
    """Exception raised when table image file is not found."""

    def __init__(self, question_id, image_file, image_path, message='Image file not found'):
        self.question_id = question_id
        self.image_file = image_file
        self.image_path = image_path
        self.message = f'{message}: {image_path}'
        super().__init__(self.message)

class TabularDataset(Dataset):
    """
    PyTorch Dataset for multimodal table understanding.
    Handles both text and image modalities with support for gating network training.
    """

    def __init__(self, data_path, target_tokenizer_name, max_seq_len_target, table_image_dir, filter_dataset_name=None):
        print('INFO: Loading data...')
        if filter_dataset_name:
            print(f"      Filtering for dataset name: '{filter_dataset_name}'")
        self.data = []
        self.categories_in_dataset = set()
        required_keys = ['table', 'question', 'category', 'answer', 'question_id']
        is_mixed_style = filter_dataset_name == 'MIXED_DATASETS'
        if is_mixed_style:
            from project_config.config import cfg
            mixed_datasets = cfg['DATA']['MIXED_DATASETS']
            samples_per_dataset = cfg['DATA']['SAMPLES_PER_DATASET']
            print(f'      Mixed dataset sampling: {mixed_datasets}, {samples_per_dataset} samples each')
            dataset_samples = {dataset: [] for dataset in mixed_datasets}
        try:
            # Load JSONL data file
            with open(data_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    try:
                        item = json.loads(line)
                        if not all((k in item for k in required_keys)):
                            missing = [k for k in required_keys if k not in item]
                            print(f"WARN: Skipping line #{line_idx + 1} due to missing keys: {missing}.")
                            continue
                        current_category = item['category']
                        self.categories_in_dataset.add(current_category)
                        if is_mixed_style:
                            dataset_prefix = current_category.split('_')[0] if '_' in current_category else current_category
                            if dataset_prefix in mixed_datasets:
                                if len(dataset_samples[dataset_prefix]) < samples_per_dataset:
                                    dataset_samples[dataset_prefix].append(item)
                        elif filter_dataset_name:
                            dataset_prefix = current_category.split('_')[0] if '_' in current_category else current_category
                            if dataset_prefix == filter_dataset_name:
                                self.data.append(item)
                        else:
                            self.data.append(item)
                    except json.JSONDecodeError:
                        print(f'WARN: Skipping malformed JSON line #{line_idx + 1}')
        except FileNotFoundError:
            print('ERROR: Data file not found')
        if is_mixed_style:
            import random
            random.seed(42)
            for dataset_name, samples in dataset_samples.items():
                if len(samples) > samples_per_dataset:
                    samples = random.sample(samples, samples_per_dataset)
                self.data.extend(samples)
                print(f'      {dataset_name}: {len(samples)} samples')
            random.shuffle(self.data)
            print(f'      Total mixed training samples: {len(self.data)}')
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
        self.max_seq_len_target = max_seq_len_target
        self.table_image_dir = table_image_dir
        if is_mixed_style:
            print(f'INFO: Mixed-style TabularDataset initialized with {len(self.data)} samples from {len(mixed_datasets)} datasets.')
        elif filter_dataset_name:
            print(f"INFO: TabularDataset initialized with {len(self.data)} samples for dataset '{filter_dataset_name}'.")
            if len(self.data) < 10:
                print(f'DEBUG: Categories found in dataset: {self.categories_in_dataset}')
                print(f"DEBUG: Using modified prefix-matching (before '_')")
        else:
            print(f'INFO: TabularDataset initialized with {len(self.data)} samples (no filter).')
        if not self.data and data_path:
            print(f"ERROR: No data loaded (filter='{filter_dataset_name}').")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            table_data = item.get('table', {})
            question = item.get('question', '')
            category = item.get('category', 'WTQ_for_TQA')
            question_id = item.get('question_id', f'unknown_{idx}')
            target_text = item.get('answer', '')
            image_file_name = item.get('image', None)
            if 'InfoTabs' in category:
                md_table_for_gpt2 = _get_markdown_table_infotabs(table_data, token_limit=3800)
            else:
                md_table_for_gpt2 = _get_markdown_table(table_data, token_limit=3800)
            prompt_template_gpt2 = markdown_template_mapping.get(category, DEFAULT_MARKDOWN_PROMPT_BODY)
            try:
                if '{md_table}' in prompt_template_gpt2:
                    prompt_for_text_expert = prompt_template_gpt2.format(md_table=md_table_for_gpt2, question=question)
                else:
                    prompt_body_gpt2 = prompt_template_gpt2.format(question=question)
                    prompt_for_text_expert = f'Markdown Table:\n{md_table_for_gpt2}\n\n{prompt_body_gpt2}'
            except KeyError as e:
                print(f"WARN: Prompt formatting failed for category '{category}' with key {e}. Falling back to basic prompt.")
                prompt_for_text_expert = f'Markdown Table:\n{md_table_for_gpt2}\n\n{question}'
            prompt_tpl_llava = image_template_mapping.get(category, DEFAULT_IMAGE_PROMPT_TEMPLATE)
            try:
                prompt_for_vlm_expert = prompt_tpl_llava.format(question=question)
            except KeyError:
                prompt_for_vlm_expert = question
            image_path_for_vlm = None
            if image_file_name and self.table_image_dir:
                potential_path = os.path.join(self.table_image_dir, image_file_name)
                if not os.path.exists(potential_path):
                    global skipped_data_log
                    if 'skipped_data_log' in globals():
                        skipped_data_log.append({'question_id': question_id, 'image': image_file_name, 'image_path': potential_path, 'error_message': f'Image file not found; question: {question}'})
                    print(f'WARN: Skipping sample with missing image. question_id={question_id}, image={image_file_name}')
                    return None
                else:
                    image_path_for_vlm = potential_path
            target_text_original = target_text
            if isinstance(target_text, list):
                target_text_for_tokenize = json.dumps({'answer': target_text})
            else:
                target_text_for_tokenize = str(target_text)
            target_text_with_eos = target_text_for_tokenize + self.target_tokenizer.eos_token if self.target_tokenizer.eos_token else target_text_for_tokenize
            tokenized_target = self.target_tokenizer(target_text_with_eos, padding='max_length', truncation=True, max_length=self.max_seq_len_target, return_tensors='pt')
            target_ids = tokenized_target.input_ids.squeeze(0)
            target_attention_mask = tokenized_target.attention_mask.squeeze(0)
            return {'raw_table': table_data, 'question': question, 'prompt_for_text_expert': prompt_for_text_expert, 'prompt_for_vlm_expert': prompt_for_vlm_expert, 'image_path_for_vlm': image_path_for_vlm, 'category': category, 'target_text_str': target_text_original, 'target_ids': target_ids, 'target_attention_mask': target_attention_mask, 'question_id': question_id, 'image': image_file_name}
        except Exception as e:
            print(f'ERROR in __getitem__ at idx {idx}: {e}')
            return None

def create_data_loader(data_path, batch_size, shuffle, num_workers, target_tokenizer_name, max_seq_len_target, table_image_dir, filter_dataset_name=None, return_dataset=False):
    if not data_path:
        return None
    if not os.path.exists(data_path):
        print("ERROR: Data path not found. Cannot create DataLoader.")
        return None
    dataset = TabularDataset(data_path=data_path, target_tokenizer_name=target_tokenizer_name, max_seq_len_target=max_seq_len_target, table_image_dir=table_image_dir, filter_dataset_name=filter_dataset_name)
    if not dataset or len(dataset) == 0:
        print(f"ERROR: Dataset empty (filter='{filter_dataset_name}'). Cannot create DataLoader.")
        return None
    if return_dataset:
        return dataset

    def collate_fn(batch_list):
        valid_batch = [item for item in batch_list if item is not None]
        if not valid_batch:
            return None
        batched = {}
        keys_to_list = ['raw_table', 'question', 'prompt_for_text_expert', 'prompt_for_vlm_expert', 'image_path_for_vlm', 'category', 'target_text_str', 'question_id', 'image']
        keys_to_stack = ['target_ids', 'target_attention_mask']
        for key in keys_to_list:
            batched[key] = [item[key] for item in valid_batch]
        for key in keys_to_stack:
            batched[key] = torch.stack([item[key] for item in valid_batch])
        return batched
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available() and num_workers > 0)
    return dataloader

def _get_markdown_table(table, token_limit):
    if not table or not isinstance(table, dict) or 'header' not in table or ('rows' not in table):
        return ''
    header = table['header']
    rows = table.get('rows', []).copy()
    if not header:
        return pd.DataFrame(columns=header).to_markdown(index=False) if isinstance(header, list) else ''
    if not rows or not isinstance(rows, list):
        rows = []
    clean_rows = [r for r in rows if isinstance(r, list) and len(r) == len(header)]
    try:
        df = pd.DataFrame(clean_rows, columns=header)
        markdown_table = df.to_markdown(index=False)
        if len(markdown_table) > token_limit * 5:
            markdown_table = markdown_table[:token_limit * 5] + '\n...[truncated]'
        return markdown_table
    except Exception as e:
        return f'[Error formatting table: {e}]'

def _get_markdown_table_infotabs(table, token_limit):
    if not isinstance(table, dict):
        return ''
    try:
        valid_keys = [k for k in table.keys() if isinstance(k, str)]
        values = ['\n'.join(map(str, table[k])) if isinstance(table[k], list) else str(table[k]) for k in valid_keys]
        df_vertical = pd.DataFrame({'Attribute': valid_keys, 'Value': values})
        markdown_table = df_vertical.to_markdown(index=False)
        if len(markdown_table) > token_limit * 5:
            markdown_table = markdown_table[:token_limit * 5] + '\n...[truncated]'
        return markdown_table
    except Exception as e:
        return f'[Error formatting infotabs table: {e}]'

def create_dummy_data(file_path, num_samples=20):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    categories = list(markdown_template_mapping.keys())
    dummy_samples = []
    for i in range(num_samples):
        cat = categories[i % len(categories)] if categories else 'WTQ_for_TQA'
        image_filename = f'table_image_dummy_{i}.png'
        sample = {'question_id': f'dummy_{i}', 'table': {'header': [f'Col1_ID{i}', f'Col2_Val{i}'], 'rows': [[f'R1C1_{i}', i * 100], [f'R2C1_{i}', (i + 1) * 100]]}, 'question': f'What is the value associated with R1C1_{i} in table {i}?', 'category': cat, 'answer': [str(i * 100)], 'image': image_filename}
        dummy_samples.append(sample)
    with open(file_path, 'w') as f:
        for sample in dummy_samples:
            f.write(json.dumps(sample) + '\n')
    print(f'Created dummy data at {file_path}')

def create_dummy_images(image_dir, num_images=20):
    os.makedirs(image_dir, exist_ok=True)
    for i in range(num_images):
        img_path = os.path.join(image_dir, f'table_image_dummy_{i}.png')
        if not os.path.exists(img_path):
            try:
                img = Image.new('RGB', (300, 100), color=(128, 128, 128))
                img.save(img_path)
            except Exception as e:
                print(f'Could not create dummy image {img_path}: {e}')
