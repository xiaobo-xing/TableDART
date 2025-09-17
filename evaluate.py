import torch
from tqdm import tqdm
import os
import argparse
import json
import re
from collections import defaultdict
from torchmetrics.text import BLEUScore, ROUGEScore
from transformers import AutoTokenizer
from rouge_score import rouge_scorer as py_rouge_scorer
import numpy as np
from models.dynamic_selector import DynamicExpertSelector
from data.dataloader import create_data_loader, create_dummy_data, create_dummy_images
from project_config.config import cfg
from utils.helpers import extract_tqa_answer_list, _normalize


def normalize_answer(s):
    def remove_articles(text):
        return re.sub("\\b(a|an|the)\\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return "".join((ch for ch in text if ch not in exclude))

    def lower(text):
        return text.lower()

    if not isinstance(s, str):
        if isinstance(s, list):
            try:
                s = sorted([str(x) for x in s])
            except Exception:
                pass
            s = str(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def flexible_accuracy_score(prediction, ground_truth):
    pred_list = extract_tqa_answer_list(prediction)
    pred_set = {_normalize(str(p)) for p in pred_list}
    if isinstance(ground_truth, str):
        gold_list = [ground_truth]
    elif isinstance(ground_truth, list):
        gold_list = ground_truth
    else:
        gold_list = [str(ground_truth)]
    processed_gold_list = []
    for item in gold_list:
        processed_gold_list.extend(str(item).split("|"))
    gold_set = {_normalize(g) for g in processed_gold_list}
    return 1.0 if gold_set == pred_set else 0.0


def calculate_accuracy(predictions, ground_truths):
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return 0.0
    correct_count = sum(
        (flexible_accuracy_score(p, gt) for p, gt in zip(predictions, ground_truths))
    )
    return correct_count / len(predictions)


def _normalize_answer_robustly(s: str) -> str:
    s = str(s).lower().strip()
    comma_number_pattern = "[-+]?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?"
    comma_numbers = re.findall(comma_number_pattern, s)
    if comma_numbers:
        for comma_num in comma_numbers:
            cleaned_num = comma_num.replace(",", "")
            if cleaned_num.endswith(".0"):
                cleaned_num = cleaned_num[:-2]
            s = s.replace(comma_num, cleaned_num)
    numbers = re.findall("[-+]?\\d*\\.\\d+|\\d+", s)
    if len(numbers) == 1:
        non_numeric_part = re.sub("[-+]?\\d*\\.\\d+|\\d+", "", s).strip()
        if len(non_numeric_part.split()) <= 1:
            cleaned_num = numbers[0]
            if cleaned_num.endswith(".0"):
                cleaned_num = cleaned_num[:-2]
            return cleaned_num
    s = re.sub("\\b(a|an|the)\\b", " ", s)
    s = "".join((ch for ch in s if ch not in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{}~"))
    s = " ".join(s.split())
    return s


def _fallback_containment_check(model_output: str, processed_gold_list: list) -> bool:
    normalized_model_output = _normalize_answer_robustly(model_output)
    for gold_item in processed_gold_list:
        normalized_gold_item = _normalize_answer_robustly(gold_item)
        if re.fullmatch("[-+]?\\d*\\.\\d+|\\d+", normalized_gold_item):
            if not re.search(
                f"\\b{re.escape(normalized_gold_item)}\\b", normalized_model_output
            ):
                return False
        elif normalized_gold_item not in normalized_model_output:
            return False
    return True


def calculate_tqa_accuracy_with_fallback(pred_raw, pred_list, gold_raw_list):
    processed_gold_list = []
    for gold in gold_raw_list:
        processed_gold_list.extend(str(gold).split("|"))
    pred_set = {_normalize_answer_robustly(p) for p in pred_list}
    gold_set = {_normalize_answer_robustly(g) for g in processed_gold_list}
    is_correct = gold_set == pred_set
    if not is_correct:
        is_correct = _fallback_containment_check(pred_raw, processed_gold_list)
    return 1.0 if is_correct else 0.0


def _extract_first_sentence_for_fetaqa_evaluation(text):
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        import os

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nltk_data_path = os.path.join(project_root, "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            print("INFO: Downloading NLTK punkt tokenizer")
            nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
        text = text.strip()
        if not text:
            return text
        sentences = sent_tokenize(text)
        if sentences and sentences[0].strip():
            return sentences[0].strip()
        else:
            return text
    except ImportError:
        try:
            import spacy

            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = spacy.blank("en")
                nlp.add_pipe("sentencizer")
            text = text.strip()
            if not text:
                return text
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            if sentences and sentences[0]:
                return sentences[0]
            else:
                return text
        except ImportError:
            text = text.strip()
            if not text:
                return text
            import re

            abbreviations = [
                "Mr",
                "Mrs",
                "Ms",
                "Dr",
                "Prof",
                "Jr",
                "Sr",
                "vs",
                "etc",
                "Inc",
                "Ltd",
                "Co",
            ]
            temp_text = text
            for abbr in abbreviations:
                temp_text = re.sub(
                    f"\\b{abbr}\\.", f"{abbr}__DOT__", temp_text, flags=re.IGNORECASE
                )
            sentences = re.split("[.!?]+\\s+", temp_text)
            if sentences and sentences[0].strip():
                first_sentence = sentences[0].replace("__DOT__", ".")
                if first_sentence and (not first_sentence[-1] in ".!?"):
                    first_sentence += "."
                return first_sentence.strip()
            return text


def _extract_fetaqa_answer_for_evaluation(model_output, selected_expert):
    if not isinstance(model_output, str) or not model_output:
        return ""
    cleaned_output = model_output.strip()
    if selected_expert == "TextExpert":
        return _extract_first_sentence_for_fetaqa_evaluation(cleaned_output)
    else:
        return cleaned_output


def evaluate(
    model, dataloader, device, generation_config, epoch_num=None, silent=False
):
    model.eval()
    path_counts = defaultdict(int)
    benchmark_data = defaultdict(list)
    bleu = BLEUScore(n_gram=4).to(device)
    desc = (
        f"Evaluating (Epoch {epoch_num})"
        if epoch_num is not None and (not silent)
        else "Evaluating"
    )
    progress_bar = tqdm(dataloader, desc=desc, leave=False, disable=silent)
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            if batch is None:
                continue
            output_dict = model(
                raw_table_batch=batch["raw_table"],
                original_questions_batch=batch["question"],
                prompts_text_batch=batch["prompt_for_text_expert"],
                prompts_vlm_batch=batch["prompt_for_vlm_expert"],
                image_paths_vlm_batch=batch["image_path_for_vlm"],
                target_texts_str_batch=batch["target_text_str"],
                categories_batch=batch["category"],
                is_training=False,
                current_epoch_for_freezing_logic=999,
                **generation_config,
            )
            raw_predictions = output_dict["generated_text"]
            selected_indices = output_dict.get("selected_indices", [])
            selected_paths_list = (
                [model.path_names[i] for i in selected_indices]
                if selected_indices
                else ["N/A"] * len(batch["question"])
            )
            for j in range(len(batch["question"])):
                category = batch["category"][j]
                benchmark_name = category.split("_")[0]
                benchmark_data[benchmark_name].append(
                    {
                        "question": batch["question"][j],
                        "pred_raw": raw_predictions[j],
                        "pred_list": extract_tqa_answer_list(raw_predictions[j]),
                        "gold_raw_list": batch["target_text_str"][j],
                        "selected_expert": selected_paths_list[j],
                    }
                )
                path_counts[selected_paths_list[j]] += 1
    results = {}
    tqa_total_correct = 0
    tqa_total_samples = 0
    if not silent:
        print("\n--- Evaluation Summary ---")
    for benchmark, items in benchmark_data.items():
        bench_results = {}
        if benchmark in ["WTQ", "TAT-QA", "TabFact", "HiTab", "TABMWP", "InfoTabs"]:
            correct_count = 0
            for item in items:
                accuracy = calculate_tqa_accuracy_with_fallback(
                    item["pred_raw"], item["pred_list"], item["gold_raw_list"]
                )
                if accuracy == 1.0:
                    correct_count += 1
                elif not silent:
                    print("\n--- Incorrect Prediction ---")
                    print(f"  Benchmark     : {benchmark}")
                    print(f"  Question      : {item['question']}")
                    print(f"  SelectedExpert: {item['selected_expert']}")
                    print(f"  Gold Answer   : {item['gold_raw_list']}")
                    print(f"  Model Output  : {item['pred_raw']}")
                    print(f"  Parsed Answer : {item['pred_list']}")
                    print("--------------------------")
            total_count = len(items)
            bench_accuracy = correct_count / total_count if total_count > 0 else 0
            bench_results = {"Accuracy": bench_accuracy, "count": total_count}
            tqa_total_correct += correct_count
            tqa_total_samples += total_count
            if not silent:
                print(
                    f"  Benchmark: {benchmark} | Accuracy: {bench_accuracy:.4f} (n={total_count})"
                )
        elif benchmark == "FeTaQA":
            bleu_scorer = BLEUScore(n_gram=4)
            preds_for_bleu = []
            for item in items:
                extracted_answer = _extract_fetaqa_answer_for_evaluation(
                    item["pred_raw"], item["selected_expert"]
                )
                preds_for_bleu.append(extracted_answer)
            golds_for_bleu = [
                [", ".join(map(str, item["gold_raw_list"]))] for item in items
            ]
            try:
                bleu_score = bleu_scorer(preds_for_bleu, golds_for_bleu).item()
                bench_results = {"BLEU-4": bleu_score, "count": len(items)}
                if not silent:
                    print(
                        f"  Benchmark: {benchmark} | BLEU-4: {bleu_score:.4f} (n={len(items)})"
                    )
            except Exception as e:
                if not silent:
                    print(f"ERROR calculating BLEU for {benchmark}: {e}")
        if bench_results:
            results[benchmark] = bench_results
    overall_accuracy = (
        tqa_total_correct / tqa_total_samples if tqa_total_samples > 0 else 0
    )
    results["Overall_Accuracy"] = overall_accuracy
    results["Total_Samples_Evaluated"] = sum((len(i) for i in benchmark_data.values()))
    if path_counts:
        results["Path_Selection_Counts"] = path_counts
        if not silent:
            print("\nPath Selection Counts:")
            total_selections = sum(path_counts.values())
            for path, count in sorted(path_counts.items()):
                percentage = (
                    count / total_selections * 100 if total_selections > 0 else 0
                )
                print(f"  {path}: {count} samples ({percentage:.2f}%)")
    return results


def main(args):
    device = torch.device(cfg["TRAINING"]["DEVICE"])
    print(f"Using device: {device}")
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        print("ERROR: Checkpoint not found")
        return
    print("Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    target_tokenizer_name_for_dataloader = cfg["MODEL"]["TEXT_EXPERT_ID"]
    try:
        _ = AutoTokenizer.from_pretrained(target_tokenizer_name_for_dataloader)
    except:
        target_tokenizer_name_for_dataloader = "gpt2"
    image_directory_for_eval = cfg["DATA"].get(
        "TEST_TABLE_IMAGE_DIR", cfg["DATA"]["TABLE_IMAGE_DIR"]
    )
    test_loader = create_data_loader(
        cfg["DATA"]["TEST_PATH"],
        batch_size=cfg["EVALUATION"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
        target_tokenizer_name=target_tokenizer_name_for_dataloader,
        max_seq_len_target=cfg["DATA"]["MAX_SEQ_LEN_LM_TARGET"],
        table_image_dir=image_directory_for_eval,
    )
    if not test_loader:
        print("ERROR: Test DataLoader could not be created. Exiting.")
        return
    loaded_gate_input_dim = checkpoint.get("gate_input_dim")
    calculated_gate_input_dim = None
    model = DynamicExpertSelector(
        gate_hidden_dim=cfg["MODEL"]["GATE_HIDDEN_DIM"],
        use_late_fusion=cfg["MODEL"]["USE_LATE_FUSION"],
    )
    calculated_gate_input_dim = model.gate_input_dim
    if loaded_gate_input_dim and loaded_gate_input_dim != calculated_gate_input_dim:
        print(
            f"WARN: Gate input dim mismatch! Checkpoint expects {loaded_gate_input_dim}, current model calculates {calculated_gate_input_dim}. State dict loading might fail or be incorrect for the gate."
        )
    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing_keys:
            print("WARN: Missing keys during state_dict loading:", missing_keys)
        if unexpected_keys:
            print("WARN: Unexpected keys during state_dict loading:", unexpected_keys)
        print(
            f"Checkpoint loaded successfully (trained for {checkpoint.get('epoch', '?')} epochs)."
        )
    except Exception as e:
        print(f"ERROR loading state_dict: {e}. Evaluation cannot proceed.")
        return
    model.to(device)
    results = evaluate(
        model,
        test_loader,
        device,
        cfg["GENERATION"],
        epoch_num=checkpoint.get("epoch", -1) - 1,
    )
    print("\nEvaluation complete.")
    results_dir = os.path.join(
        os.path.dirname(args.checkpoint_path), "evaluation_results"
    )
    os.makedirs(results_dir, exist_ok=True)
    results_filename = f"eval_results_{os.path.basename(args.checkpoint_path).replace('.pth', '')}.json"
    if "predictions" in results:
        del results["predictions"]
    if "targets" in results:
        del results["targets"]
    with open(os.path.join(results_dir, results_filename), "w") as f:
        json.dump(results, f, indent=4)
    print("Evaluation results saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Dynamic Tabular Expert Selector"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.join(cfg["TRAINING"]["CHECKPOINT_DIR"], "best_model_gate.pth"),
        help="Path to model checkpoint.",
    )
    args = parser.parse_args()
    main(args)
