import os
import torch
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, ".env")
loaded_env = load_dotenv(dotenv_path=dotenv_path)
print(f"INFO (inference.py): .env loaded: {loaded_env}")
from models.dynamic_selector import DynamicExpertSelector
from project_config.config import cfg
from data.dataloader import create_data_loader
from evaluate import evaluate
from utils.efficiency_utils import InferenceEfficiencyMonitor


def parse_args():
    parser = argparse.ArgumentParser(description="TableGPT2Expert Model Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint, or 'None' to use random gating network (default: use from config)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data (default: use TEST_PATH from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save inference results (default: use from config)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Inference batch size"
    )
    parser.add_argument(
        "--dataset_filter",
        type=str,
        default=None,
        help="Filter test data by dataset name (e.g., 'FeTaQA')",
    )
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        default=False,
        help="Compute evaluation metrics after inference (default: False)",
    )
    parser.add_argument(
        "--measure_efficiency",
        action="store_true",
        help="Measure and report inference efficiency (latency, TTFT, tokens/sec)",
    )
    parser.add_argument(
        "--save_details",
        action="store_true",
        help="Save detailed outputs including all expert answers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (default: from config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or cfg["TRAINING"]["DEVICE"]
    device = torch.device(device)
    print(f"Using device: {device}")
    efficiency_monitor = None
    if args.measure_efficiency:
        print("INFO: Efficiency measurement enabled.")
        efficiency_monitor = InferenceEfficiencyMonitor(
            tokenizer_name=cfg["MODEL"]["TEXT_EXPERT_ID"]
        )
    test_path = args.test_data or cfg["DATA"]["TEST_PATH"]
    if not os.path.exists(test_path):
        raise FileNotFoundError("Test data not found")
    print("Using test data")
    output_dir = args.output_dir or cfg["DATA"].get(
        "INFERENCE_OUTPUT_DIR", "./inference_results"
    )
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory prepared")
    checkpoint_path = cfg["TRAINING"].get("INFERENCE_CHECKPOINT", None)
    use_checkpoint = True
    if checkpoint_path is None or checkpoint_path.lower() == "none":
        print("INFO: Checkpoint not provided; using randomly initialized gating network.")
        use_checkpoint = False
        checkpoint_path = None
    elif not os.path.exists(checkpoint_path):
        print("WARN: Checkpoint not found")
        checkpoint_dir = cfg["TRAINING"]["CHECKPOINT_DIR"]
        best_model_path = os.path.join(checkpoint_dir, "best_model_gate.pth")
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
            print("INFO: Using best checkpoint discovered")
        else:
            import glob

            final_models = glob.glob(
                os.path.join(checkpoint_dir, "final_model_epoch*_gate*.pth")
            )
            if final_models:
                checkpoint_path = max(final_models, key=os.path.getctime)
                print("INFO: Using most recent checkpoint discovered")
            else:
                print(
                    f"WARN: No checkpoint found in {checkpoint_dir}; using randomly initialized gating network"
                )
                use_checkpoint = False
                checkpoint_path = None
    if use_checkpoint:
        print("Using checkpoint")
    else:
        print("Using randomly initialized gating network (no checkpoint)")
    print("\n--- Initializing Model ---")
    model = DynamicExpertSelector(
        gate_hidden_dim=cfg["MODEL"]["GATE_HIDDEN_DIM"],
        use_late_fusion=cfg["MODEL"]["USE_LATE_FUSION"],
    )
    if use_checkpoint and checkpoint_path:
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.gating_network.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if "EXPERT_COSTS" in cfg["MODEL"]:
            path_costs_list = [
                cfg["MODEL"]["EXPERT_COSTS"].get("TextExpert", 1.2),
                cfg["MODEL"]["EXPERT_COSTS"].get("VLMExpert", 2.5),
            ]
            if model.use_late_fusion:
                path_costs_list.append(cfg["MODEL"]["EXPERT_COSTS"].get("Fusion", 4.0))
            model.path_costs_tensor = torch.tensor(
                path_costs_list, dtype=torch.float32
            ).to(device)
            print(f"INFO: Using expert costs from config: {path_costs_list}")
        elif "path_costs" in checkpoint:
            model.path_costs_tensor = torch.tensor(
                checkpoint["path_costs"], dtype=torch.float32
            ).to(device)
            print(f"INFO: Loaded path costs from checkpoint: {model.path_costs_tensor.cpu().tolist()}")
        else:
            print(
                f"INFO: No path-cost metadata in config or checkpoint; using model defaults: {model.path_costs_tensor.cpu().tolist()}"
            )
    else:
        print("Using randomly initialized gating network with default expert costs")
        if "EXPERT_COSTS" in cfg["MODEL"]:
            path_costs_list = [
                cfg["MODEL"]["EXPERT_COSTS"].get("TextExpert", 1.2),
                cfg["MODEL"]["EXPERT_COSTS"].get("VLMExpert", 2.5),
            ]
            if model.use_late_fusion:
                path_costs_list.append(cfg["MODEL"]["EXPERT_COSTS"].get("Fusion", 4.0))
            model.path_costs_tensor = torch.tensor(
                path_costs_list, dtype=torch.float32
            ).to(device)
            print(f"INFO: Using expert costs from config: {path_costs_list}")
        else:
            print(f"INFO: Using model default path costs: {model.path_costs_tensor.cpu().tolist()}")
    model = model.to(device)
    model.eval()
    print("\n--- Loading Test Data ---")
    target_tokenizer_name = cfg["MODEL"]["TEXT_EXPERT_ID"]
    test_image_dir = cfg["DATA"].get(
        "TEST_TABLE_IMAGE_DIR", cfg["DATA"]["TABLE_IMAGE_DIR"]
    )
    inference_filters = cfg["DATA"].get("INFERENCE_DATASET_FILTERS", [])
    dataloader_filter = None
    if inference_filters and len(inference_filters) == 1:
        dataloader_filter = inference_filters[0]
        print(f"INFO: Creating DataLoader with config filter: {dataloader_filter}")
    else:
        print(f"INFO: Config filters: {inference_filters}; no DataLoader filter applied")
    test_loader = create_data_loader(
        test_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        target_tokenizer_name=target_tokenizer_name,
        max_seq_len_target=cfg["DATA"]["MAX_SEQ_LEN_LM_TARGET"],
        table_image_dir=test_image_dir,
        filter_dataset_name=dataloader_filter,
    )
    if not test_loader:
        raise ValueError(f"Failed to create DataLoader for test data: {test_path}")
    print(f"Test data loaded with {len(test_loader.dataset)} samples")
    results = []
    skipped_samples = 0
    gate_prob_stats = {"TextExpert": [], "VLMExpert": [], "Fusion": []}
    total_samples = 0
    allowed_datasets = set(cfg["DATA"].get("INFERENCE_DATASET_FILTERS", []))

    def is_allowed_category(cat):
        if not allowed_datasets:
            return True
        prefix = cat.split("_")[0] if "_" in cat else cat
        is_allowed = prefix in allowed_datasets
        return is_allowed

    print("\n--- Running Inference ---")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Inference")
        for i, batch in enumerate(progress_bar):
            if batch is None:
                skipped_samples += 1
                continue
            batch_category = (
                batch["category"][0]
                if isinstance(batch["category"], list)
                else batch["category"]
            )
            if not is_allowed_category(batch_category):
                skipped_samples += 1
                continue
            try:
                generation_kwargs = cfg["GENERATION"].copy()
                current_prompt_for_text_expert = (
                    batch["prompt_for_text_expert"][0]
                    if isinstance(batch["prompt_for_text_expert"], list)
                    else batch["prompt_for_text_expert"]
                )
                if efficiency_monitor:
                    generation_kwargs["streamer"] = efficiency_monitor.streamer
                    efficiency_monitor.start_batch()
                output_dict = model(
                    raw_table_batch=batch["raw_table"],
                    original_questions_batch=batch["question"],
                    prompts_text_batch=current_prompt_for_text_expert,
                    prompts_vlm_batch=batch["prompt_for_vlm_expert"],
                    image_paths_vlm_batch=batch["image_path_for_vlm"],
                    target_texts_str_batch=batch["target_text_str"],
                    categories_batch=batch["category"],
                    is_training=False,
                    **generation_kwargs,
                )
                selected_indices = output_dict["selected_indices"]
                if selected_indices is not None:
                    selected_idx = (
                        selected_indices[0]
                        if isinstance(selected_indices, (list, tuple))
                        else selected_indices
                    )
                else:
                    selected_idx = None
                selected_expert = (
                    ["TextExpert", "VLMExpert", "Fusion"][selected_idx]
                    if selected_idx is not None
                    else "None"
                )
                gate_probs = output_dict.get("gate_probabilities_for_selection")
                if gate_probs is not None:
                    probs = (
                        gate_probs[0].cpu().numpy()
                        if gate_probs.dim() > 1
                        else gate_probs.cpu().numpy()
                    )
                    gate_prob_stats["TextExpert"].append(probs[0])
                    gate_prob_stats["VLMExpert"].append(probs[1])
                    if len(probs) > 2:
                        gate_prob_stats["Fusion"].append(probs[2])
                    else:
                        gate_prob_stats["Fusion"].append(0.0)
                    total_samples += 1
                final_answer = (
                    output_dict["generated_text"][0]
                    if isinstance(output_dict["generated_text"], list)
                    else output_dict["generated_text"]
                )
                if efficiency_monitor:
                    try:
                        ttft_measurement = None
                        if selected_idx is not None:
                            if selected_idx == 0:
                                expert_model = model.expert_text
                                (
                                    _,
                                    intermediate_state,
                                ) = expert_model.extract_gate_features_and_intermediate_state(
                                    [current_prompt_for_text_expert]
                                )
                                single_state = {
                                    k: v[0:1] if isinstance(v, torch.Tensor) else v
                                    for k, v in intermediate_state.items()
                                }
                                ttft_measurement = (
                                    efficiency_monitor.measure_ttft_for_expert(
                                        expert_model, single_state, device
                                    )
                                )
                            elif selected_idx == 1:
                                expert_model = model.expert_vlm
                                vlm_prompt = (
                                    batch["prompt_for_vlm_expert"][0]
                                    if isinstance(batch["prompt_for_vlm_expert"], list)
                                    else batch["prompt_for_vlm_expert"]
                                )
                                vlm_image = (
                                    batch["image_path_for_vlm"][0]
                                    if isinstance(batch["image_path_for_vlm"], list)
                                    else batch["image_path_for_vlm"]
                                )
                                (
                                    _,
                                    intermediate_state,
                                ) = expert_model.extract_gate_features_and_intermediate_state(
                                    [vlm_prompt], [vlm_image]
                                )
                                if hasattr(expert_model, "text_tokenizer"):
                                    single_state = {}
                                    for key, value in intermediate_state.items():
                                        if isinstance(value, list) and len(value) > 0:
                                            single_state[key] = value[0]
                                        else:
                                            single_state[key] = value
                                else:
                                    single_state = {
                                        k: v[0:1] if isinstance(v, torch.Tensor) else v
                                        for k, v in intermediate_state.items()
                                    }
                                ttft_measurement = (
                                    efficiency_monitor.measure_ttft_for_expert(
                                        expert_model, single_state, device
                                    )
                                )
                        if ttft_measurement is not None:
                            efficiency_monitor._current_ttft_override = ttft_measurement
                    except Exception as e:
                        print(f"WARN: Expert TTFT measurement failed: {e}")
                    efficiency_monitor.end_batch(final_answer)
                actual_question_id_val_from_batch = batch.get("question_id")
                if isinstance(actual_question_id_val_from_batch, list):
                    if actual_question_id_val_from_batch:
                        current_id = actual_question_id_val_from_batch[0]
                        if current_id is None:
                            actual_question_id = "unknown_id"
                        else:
                            actual_question_id = str(current_id)
                    else:
                        actual_question_id = "unknown_id"
                elif actual_question_id_val_from_batch is None:
                    actual_question_id = "unknown_id"
                else:
                    actual_question_id = str(actual_question_id_val_from_batch)
                actual_question_text_val_from_batch = batch.get("question")
                if isinstance(actual_question_text_val_from_batch, list):
                    if actual_question_text_val_from_batch:
                        actual_question_text = str(
                            actual_question_text_val_from_batch[0]
                        )
                    else:
                        actual_question_text = "unknown_question"
                elif actual_question_text_val_from_batch is None:
                    actual_question_text = "unknown_question"
                else:
                    actual_question_text = str(actual_question_text_val_from_batch)
                result = {
                    "question_id": actual_question_id,
                    "category": batch["category"][0]
                    if isinstance(batch.get("category"), list)
                    else batch.get("category", "unknown_category"),
                    "question": actual_question_text,
                    "target_answer": batch["target_text_str"][0]
                    if isinstance(batch.get("target_text_str"), list)
                    else batch.get("target_text_str", "N/A"),
                    "output": final_answer,
                    "selected_expert": selected_expert,
                }
                if args.save_details:
                    result.update(
                        {
                            "text_expert_answer": output_dict.get(
                                "text_answer_str", [""]
                            )[0]
                            if isinstance(output_dict.get("text_answer_str", ""), list)
                            else output_dict.get("text_answer_str", ""),
                            "vlm_expert_answer": output_dict.get(
                                "vlm_answer_str", [""]
                            )[0]
                            if isinstance(output_dict.get("vlm_answer_str", ""), list)
                            else output_dict.get("vlm_answer_str", ""),
                            "fusion_answer": output_dict.get("fused_answer_str", [""])[
                                0
                            ]
                            if isinstance(output_dict.get("fused_answer_str", ""), list)
                            else output_dict.get("fused_answer_str", ""),
                            "gate_probs": output_dict.get(
                                "gate_probabilities_for_selection", [0, 0, 0]
                            )[0].tolist()
                            if output_dict.get("gate_probabilities_for_selection")
                            is not None
                            else [0, 0, 0],
                        }
                    )
                if efficiency_monitor:
                    result.update(efficiency_monitor.get_last_instance_metrics())
                results.append(result)
                if i % 10 == 0:
                    progress_bar.set_postfix(
                        {
                            "Processed": len(results),
                            "Skipped": skipped_samples,
                            "Selected": selected_expert,
                        }
                    )
            except Exception as e:
                print(f"\nERROR during inference on batch {i}: {e}")
                import traceback

                traceback.print_exc()
                continue
    if gate_prob_stats["TextExpert"]:
        avg_text = sum(gate_prob_stats["TextExpert"]) / len(
            gate_prob_stats["TextExpert"]
        )
        avg_vlm = sum(gate_prob_stats["VLMExpert"]) / len(gate_prob_stats["VLMExpert"])
        avg_fusion = (
            sum(gate_prob_stats["Fusion"]) / len(gate_prob_stats["Fusion"])
            if gate_prob_stats["Fusion"]
            else 0.0
        )
        print(
            f"\nGate Network Statistics: TextExpert={avg_text:.3f}, VLMExpert={avg_vlm:.3f}, Fusion={avg_fusion:.3f}"
        )
    results_file_jsonl = os.path.join(output_dir, "inference_results.jsonl")
    with open(results_file_jsonl, "w") as f_jsonl:
        for item in results:
            out_item = {
                "question_id": item.get("question_id", ""),
                "category": item.get("category", ""),
                "question": item.get("question", ""),
                "output": item.get("output", ""),
                "selected_expert": item.get("selected_expert", ""),
            }
            if efficiency_monitor:
                efficiency_fields = [
                    "latency",
                    "ttft",
                    "generated_tokens",
                    "tokens_per_second",
                    "comprehensive_cost",
                ]
                for field in efficiency_fields:
                    if field in item:
                        out_item[field] = item[field]
            f_jsonl.write(json.dumps(out_item, ensure_ascii=False) + "\n")
    print("Results (jsonl) saved")
    summary_data = []
    for r in results:
        summary_data.append(
            {
                "question_id": r["question_id"],
                "category": r["category"],
                "question": r["question"][:50] + "..."
                if len(r["question"]) > 50
                else r["question"],
                "output": r["output"][:50] + "..."
                if len(r["output"]) > 50
                else r["output"],
                "selected_expert": r["selected_expert"],
            }
        )
    summary_df = pd.DataFrame(summary_data)
    if efficiency_monitor and results:
        efficiency_cols = pd.DataFrame(
            [
                {
                    "latency(s)": r.get("latency", 0),
                    "TTFT(s)": r.get("ttft", 0),
                    "gen_tokens": r.get("generated_tokens", 0),
                    "tokens_per_sec": r.get("tokens_per_second", 0),
                    "comprehensive_cost": r.get("comprehensive_cost", 0),
                }
                for r in results
            ]
        )
        summary_df = summary_df.reset_index(drop=True)
        efficiency_cols = efficiency_cols.reset_index(drop=True)
        summary_df = pd.concat([summary_df, efficiency_cols], axis=1)
    summary_file = os.path.join(output_dir, "results_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print("Summary saved")
    if args.compute_metrics:
        print("\n--- Computing Evaluation Metrics ---")
        metrics_results = evaluate(model, test_loader, device, cfg["GENERATION"])
        metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_file, "w") as f:
            serializable_metrics = {}
            for k, v in metrics_results.items():
                if isinstance(v, torch.Tensor):
                    serializable_metrics[k] = v.cpu().tolist()
                elif isinstance(v, dict):
                    serializable_metrics[k] = {
                        sk: sv.cpu().tolist() if isinstance(sv, torch.Tensor) else sv
                        for sk, sv in v.items()
                    }
                else:
                    serializable_metrics[k] = v
            json.dump(serializable_metrics, f, indent=2)
        print("Evaluation metrics saved")
        print("\nKey Metrics:")
        general_bleu_4 = serializable_metrics.get("General_BLEU-4", "N/A")
        if isinstance(general_bleu_4, float):
            print(f"General BLEU-4: {general_bleu_4:.4f}")
        else:
            print(f"General BLEU-4: {general_bleu_4}")
        general_rouge_scores = serializable_metrics.get("General_ROUGE", {})
        rouge_l_f_score = general_rouge_scores.get("rougeL-F", "N/A")
        if isinstance(rouge_l_f_score, float):
            print(f"General ROUGE-L F-score: {rouge_l_f_score:.4f}")
        else:
            print(f"General ROUGE-L F-score: {rouge_l_f_score}")
    print("\n--- Inference Complete ---")
    print(f"Processed {len(results)} samples (skipped {skipped_samples})")
    print(f"Expert selection statistics:")
    expert_counts = {"TextExpert": 0, "VLMExpert": 0, "Fusion": 0, "None": 0}
    for r in results:
        expert_counts[r["selected_expert"]] = (
            expert_counts.get(r["selected_expert"], 0) + 1
        )
    for expert, count in expert_counts.items():
        percentage = count / len(results) * 100 if results else 0
        print(f"  {expert}: {count} ({percentage:.1f}%)")
    if efficiency_monitor:
        print(efficiency_monitor.get_summary_report())


if __name__ == "__main__":
    main()
