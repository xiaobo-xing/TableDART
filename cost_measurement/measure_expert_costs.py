import os
import torch
import json
import random
import argparse
import time
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import psutil
import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)
print("INFO: Loading .env")
from models.experts import (
    TableGPT2Expert,
    TableLLavaExpert,
    LateFusionExpert,
    Ovis2Expert,
)
from project_config.config import cfg
from data.dataloader import create_data_loader
from models.prompt.markdown_prompt import (
    markdown_template_mapping,
    DEFAULT_MARKDOWN_PROMPT_BODY,
)
from models.prompt.image_prompt import (
    image_template_mapping,
    DEFAULT_IMAGE_PROMPT_TEMPLATE,
)
import torch.nn.functional as F
import numpy as np


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


def measure_memory_footprint(expert_model, state_fn, device, num_samples=3):
    return {"gpu_memory_gb": 0.0, "cpu_memory_gb": 0.0, "total_memory_gb": 0.0}


def count_tokens(text, tokenizer=None):
    if tokenizer is not None:
        tokens = tokenizer(text, return_tensors="pt").input_ids
        return tokens.numel()
    else:
        return len(text.split())


def measure_expert_latency_with_ttft_and_memory(
    expert_model, state_fn, device, num_warmup=2, num_repeats=3
):
    for _ in range(num_warmup):
        try:
            intermediate_state = state_fn()
            if intermediate_state is None:
                continue
            _ = expert_model.continue_generation_from_state(
                intermediate_state, **cfg["GENERATION"]
            )
        except Exception as e:
            print(f"Warmup error ({type(expert_model).__name__}): {e}")
    latencies = []
    ttfts = []
    token_counts = []
    tokens_per_second = []
    for _ in range(num_repeats):
        try:
            intermediate_state = state_fn()
            if intermediate_state is None:
                continue
            ttft_measured = False
            try:
                ttft = measure_ttft_precisely(expert_model, intermediate_state, device)
                ttft_measured = True
            except Exception as e:
                print(f"Precise TTFT measurement failed: {e}")
                ttft = float("nan")
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            output = expert_model.continue_generation_from_state(
                intermediate_state, **cfg["GENERATION"]
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_latency = end_time - start_time
            if isinstance(output, list):
                output_text = output[0]
            else:
                output_text = output
            tokenizer = None
            if hasattr(expert_model, "tokenizer"):
                tokenizer = expert_model.tokenizer
            elif hasattr(expert_model, "text_tokenizer"):
                tokenizer = expert_model.text_tokenizer
            num_tokens = count_tokens(output_text, tokenizer)
            if num_tokens > 0 and total_latency > 1e-06:
                tps = num_tokens / total_latency
            else:
                tps = 0.0
            latencies.append(total_latency)
            ttfts.append(ttft)
            token_counts.append(num_tokens)
            tokens_per_second.append(tps)
            if not ttft_measured:
                print(
                    f"{type(expert_model).__name__}: TTFT measurement failed (total latency: {total_latency:.4f}s)"
                )
        except Exception as e:
            print(f"[ERROR] {type(expert_model).__name__} generation failed: {e}")
    memory_footprint = measure_memory_footprint(
        expert_model, state_fn, device, num_samples=3
    )
    if not latencies:
        return (10.0, float("nan"), 0.0, memory_footprint)
    avg_latency = sum(latencies) / len(latencies)
    valid_ttfts = [t for t in ttfts if not np.isnan(t)]
    avg_ttft = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else float("nan")
    avg_tps = (
        sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0.0
    )
    success_rate = len(valid_ttfts) / len(ttfts) * 100 if ttfts else 0
    if success_rate < 100:
        print(
            f"WARN: {type(expert_model).__name__}: TTFT measurement success rate {success_rate:.1f}% ({len(valid_ttfts)}/{len(ttfts)})"
        )
    return (avg_latency, avg_ttft, avg_tps, memory_footprint)


def measure_ttft_precisely(expert_model, intermediate_state, device):
    print(f"Starting measure_ttft_precisely - model type: {type(expert_model)}")
    print(f"device: {device}")
    print(
        f"intermediate_state keys: {(list(intermediate_state.keys()) if intermediate_state else 'None')}"
    )
    if hasattr(expert_model, "model") and hasattr(expert_model.model, "generate"):
        print(f"Model has generate method, continuing...")
        try:
            base_cfg = cfg["GENERATION"]
            single_token_config = {
                "max_new_tokens": 1,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
                "output_attentions": False,
                "use_cache": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 1,
            }
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            if hasattr(expert_model, "model") and "TableGPT2" in str(
                type(expert_model)
            ):
                with torch.no_grad():
                    _ = expert_model.model.generate(
                        inputs_embeds=intermediate_state["inputs_embeds"],
                        attention_mask=intermediate_state["attention_mask"],
                        **single_token_config,
                    )
            elif "Ovis2" in str(type(expert_model)):
                for key, value in intermediate_state.items():
                    if value is None:
                        print(f"   {key}: None")
                    elif isinstance(value, list):
                        print(f"   {key}: list with {len(value)} items")
                        for i, item in enumerate(value):
                            if item is None:
                                print(f"     [{i}]: None")
                            elif hasattr(item, "shape"):
                                print(
                                    f"     [{i}]: shape {item.shape}, dtype {item.dtype}"
                                )
                            else:
                                print(f"     [{i}]: {type(item)}")
                    elif hasattr(value, "shape"):
                        print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"   {key}: {type(value)}")
                try:
                    if isinstance(intermediate_state["pixel_values"], list):
                        print(
                        )
                        pixel_values = intermediate_state["pixel_values"][0]
                        input_ids = intermediate_state["input_ids"][0]
                        attention_mask = intermediate_state["attention_mask"][0]
                    else:
                        pixel_values = intermediate_state["pixel_values"]
                        input_ids = intermediate_state["input_ids"]
                        attention_mask = intermediate_state["attention_mask"]
                    if pixel_values is None:
                        print(f"pixel_values is None!")
                        raise ValueError("pixel_values is None")
                    if input_ids is None:
                        print(f"input_ids is None!")
                        raise ValueError("input_ids is None")
                    if attention_mask is None:
                        print(f"attention_mask is None!")
                        raise ValueError("attention_mask is None")
                    print(
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise ValueError(f"Ovis2 state extraction failed: {e}")
                try:
                    if not hasattr(expert_model, "text_tokenizer"):
                        raise ValueError("expert_model has no text_tokenizer attribute")
                    if expert_model.text_tokenizer is None:
                        print(f"expert_model.text_tokenizer is None")
                        raise ValueError("text_tokenizer is None")
                    pad_token_id = expert_model.text_tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = expert_model.text_tokenizer.eos_token_id
                        if pad_token_id is None:
                            pad_token_id = 151643
                    eos_token_id = None
                    if hasattr(expert_model, "model") and hasattr(
                        expert_model.model, "generation_config"
                    ):
                        eos_token_id = expert_model.model.generation_config.eos_token_id
                    if eos_token_id is None:
                        eos_token_id = expert_model.text_tokenizer.eos_token_id
                        if eos_token_id is None:
                            eos_token_id = 151645
                    if isinstance(eos_token_id, list):
                        eos_token_id = eos_token_id[0]
                    print(
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise ValueError(f"Ovis2 tokenizer config failed: {e}")
                try:
                    ovis_single_token_config = {
                        "max_new_tokens": 1,
                        "do_sample": False,
                        "use_cache": True,
                        "eos_token_id": eos_token_id,
                        "pad_token_id": pad_token_id,
                        "output_scores": False,
                        "output_attentions": False,
                        "return_dict_in_generate": False,
                    }
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise ValueError(f"Ovis2 generation config failed: {e}")
                try:
                    print(
                        f"input_ids: {(input_ids.shape if hasattr(input_ids, 'shape') else type(input_ids))}"
                    )
                    print(
                        f"pixel_values: {(pixel_values.shape if hasattr(pixel_values, 'shape') else type(pixel_values))}"
                    )
                    print(
                        f"attention_mask: {(attention_mask.shape if hasattr(attention_mask, 'shape') else type(attention_mask))}"
                    )
                    with torch.no_grad():
                        was_training = expert_model.model.training
                        expert_model.model.eval()
                        try:
                            _ = expert_model.model.generate(
                                input_ids,
                                pixel_values=[pixel_values],
                                attention_mask=attention_mask,
                                **ovis_single_token_config,
                            )
                        finally:
                            if was_training:
                                expert_model.model.train()
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise ValueError(f"Ovis2 generation failed: {e}")
            elif hasattr(expert_model, "model") and hasattr(expert_model, "processor"):
                model_inputs = {
                    "input_ids": intermediate_state["input_ids"],
                    "attention_mask": intermediate_state["attention_mask"],
                    "pixel_values": intermediate_state["pixel_values"].to(
                        expert_model.model.dtype
                    ),
                }
                with torch.no_grad():
                    _ = expert_model.model.generate(
                        **model_inputs, **single_token_config
                    )
            else:
                raise NotImplementedError(
                    f"TTFT measurement not implemented for {type(expert_model)}"
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            ttft = end_time - start_time
            return ttft
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
    raise NotImplementedError(
        f"No precise TTFT measurement method available for {type(expert_model)}"
    )


def measure_expert_latency(expert_model, state_fn, device, num_warmup=2, num_repeats=3):
    (
        avg_latency,
        avg_ttft,
        avg_tps,
        memory_footprint,
    ) = measure_expert_latency_with_ttft_and_memory(
        expert_model, state_fn, device, num_warmup, num_repeats
    )
    return (avg_latency, avg_tps)


def calculate_comprehensive_cost(latencies, tokens_per_second, expert_name="Unknown"):
    if not latencies:
        return 10.0
    avg_latency = sum(latencies) / len(latencies)
    latency_std = np.std(latencies) if len(latencies) > 1 else 0.0
    avg_tps = (
        sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0.0
    )
    time_cost = avg_latency
    consistency_penalty = latency_std * 0.5
    efficiency_bonus = 0.0
    if avg_tps > 10:
        efficiency_bonus = -min(0.1, (avg_tps - 10) / 100)
    comprehensive_cost = time_cost + consistency_penalty + efficiency_bonus
    print(f"[COST] {expert_name} cost analysis")
    print(f"  Base latency: {time_cost:.3f}s")
    print(f"  Consistency penalty: +{consistency_penalty:.3f}s (std={latency_std:.3f})")
    print(f"  Efficiency bonus: {efficiency_bonus:.3f}s (TPS={avg_tps:.1f})")
    print(f"  Total cost: {comprehensive_cost:.3f}s")
    return max(0.1, comprehensive_cost)


def calculate_user_perceived_cost(latencies, expert_name="Unknown"):
    if not latencies:
        return 10.0
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = np.percentile(latencies, 95) if len(latencies) > 1 else avg_latency
    user_perceived_cost = 0.7 * avg_latency + 0.3 * p95_latency
    print(f"[COST] {expert_name} user perceived cost")
    print(f"  Weighted latency: {user_perceived_cost:.3f}s")
    return user_perceived_cost


def calculate_comprehensive_cost_v2(
    latencies, tokens_per_second, memory_footprint, expert_name="Unknown"
):
    if not latencies:
        return 10.0
    avg_latency = sum(latencies) / len(latencies)
    avg_tps = (
        sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0.0
    )
    latency_cost = avg_latency
    if avg_tps > 0:
        throughput_cost = 1.0 / avg_tps
    else:
        throughput_cost = 10.0
    comprehensive_cost = 1 / 2 * latency_cost + 1 / 2 * throughput_cost
    print(f"{expert_name} Comprehensive Cost Analysis:")
    print(f"   Latency Cost: {latency_cost:.3f}s (weight 50%)")
    print(f"   Throughput Cost: {throughput_cost:.3f}s (TPS={avg_tps:.1f}, weight 50%)")
    print(f"   Comprehensive Cost: {comprehensive_cost:.3f}s")
    return max(0.1, comprehensive_cost)


def calculate_resource_efficiency_score(
    latencies, tokens_per_second, memory_footprint, expert_name="Unknown"
):
    if not latencies:
        return 0.0
    avg_latency = sum(latencies) / len(latencies)
    avg_tps = (
        sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0.0
    )
    latency_efficiency = max(0, 100 - avg_latency * 50)
    throughput_efficiency = min(100, avg_tps * 5)
    efficiency_score = 1 / 2 * latency_efficiency + 1 / 2 * throughput_efficiency
    print(f"{expert_name} Resource Efficiency Score:")
    print(
        f"   Latency Efficiency: {latency_efficiency:.1f}/100 (latency={avg_latency:.3f}s)"
    )
    print(
        f"   Throughput Efficiency: {throughput_efficiency:.1f}/100 (TPS={avg_tps:.1f})"
    )
    print(f"   Overall Efficiency: {efficiency_score:.1f}/100")
    return efficiency_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure expert model computational costs"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Test data path (default: use TEST_PATH from config)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to extract per dataset",
    )
    parser.add_argument(
        "--warmup_runs", type=int, default=2, help="Number of warmup runs per sample"
    )
    parser.add_argument(
        "--timed_runs", type=int, default=3, help="Number of timed runs per sample"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: use DEVICE from config)",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Whether to save results to file"
    )
    default_output = os.path.join("cost_measurement", "expert_costs.json")
    parser.add_argument(
        "--output_file", type=str, default=default_output, help="Result file name"
    )
    return parser.parse_args()


def get_representative_samples(test_path, samples_per_dataset=3):
    dataset_samples = {}
    with open(test_path, "r") as f:
        test_data = [json.loads(line) for line in f]
    print(f"Read {len(test_data)} test data entries")
    for item in test_data:
        category = item.get("category", "")
        dataset_name = category.split("_")[0] if "_" in category else category
        if dataset_name not in dataset_samples:
            dataset_samples[dataset_name] = []
        dataset_samples[dataset_name].append(item)
    selected_samples = {}
    for dataset, samples in dataset_samples.items():
        count = min(samples_per_dataset, len(samples))
        if count == 0:
            continue
        selected = random.sample(samples, count)
        selected_samples[dataset] = selected
        print(f"Extracted {len(selected)} samples from dataset {dataset}")
    return selected_samples


def prepare_inputs_for_experts(sample, table_image_dir):
    table_data = sample.get("table", {})
    question = sample.get("question", "")
    category = sample.get("category", "")
    image_file_name = sample.get("image", None)
    if "InfoTabs" in category:
        from data.dataloader import _get_markdown_table_infotabs

        md_table = _get_markdown_table_infotabs(table_data, token_limit=3800)
    else:
        from data.dataloader import _get_markdown_table

        md_table = _get_markdown_table(table_data, token_limit=3800)
    prompt_body_tpl = markdown_template_mapping.get(
        category, DEFAULT_MARKDOWN_PROMPT_BODY
    )
    try:
        prompt_body = prompt_body_tpl.format(question=question)
    except KeyError:
        prompt_body = question
    prompt_for_text_expert = f"Markdown Table:\n{md_table}\n\n{prompt_body}"
    prompt_tpl_vlm = image_template_mapping.get(category, DEFAULT_IMAGE_PROMPT_TEMPLATE)
    try:
        prompt_for_vlm = prompt_tpl_vlm.format(question=question)
    except KeyError:
        prompt_for_vlm = question
    image_path_for_vlm = None
    if image_file_name and table_image_dir:
        potential_path = os.path.join(table_image_dir, image_file_name)
        if os.path.exists(potential_path):
            image_path_for_vlm = potential_path
    return {
        "text_expert_prompt": prompt_for_text_expert,
        "vlm_prompt": prompt_for_vlm,
        "vlm_image_path": image_path_for_vlm,
        "raw_table": table_data,
        "question": question,
        "category": category,
    }


def main():
    args = parse_args()
    device = args.device or cfg["TRAINING"]["DEVICE"]
    device = torch.device(device)
    print(f"Using device: {device}")
    test_path = args.test_data or cfg["DATA"]["TEST_PATH"]
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    print("Using test data")
    table_image_dir = cfg["DATA"].get(
        "TEST_TABLE_IMAGE_DIR", cfg["DATA"]["TABLE_IMAGE_DIR"]
    )
    print("\n--- Initializing Expert Models ---")
    text_expert = TableGPT2Expert(model_path=cfg["MODEL"]["TEXT_EXPERT_ID"]).to(device)
    vlm_expert_type = cfg["MODEL"].get("VLM_EXPERT_TYPE", "TableLLaVA")
    vlm_expert_id = cfg["MODEL"]["VLM_EXPERT_ID"]
    if vlm_expert_type.lower() == "ovis2":
        print(
            f"INFO: Using Ovis2Expert for cost measurement, model path: {vlm_expert_id}"
        )
        vlm_expert = Ovis2Expert(model_path=vlm_expert_id).to(device)
    elif vlm_expert_type.lower() == "tablellava":
        print(
            f"INFO: Using TableLLavaExpert for cost measurement, model path: {vlm_expert_id}"
        )
        vlm_expert = TableLLavaExpert(model_path=vlm_expert_id).to(device)
    else:
        print(
            f"WARN: Unknown VLM_EXPERT_TYPE '{vlm_expert_type}'. Defaulting to TableLLavaExpert"
        )
        vlm_expert = TableLLavaExpert(model_path=vlm_expert_id).to(device)
    fusion_api_cost = 0.3
    if cfg["MODEL"]["USE_LATE_FUSION"]:
        fusion_expert = LateFusionExpert(
            model_name=cfg["MODEL"]["FUSION_EXPERT_MODEL_ID"],
            api_key=cfg["MODEL"]["GEMINI_API_KEY"],
        )
        fusion_api_cost = fusion_expert.cost
    print("\n--- Extracting Representative Samples ---")
    selected_samples = get_representative_samples(test_path, args.num_samples)
    all_samples = []
    for samples in selected_samples.values():
        all_samples.extend(samples)
    print(
        f"\n--- Measuring Expert Latency ({args.warmup_runs} warmup runs, {args.timed_runs} timed runs per sample) ---"
    )
    text_expert_latencies = []
    vlm_expert_latencies = []
    text_expert_tokens_per_second = []
    vlm_expert_tokens_per_second = []
    progress_bar = tqdm(all_samples, desc="Measuring sample latency")
    for sample in progress_bar:
        inputs = prepare_inputs_for_experts(sample, table_image_dir)

        def get_text_expert_state():
            _, state = text_expert.extract_gate_features_and_intermediate_state(
                [inputs["text_expert_prompt"]]
            )
            return {k: v[0:1] for k, v in state.items()}

        def get_vlm_expert_state():
            _, state = vlm_expert.extract_gate_features_and_intermediate_state(
                [inputs["vlm_prompt"]], [inputs["vlm_image_path"]]
            )
            if isinstance(vlm_expert, Ovis2Expert):
                single_sample_state = {}
                for key, value in state.items():
                    if isinstance(value, list) and len(value) > 0:
                        single_sample_state[key] = value[0]
                    else:
                        single_sample_state[key] = value
                return single_sample_state
            else:
                return {
                    k: v[0:1] if isinstance(v, torch.Tensor) else v
                    for k, v in state.items()
                }

        (
            text_latency,
            text_ttft,
            text_tps,
            text_memory,
        ) = measure_expert_latency_with_ttft_and_memory(
            text_expert,
            get_text_expert_state,
            device,
            num_warmup=args.warmup_runs,
            num_repeats=args.timed_runs,
        )
        (
            vlm_latency,
            vlm_ttft,
            vlm_tps,
            vlm_memory,
        ) = measure_expert_latency_with_ttft_and_memory(
            vlm_expert,
            get_vlm_expert_state,
            device,
            num_warmup=args.warmup_runs,
            num_repeats=args.timed_runs,
        )
        text_expert_latencies.append(text_latency)
        vlm_expert_latencies.append(vlm_latency)
        text_expert_tokens_per_second.append(text_tps)
        vlm_expert_tokens_per_second.append(vlm_tps)
        ttft_text_display = f"{text_ttft:.4f}s" if not np.isnan(text_ttft) else "FAIL"
        ttft_vlm_display = f"{vlm_ttft:.4f}s" if not np.isnan(vlm_ttft) else "FAIL"
        progress_bar.set_postfix(
            {
                "TextExpert": f"{text_latency:.4f}s/{text_tps:.1f}tps (TTFT:{ttft_text_display})",
                "VLMExpert": f"{vlm_latency:.4f}s/{vlm_tps:.1f}tps (TTFT:{ttft_vlm_display})",
            }
        )
    avg_text_memory = {
        "gpu_memory_gb": 0.0,
        "cpu_memory_gb": 0.0,
        "total_memory_gb": 0.0,
    }
    avg_vlm_memory = {
        "gpu_memory_gb": 0.0,
        "cpu_memory_gb": 0.0,
        "total_memory_gb": 0.0,
    }
    print("\n" + "=" * 80)
    print("Expert Efficiency Analysis (Based on LLM Efficiency Standards)")
    print("=" * 80)
    print(
        "Using equal weights: Latency=50%, Throughput=50% (Memory measurement disabled)"
    )
    avg_text_latency = (
        sum(text_expert_latencies) / len(text_expert_latencies)
        if text_expert_latencies
        else 0
    )
    avg_vlm_latency = (
        sum(vlm_expert_latencies) / len(vlm_expert_latencies)
        if vlm_expert_latencies
        else 0
    )
    avg_text_tps = (
        sum(text_expert_tokens_per_second) / len(text_expert_tokens_per_second)
        if text_expert_tokens_per_second
        else 0
    )
    avg_vlm_tps = (
        sum(vlm_expert_tokens_per_second) / len(vlm_expert_tokens_per_second)
        if vlm_expert_tokens_per_second
        else 0
    )
    text_comprehensive_cost_v2 = calculate_comprehensive_cost_v2(
        text_expert_latencies,
        text_expert_tokens_per_second,
        avg_text_memory,
        "TextExpert",
    )
    vlm_comprehensive_cost_v2 = calculate_comprehensive_cost_v2(
        vlm_expert_latencies, vlm_expert_tokens_per_second, avg_vlm_memory, "VLMExpert"
    )
    text_efficiency_score = calculate_resource_efficiency_score(
        text_expert_latencies,
        text_expert_tokens_per_second,
        avg_text_memory,
        "TextExpert",
    )
    vlm_efficiency_score = calculate_resource_efficiency_score(
        vlm_expert_latencies, vlm_expert_tokens_per_second, avg_vlm_memory, "VLMExpert"
    )
    parallel_experts_latency = max(avg_text_latency, avg_vlm_latency)
    fusion_latency = parallel_experts_latency + fusion_api_cost
    fusion_memory_dict = {
        "total_memory_gb": 0.0,
        "gpu_memory_gb": 0.0,
        "cpu_memory_gb": 0.0,
    }
    fusion_estimated_tps = (
        min(avg_text_tps, avg_vlm_tps) if avg_text_tps > 0 and avg_vlm_tps > 0 else 0
    )
    fusion_comprehensive_cost_v2 = calculate_comprehensive_cost_v2(
        [fusion_latency], [fusion_estimated_tps], fusion_memory_dict, "Fusion"
    )
    fusion_efficiency_score = calculate_resource_efficiency_score(
        [fusion_latency], [fusion_estimated_tps], fusion_memory_dict, "Fusion"
    )
    print(f"\nLLM Efficiency Evaluation Results:")
    print(
        f"{'Expert':<12} {'Latency(s)':<10} {'Throughput(TPS)':<15} {'Comp.Cost':<10} {'Efficiency':<10}"
    )
    print("-" * 65)
    print(
        f"{'TextExpert':<12} {avg_text_latency:<10.3f} {avg_text_tps:<15.1f} {text_comprehensive_cost_v2:<10.3f} {text_efficiency_score:<10.1f}"
    )
    print(
        f"{'VLMExpert':<12} {avg_vlm_latency:<10.3f} {avg_vlm_tps:<15.1f} {vlm_comprehensive_cost_v2:<10.3f} {vlm_efficiency_score:<10.1f}"
    )
    print(
        f"{'Fusion':<12} {fusion_latency:<10.3f} {fusion_estimated_tps:<15.1f} {fusion_comprehensive_cost_v2:<10.3f} {fusion_efficiency_score:<10.1f}"
    )
    print(f"\n" + "=" * 80)
    print("Recommended Configuration Code (Based on LLM Efficiency Standards)")
    print("=" * 80)
    print(
        f"\nComprehensive cost configuration (latency + throughput, equal weights)"
    )
    print(f'"EXPERT_COSTS": {{')
    print(
        f'    "TextExpert": {text_comprehensive_cost_v2:.6f}, latency+throughput'
    )
    print(
        f'    "VLMExpert": {vlm_comprehensive_cost_v2:.6f}, latency+throughput'
    )
    print(
        f'    "Fusion": {fusion_comprehensive_cost_v2:.6f}, latency+throughput'
    )
    print(f"}}")
    print(f"\nLatency-only configuration (comparison)")
    print(f'"EXPERT_COSTS": {{')
    print(f'    "TextExpert": {avg_text_latency:.6f}, average latency')
    print(f'    "VLMExpert": {avg_vlm_latency:.6f}, average latency')
    print(f'    "Fusion": {fusion_latency:.6f}, parallel latency + API cost')
    print(f"}}")
    print(f"\nRecommended LAMBDA_RESOURCE_LOSS value:")
    recommended_cost = min(
        text_comprehensive_cost_v2,
        vlm_comprehensive_cost_v2,
        fusion_comprehensive_cost_v2,
    )
    suggested_lambda = 0.05 / max(recommended_cost, 1.0)
    print(
        f"Based on comprehensive cost: {suggested_lambda:.6f} (current lowest cost: {recommended_cost:.3f}s)"
    )
    print(f"\nEfficiency Comparison Analysis:")
    max_efficiency = max(
        text_efficiency_score, vlm_efficiency_score, fusion_efficiency_score
    )
    min_cost = min(
        text_comprehensive_cost_v2,
        vlm_comprehensive_cost_v2,
        fusion_comprehensive_cost_v2,
    )
    if text_efficiency_score == max_efficiency:
        print(f"Highest Efficiency: TextExpert ({text_efficiency_score:.1f}/100)")
    elif vlm_efficiency_score == max_efficiency:
        print(f"Highest Efficiency: VLMExpert ({vlm_efficiency_score:.1f}/100)")
    else:
        print(f"Highest Efficiency: Fusion ({fusion_efficiency_score:.1f}/100)")
    if text_comprehensive_cost_v2 == min_cost:
        print(f"Lowest Cost: TextExpert ({text_comprehensive_cost_v2:.3f}s)")
    elif vlm_comprehensive_cost_v2 == min_cost:
        print(f"Lowest Cost: VLMExpert ({vlm_comprehensive_cost_v2:.3f}s)")
    else:
        print(f"Lowest Cost: Fusion ({fusion_comprehensive_cost_v2:.3f}s)")
    if args.save_results:
        results = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cost_calculation_weights": {
                "latency_weight": 1 / 2,
                "throughput_weight": 1 / 2,
                "memory_weight": 0,
                "note": "Equal weights (1/2:1/2) hardcoded for latency and throughput, memory measurement disabled",
            },
            "expert_costs": {
                "TextExpert": {
                    "avg_latency": avg_text_latency,
                    "tokens_per_second": avg_text_tps,
                    "comprehensive_cost_v2": text_comprehensive_cost_v2,
                    "efficiency_score": text_efficiency_score,
                    "samples": len(text_expert_latencies),
                },
                "VLMExpert": {
                    "avg_latency": avg_vlm_latency,
                    "tokens_per_second": avg_vlm_tps,
                    "comprehensive_cost_v2": vlm_comprehensive_cost_v2,
                    "efficiency_score": vlm_efficiency_score,
                    "samples": len(vlm_expert_latencies),
                },
                "Fusion": {
                    "total_cost": fusion_latency,
                    "estimated_tps": fusion_estimated_tps,
                    "comprehensive_cost_v2": fusion_comprehensive_cost_v2,
                    "efficiency_score": fusion_efficiency_score,
                    "breakdown": {
                        "parallel_experts_max_latency": parallel_experts_latency,
                        "api_cost": fusion_api_cost,
                    },
                },
            },
            "settings": {
                "num_samples_per_dataset": args.num_samples,
                "warmup_runs": args.warmup_runs,
                "timed_runs": args.timed_runs,
                "device": str(device),
                "total_samples": len(all_samples),
                "datasets": list(selected_samples.keys()),
            },
        }
        output_dir = os.path.dirname(args.output_file)
        if output_dir and (not os.path.exists(output_dir)):
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved")
    print("\nMeasurement completed.")
    print(
        "\nRecommend adjusting LAMBDA_RESOURCE_LOSS in TRAINING section of config.py:"
    )
    print(
        f"Based on LLM efficiency evaluation results, in parallel execution, slower expert is {('TextExpert' if avg_text_latency >= avg_vlm_latency else 'VLMExpert')} (latency: {parallel_experts_latency:.2f}s)"
    )
    print(
        f"Recommended LAMBDA_RESOURCE_LOSS value (based on comprehensive cost): {suggested_lambda:.6f}"
    )


if __name__ == "__main__":
    main()
