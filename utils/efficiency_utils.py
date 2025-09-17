import time
import numpy as np
import torch
from transformers import AutoTokenizer, TextStreamer


class EfficiencyStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, monitor: "InferenceEfficiencyMonitor"):
        super().__init__(tokenizer)
        self.monitor = monitor
        self.first_token_received = False
        self.token_timestamps = []

    def on_text(self, text: str, **kwargs):
        if text:
            self.token_timestamps.append(time.perf_counter())
        if not self.first_token_received and text.strip():
            self.monitor.record_first_token_time()
            self.first_token_received = True

    def reset(self):
        self.first_token_received = False
        self.token_timestamps.clear()


class InferenceEfficiencyMonitor:
    def __init__(self, tokenizer_name: str, trust_remote_code: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code
        )
        self.streamer = EfficiencyStreamer(self.tokenizer, self)
        self.latencies = []
        self.ttfts = []
        self.generated_tokens_counts = []
        self.tokens_per_sec = []
        self.comprehensive_costs = []
        self._start_time = None
        self._first_token_time = None
        self._current_model = None
        self._current_intermediate_state = None
        self._current_device = None
        self._current_ttft_override = None

    def start_batch(self, model=None, intermediate_state=None, device=None):
        self.streamer.reset()
        self._first_token_time = None
        self._start_time = time.perf_counter()
        self._current_model = model
        self._current_intermediate_state = intermediate_state
        self._current_device = device
        self._current_ttft_override = None

    def record_first_token_time(self):
        if self._first_token_time is None and len(self.streamer.token_timestamps) > 0:
            self._first_token_time = self.streamer.token_timestamps[0]
        elif self._first_token_time is None:
            self._first_token_time = time.perf_counter()

    def measure_ttft_for_expert(self, expert_model, intermediate_state, device):
        try:
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
                        inputs_embeds=intermediate_state.get("inputs_embeds"),
                        attention_mask=intermediate_state.get("attention_mask"),
                        **single_token_config,
                    )
            elif hasattr(expert_model, "model") and hasattr(
                expert_model, "text_tokenizer"
            ):
                with torch.no_grad():
                    input_ids = intermediate_state.get("input_ids")
                    pixel_values = intermediate_state.get("pixel_values")
                    attention_mask = intermediate_state.get("attention_mask")
                    if pixel_values is not None and (
                        not isinstance(pixel_values, list)
                    ):
                        pixel_values = [pixel_values]
                    _ = expert_model.model.generate(
                        input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        **single_token_config,
                    )
            elif hasattr(expert_model, "model") and hasattr(expert_model, "processor"):
                with torch.no_grad():
                    model_inputs = {
                        "input_ids": intermediate_state.get("input_ids"),
                        "attention_mask": intermediate_state.get("attention_mask"),
                        "pixel_values": intermediate_state.get("pixel_values"),
                    }
                    if model_inputs["pixel_values"] is not None:
                        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(
                            expert_model.model.dtype
                        )
                    _ = expert_model.model.generate(
                        **model_inputs, **single_token_config
                    )
            else:
                return float("nan")
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            ttft = end_time - start_time
            return ttft
        except Exception as e:
            return float("nan")

    def measure_ttft_precisely(self, model, intermediate_state, device):
        if model is None or intermediate_state is None or device is None:
            return float("nan")
        try:
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
            model_type = type(model).__name__
            if hasattr(model, "text_expert") and hasattr(model.text_expert, "model"):
                expert_model = model.text_expert.model
                if hasattr(expert_model, "generate"):
                    with torch.no_grad():
                        _ = expert_model.generate(
                            inputs_embeds=intermediate_state.get("inputs_embeds"),
                            attention_mask=intermediate_state.get("attention_mask"),
                            **single_token_config,
                        )
            elif hasattr(model, "vlm_expert") and hasattr(model.vlm_expert, "model"):
                expert_model = model.vlm_expert.model
                if hasattr(expert_model, "generate"):
                    with torch.no_grad():
                        model_inputs = {
                            "input_ids": intermediate_state.get("input_ids"),
                            "attention_mask": intermediate_state.get("attention_mask"),
                            "pixel_values": intermediate_state.get("pixel_values"),
                        }
                        if model_inputs["pixel_values"] is not None:
                            model_inputs["pixel_values"] = model_inputs[
                                "pixel_values"
                            ].to(expert_model.dtype)
                        _ = expert_model.generate(**model_inputs, **single_token_config)
            elif hasattr(model, "model") and hasattr(model.model, "generate"):
                with torch.no_grad():
                    if "pixel_values" in intermediate_state:
                        model_inputs = {
                            "input_ids": intermediate_state.get("input_ids"),
                            "attention_mask": intermediate_state.get("attention_mask"),
                            "pixel_values": intermediate_state.get("pixel_values").to(
                                model.model.dtype
                            ),
                        }
                        _ = model.model.generate(**model_inputs, **single_token_config)
                    else:
                        _ = model.model.generate(
                            inputs_embeds=intermediate_state.get("inputs_embeds"),
                            attention_mask=intermediate_state.get("attention_mask"),
                            **single_token_config,
                        )
            else:
                return float("nan")
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            ttft = end_time - start_time
            return ttft
        except Exception as e:
            return float("nan")

    def count_tokens(self, text):
        if not text:
            return 0
        tokens = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        return tokens.numel()

    def calculate_comprehensive_cost(self, latency, tokens_per_second):
        if latency <= 0:
            return 10.0
        latency_cost = latency
        if tokens_per_second > 0:
            throughput_cost = 1.0 / tokens_per_second
        else:
            throughput_cost = 10.0
        comprehensive_cost = 1 / 2 * latency_cost + 1 / 2 * throughput_cost
        return max(0.1, comprehensive_cost)

    def end_batch(self, generated_text: str):
        end_time = time.perf_counter()
        if self._start_time is None:
            print(
                "WARNING (EfficiencyMonitor): end_batch() called without start_batch(). Skipping."
            )
            return
        latency = end_time - self._start_time
        self.latencies.append(latency)
        ttft = float("nan")
        if self._current_ttft_override is not None:
            ttft = self._current_ttft_override
        elif (
            self._current_model is not None
            and self._current_intermediate_state is not None
            and (self._current_device is not None)
        ):
            try:
                ttft = self.measure_ttft_precisely(
                    self._current_model,
                    self._current_intermediate_state,
                    self._current_device,
                )
            except Exception as e:
                ttft = float("nan")
        elif self._first_token_time is not None:
            ttft = self._first_token_time - self._start_time
            if ttft > latency:
                print(
                    f"WARNING: TTFT ({ttft:.4f}s) > Latency ({latency:.4f}s), using latency as TTFT"
                )
                ttft = latency
            elif ttft < 0:
                print(f"WARNING: Negative TTFT ({ttft:.4f}s), using 0.001s as minimum")
                ttft = 0.001
        if np.isnan(ttft):
            ttft = float("nan")
        self.ttfts.append(ttft)
        num_tokens = self.count_tokens(generated_text)
        self.generated_tokens_counts.append(num_tokens)
        tps = 0.0
        timestamps = self.streamer.token_timestamps
        if num_tokens > 1 and len(timestamps) > 1:
            generation_time = timestamps[-1] - timestamps[0]
            if generation_time > 1e-06:
                tps = (num_tokens - 1) / generation_time
        elif num_tokens > 1 and (not np.isnan(ttft)) and (ttft < latency):
            generation_time = latency - ttft
            if generation_time > 1e-06:
                tps = (num_tokens - 1) / generation_time
        elif num_tokens > 0 and latency > 1e-06:
            tps = num_tokens / latency
        self.tokens_per_sec.append(tps)
        comprehensive_cost = self.calculate_comprehensive_cost(latency, tps)
        self.comprehensive_costs.append(comprehensive_cost)
        self._start_time = None
        self._first_token_time = None
        self._current_model = None
        self._current_intermediate_state = None
        self._current_device = None
        self._current_ttft_override = None

    def get_summary_report(self) -> str:
        if not self.latencies:
            return "No data collected for efficiency summary."
        total_samples = len(self.latencies)
        total_time = np.sum(self.latencies)
        total_tokens = np.sum(self.generated_tokens_counts)
        avg_latency = np.mean(self.latencies)
        valid_ttfts = [t for t in self.ttfts if not np.isnan(t)]
        avg_ttft = np.mean(valid_ttfts) if valid_ttfts else float("nan")
        ttft_success_rate = (
            len(valid_ttfts) / len(self.ttfts) * 100 if self.ttfts else 0
        )
        overall_tps = total_tokens / total_time if total_time > 0 else 0
        avg_instance_tps = (
            np.mean([tps for tps in self.tokens_per_sec if tps > 0])
            if any((tps > 0 for tps in self.tokens_per_sec))
            else 0
        )
        avg_comprehensive_cost = (
            np.mean(self.comprehensive_costs) if self.comprehensive_costs else 0
        )
        summary = f"\n\n--- Enhanced Efficiency Measurement Summary ---\nTotal Samples: {total_samples}\nTotal Inference Time: {total_time:.2f} s\nOverall QPS (Queries Per Second): {total_samples / total_time:.2f}\nOverall Generation Speed: {overall_tps:.2f} tokens/s\n----------------------------------------\nLatency (End-to-End): \n  - Average: {avg_latency:.4f} s\n  - Std Dev: {np.std(self.latencies):.4f} s\nTime To First Token (TTFT): \n  - Average: {avg_ttft:.4f} s (Success Rate: {ttft_success_rate:.1f}%)\n  - Std Dev: {np.std(valid_ttfts):.4f} s\nGeneration Speed (per instance, after first token):\n  - Average: {avg_instance_tps:.2f} tokens/s\n----------------------------------------\nComprehensive Cost Analysis:\n  - Average Cost: {avg_comprehensive_cost:.4f} s\n  - Cost Std Dev: {np.std(self.comprehensive_costs):.4f} s\n  - Cost Formula: 50% Latency + 50% Throughput^-1\n"
        return summary

    def get_last_instance_metrics(self) -> dict:
        if not self.latencies:
            return {}
        metrics = {
            "latency": self.latencies[-1],
            "ttft": self.ttfts[-1],
            "generated_tokens": self.generated_tokens_counts[-1],
            "tokens_per_second": self.tokens_per_sec[-1],
        }
        if self.comprehensive_costs:
            metrics["comprehensive_cost"] = self.comprehensive_costs[-1]
        return metrics
