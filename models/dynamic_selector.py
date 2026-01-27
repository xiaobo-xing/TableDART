import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import json
from .experts import TableGPT2Expert, TableLLavaExpert, LateFusionExpert
from .gating_network import MlpGatingNetwork
from utils.helpers import (
    gumbel_softmax,
    get_accuracy_for_training,
    extract_tqa_answer_list,
)
from data.preprocessing import get_question_embedding_for_gate

try:
    from project_config.config import cfg
except ImportError:
    from config.config import cfg
DEBUG_COST_CALCULATION = False
COST_DEBUG_SAMPLE_LIMIT = 3
DEBUG_EXPERT_SELECTION = False


def get_expert_for_vlm_type():
    vlm_expert_id = cfg["MODEL"]["VLM_EXPERT_ID"]
    vlm_expert_type = cfg["MODEL"].get("VLM_EXPERT_TYPE", "TableLLaVA")
    if vlm_expert_type.lower() == "ovis2":
        from .experts import Ovis2Expert

        print(f"INFO: VLM_EXPERT_TYPE set to Ovis2; loading Ovis2Expert from {vlm_expert_id}")
        return Ovis2Expert(model_path=vlm_expert_id)
    elif vlm_expert_type.lower() == "tablellava":
        print(
            f"INFO: VLM_EXPERT_TYPE set to TableLLaVA; loading TableLLavaExpert from {vlm_expert_id}"
        )
        return TableLLavaExpert(model_path=vlm_expert_id)
    else:
        print(
            f"WARN: Unknown VLM_EXPERT_TYPE {vlm_expert_type}; defaulting to TableLLavaExpert from {vlm_expert_id}"
        )
        return TableLLavaExpert(model_path=vlm_expert_id)


class DynamicExpertSelector(nn.Module):
    """
    Dynamic expert selector that routes table-query pairs to optimal paths.
    - Maintains frozen pretrained single-modality experts (Text, VLM)
    - Learns a lightweight MLP gating network to route queries
    - Optionally supports late fusion via API calls
    """
    def __init__(self, gate_hidden_dim, use_late_fusion):
        super().__init__()
        self.use_late_fusion = use_late_fusion
        self.device = torch.device(cfg["TRAINING"]["DEVICE"])
        
        # Initialize frozen text-based expert
        print("Initializing Experts...")
        self.expert_text = TableGPT2Expert(
            model_path=cfg["MODEL"]["TEXT_EXPERT_ID"]
        ).to(self.device)
        
        # Initialize frozen vision-language expert
        vlm_expert_id = cfg["MODEL"]["VLM_EXPERT_ID"]
        vlm_expert_type = cfg["MODEL"].get("VLM_EXPERT_TYPE", "TableLLaVA")
        if vlm_expert_type.lower() == "ovis2":
            from .experts import Ovis2Expert

            print(
                f"INFO: VLM_EXPERT_TYPE set to Ovis2; loading Ovis2Expert from {vlm_expert_id}"
            )
            self.expert_vlm = Ovis2Expert(model_path=vlm_expert_id).to(self.device)
        elif vlm_expert_type.lower() == "tablellava":
            print(
                f"INFO: VLM_EXPERT_TYPE set to TableLLaVA; loading TableLLavaExpert from {vlm_expert_id}"
            )
            self.expert_vlm = TableLLavaExpert(model_path=vlm_expert_id).to(self.device)
        else:
            print(
                f"WARN: Unknown VLM_EXPERT_TYPE {vlm_expert_type}; defaulting to TableLLavaExpert from {vlm_expert_id}"
            )
            self.expert_vlm = TableLLavaExpert(model_path=vlm_expert_id).to(self.device)
        
        # Store base experts (only Text and VLM are frozen)
        self.base_experts = nn.ModuleList([self.expert_text, self.expert_vlm])
        self.all_experts_for_selection_list_for_cost = [
            self.expert_text,
            self.expert_vlm,
        ]
        
        # Initialize fusion expert via API (if enabled)
        self.fusion_expert_api = None
        self.path_names = ["TextExpert", "VLMExpert"]
        if self.use_late_fusion:
            self.fusion_expert_api = LateFusionExpert(
                model_name=cfg["MODEL"]["FUSION_EXPERT_MODEL_ID"],
                api_key=cfg["MODEL"]["GEMINI_API_KEY"],
            )
            self.all_experts_for_selection_list_for_cost.append(self.fusion_expert_api)
            self.path_names.append("FusionExpert")
        
        # Total number of routing paths
        self.num_paths = len(self.all_experts_for_selection_list_for_cost)
        
        # Determine question embedding dimension for gating network
        print("Determining question embedding dimension...")
        from transformers import AutoConfig

        try:
            _q_embed_config = AutoConfig.from_pretrained(
                cfg["MODEL"]["QUESTION_EMBED_MODEL_FOR_GATE_ID"]
            )
            self.question_embed_dim = _q_embed_config.hidden_size
        except Exception:
            self.question_embed_dim = 384
        print(f"Question embedding dimension: {self.question_embed_dim}")
        
        # Calculate gating network input dimension
        print("Calculating Gate Input Dimension...")
        self.gate_input_dim = (
            self.expert_text.gate_feature_dim
            + self.expert_vlm.gate_feature_dim
            + self.question_embed_dim
        )
        cfg["DATA"]["GATE_INPUT_DIM"] = self.gate_input_dim
        print(f"INFO: DynamicExpertSelector GATE_INPUT_DIM: {self.gate_input_dim}")
        self.gating_network = MlpGatingNetwork(
            input_dim=self.gate_input_dim,
            num_paths=self.num_paths,
            hidden_dim=cfg["MODEL"]["GATE_HIDDEN_DIM"],
        ).to(self.device)
        print("INFO: Using fixed expert costs from config...")
        path_costs_list = []
        if "EXPERT_COSTS" in cfg["MODEL"]:
            path_costs_list = [
                cfg["MODEL"]["EXPERT_COSTS"].get("TextExpert", 1.2),
                cfg["MODEL"]["EXPERT_COSTS"].get("VLMExpert", 2.5),
            ]
            if self.use_late_fusion:
                path_costs_list.append(cfg["MODEL"]["EXPERT_COSTS"].get("Fusion", 4.0))
            print(f"INFO: Using configured expert costs: {path_costs_list}")
        else:
            path_costs_list = [
                expert.cost for expert in self.all_experts_for_selection_list_for_cost
            ]
            print(f"INFO: Using expert default costs: {path_costs_list}")
        self.path_costs_tensor = torch.tensor(path_costs_list, dtype=torch.float32).to(
            self.device
        )
        print(f"INFO: Path costs for regularization: {self.path_costs_tensor.cpu().tolist()}")

    def forward(
        self,
        raw_table_batch,
        original_questions_batch,
        prompts_text_batch,
        prompts_vlm_batch,
        image_paths_vlm_batch,
        target_texts_str_batch=None,
        categories_batch=None,
        is_training=True,
        **generation_kwargs,
    ):
        batch_size = len(original_questions_batch)
        actual_batch_size = min(
            len(original_questions_batch),
            len(prompts_text_batch),
            len(prompts_vlm_batch),
            len(image_paths_vlm_batch),
        )
        if actual_batch_size != batch_size:
            print(
                f"WARN: Input batch size mismatch, using min size: {actual_batch_size}"
            )
            batch_size = actual_batch_size
            original_questions_batch = original_questions_batch[:batch_size]
            prompts_text_batch = prompts_text_batch[:batch_size]
            prompts_vlm_batch = prompts_vlm_batch[:batch_size]
            image_paths_vlm_batch = image_paths_vlm_batch[:batch_size]
            if target_texts_str_batch:
                target_texts_str_batch = target_texts_str_batch[:batch_size]
            if categories_batch:
                categories_batch = categories_batch[:batch_size]
        experts_should_have_grads = (
            is_training and cfg["TRAINING"]["FREEZE_EXPERTS"] is False
        )
        with torch.set_grad_enabled(experts_should_have_grads):
            (
                text_gate_feats,
                text_intermediate_state,
            ) = self.expert_text.extract_gate_features_and_intermediate_state(
                prompts_text_batch
            )
            (
                vlm_gate_feats,
                vlm_intermediate_state,
            ) = self.expert_vlm.extract_gate_features_and_intermediate_state(
                prompts_vlm_batch, image_paths_vlm_batch
            )
            if text_gate_feats.shape[0] != vlm_gate_feats.shape[0]:
                print(
                    f"WARN: Feature batch size mismatch - Text: {text_gate_feats.shape[0]}, VLM: {vlm_gate_feats.shape[0]}"
                )
                min_batch_size = min(text_gate_feats.shape[0], vlm_gate_feats.shape[0])
                text_gate_feats = text_gate_feats[:min_batch_size]
                vlm_gate_feats = vlm_gate_feats[:min_batch_size]
        with torch.no_grad():
            q_embeddings_for_gate = torch.stack(
                [
                    get_question_embedding_for_gate(q, self.device)
                    for q in original_questions_batch
                ]
            )
            min_batch_size = min(text_gate_feats.shape[0], vlm_gate_feats.shape[0])
            if q_embeddings_for_gate.shape[0] != min_batch_size:
                print(
                    f"WARN: Question embedding batch size mismatch - Expected: {min_batch_size}, Got: {q_embeddings_for_gate.shape[0]}"
                )
                q_embeddings_for_gate = q_embeddings_for_gate[:min_batch_size]
        combined_gate_features = torch.cat(
            [text_gate_feats, vlm_gate_feats, q_embeddings_for_gate], dim=1
        )
        gate_logits = self.gating_network(combined_gate_features)
        task_loss, resource_loss, total_loss_for_gate = (None, None, None)
        all_generated_texts_for_return = [""] * batch_size
        gate_probs_for_selection_output = None
        if (
            is_training
            and target_texts_str_batch is not None
            and (categories_batch is not None)
        ):
            gate_pred_log_probs = F.log_softmax(
                gate_logits / (cfg["TRAINING"]["GATE_LOGITS_TEMPERATURE"] + 1e-06),
                dim=-1,
            )
            gate_pred_probs_for_resource_loss = F.softmax(
                gate_logits / (cfg["TRAINING"]["GATE_LOGITS_TEMPERATURE"] + 1e-06),
                dim=-1,
            )
            total_task_loss = 0.0
            total_resource_loss = 0.0
            for i in range(batch_size):
                current_text_state = {
                    k: v[i : i + 1] for k, v in text_intermediate_state.items()
                }
                current_vlm_state = {}
                if (
                    hasattr(self.expert_vlm, "__class__")
                    and "Ovis2" in self.expert_vlm.__class__.__name__
                ):
                    for k, v in vlm_intermediate_state.items():
                        if isinstance(v, list) and len(v) > i:
                            current_vlm_state[k] = v[i]
                        else:
                            print(
                                f"WARN: Ovis2 training state {k} missing for sample {i}"
                            )
                            current_vlm_state[k] = None
                else:
                    for k, v in vlm_intermediate_state.items():
                        current_vlm_state[k] = v[i : i + 1]
                target_str = target_texts_str_batch[i]
                actual_sample_category = categories_batch[i]
                with torch.no_grad():
                    pred_text_e1_list = self.expert_text.continue_generation_from_state(
                        current_text_state, **generation_kwargs
                    )
                    pred_text_e1 = (
                        pred_text_e1_list[0]
                        if isinstance(pred_text_e1_list, list)
                        else pred_text_e1_list
                    )
                    score1 = get_accuracy_for_training(pred_text_e1, target_str)
                    pred_text_e2_list = self.expert_vlm.continue_generation_from_state(
                        current_vlm_state, **generation_kwargs
                    )
                    pred_text_e2 = (
                        pred_text_e2_list[0]
                        if isinstance(pred_text_e2_list, list)
                        else pred_text_e2_list
                    )
                    score2 = get_accuracy_for_training(pred_text_e2, target_str)
                    score3 = 0.0
                    pred_text_e3 = ""
                    if self.use_late_fusion and self.fusion_expert_api:
                        pred_text_e3 = self.fusion_expert_api.generate_full(
                            original_questions_batch[i],
                            raw_table_batch[i],
                            pred_text_e1,
                            pred_text_e2,
                            dataset_category=actual_sample_category,
                            **generation_kwargs,
                        )
                        score3 = get_accuracy_for_training(pred_text_e3, target_str)
                path_scores_sample = [score1, score2]
                if self.use_late_fusion:
                    path_scores_sample.append(score3)
                scores_tensor_sample = torch.tensor(
                    path_scores_sample, device=self.device, dtype=torch.float32
                )
                target_probs_sample = F.softmax(
                    scores_tensor_sample
                    / (cfg["TRAINING"]["GATE_TARGET_SCORE_TEMP"] + 1e-06),
                    dim=-1,
                ).detach()
                kl_loss_fn = nn.KLDivLoss(reduction="sum")
                sample_task_loss = kl_loss_fn(
                    gate_pred_log_probs[i], target_probs_sample
                )
                expected_path_cost_sample = (
                    gate_pred_probs_for_resource_loss[i] * self.path_costs_tensor
                ).sum()
                sample_resource_loss = (
                    expected_path_cost_sample * cfg["TRAINING"]["LAMBDA_RESOURCE_LOSS"]
                )
                if DEBUG_COST_CALCULATION and i < COST_DEBUG_SAMPLE_LIMIT:
                    print("\n" + "=" * 30 + " COST CALCULATION DEBUG " + "=" * 30)
                    print(
                        f"Sample {i + 1}/{batch_size} - Category: {actual_sample_category}"
                    )
                    current_gate_logits = gate_logits[i].cpu().detach().numpy()
                    current_gate_probs = (
                        gate_pred_probs_for_resource_loss[i].cpu().detach().numpy()
                    )
                    print(f"\nGate Logits (raw): {current_gate_logits}")
                    print(f"Gate Probabilities (softmax): {current_gate_probs}")
                    print(f"   -> TextExpert prob: {current_gate_probs[0]:.4f}")
                    print(f"   -> VLMExpert prob: {current_gate_probs[1]:.4f}")
                    if self.use_late_fusion:
                        print(f"   -> Fusion prob: {current_gate_probs[2]:.4f}")
                    path_costs_cpu = self.path_costs_tensor.cpu().detach().numpy()
                    print(f"\nPath Costs: {path_costs_cpu}")
                    print(f"   -> TextExpert cost: {path_costs_cpu[0]:.2f}s")
                    print(f"   -> VLMExpert cost: {path_costs_cpu[1]:.2f}s")
                    if len(path_costs_cpu) > 2:
                        print(f"   -> Fusion cost: {path_costs_cpu[2]:.2f}s")
                    expected_cost_breakdown = current_gate_probs * path_costs_cpu
                    print(f"\nExpected Cost Calculation:")
                    print(
                        f"   prob[TextExpert] × cost[TextExpert] = {current_gate_probs[0]:.4f} × {path_costs_cpu[0]:.2f} = {expected_cost_breakdown[0]:.4f}"
                    )
                    print(
                        f"   prob[VLMExpert] × cost[VLMExpert] = {current_gate_probs[1]:.4f} × {path_costs_cpu[1]:.2f} = {expected_cost_breakdown[1]:.4f}"
                    )
                    if len(path_costs_cpu) > 2:
                        print(
                            f"   prob[Fusion] × cost[Fusion] = {current_gate_probs[2]:.4f} × {path_costs_cpu[2]:.2f} = {expected_cost_breakdown[2]:.4f}"
                        )
                    print(
                        f"   Expected Total Cost = {expected_cost_breakdown.sum():.4f}s"
                    )
                    lambda_resource = cfg["TRAINING"]["LAMBDA_RESOURCE_LOSS"]
                    print(f"\nResource Loss Calculation:")
                    print(f"   Expected Cost: {expected_path_cost_sample.item():.4f}s")
                    print(f"   Lambda Resource: {lambda_resource}")
                    print(
                        f"   Resource Loss = {expected_path_cost_sample.item():.4f} × {lambda_resource} = {sample_resource_loss.item():.6f}"
                    )
                    print(f"\nLoss Components:")
                    print(
                        f"   Task Loss (KL divergence): {sample_task_loss.item():.6f}"
                    )
                    print(
                        f"   Resource Loss (cost penalty): {sample_resource_loss.item():.6f}"
                    )
                    print(
                        f"   Total Loss: {(sample_task_loss + sample_resource_loss).item():.6f}"
                    )
                    print(
                        f"   Resource/Task Ratio: {sample_resource_loss.item() / max(sample_task_loss.item(), 1e-08):.2f}"
                    )
                    best_performance_idx = torch.argmax(scores_tensor_sample).item()
                    best_cost_idx = torch.argmin(self.path_costs_tensor).item()
                    gate_preferred_idx = torch.argmax(
                        gate_pred_probs_for_resource_loss[i]
                    ).item()
                    print(f"\nTrade-off Analysis:")
                    print(
                        f"   Best Performance: {self.path_names[best_performance_idx]} (score: {path_scores_sample[best_performance_idx]:.4f})"
                    )
                    print(
                        f"   Lowest Cost: {self.path_names[best_cost_idx]} (cost: {path_costs_cpu[best_cost_idx]:.2f}s)"
                    )
                    print(
                        f"   Gate Prefers: {self.path_names[gate_preferred_idx]} (prob: {current_gate_probs[gate_preferred_idx]:.4f})"
                    )
                    if gate_preferred_idx == best_performance_idx:
                        print("   Gate chooses best performer")
                    elif gate_preferred_idx == best_cost_idx:
                        print("   Gate chooses cheapest option")
                    else:
                        print("   Gate balances performance vs cost")
                    print("=" + "=" * 88 + "=\n")
                total_task_loss += sample_task_loss
                total_resource_loss += sample_resource_loss
            task_loss = total_task_loss / batch_size
            resource_loss = total_resource_loss / batch_size
            total_loss_for_gate = task_loss + resource_loss
            gate_probs_for_selection_output = gate_pred_probs_for_resource_loss
        else:
            gate_probs_for_selection_output = F.softmax(gate_logits, dim=-1)
            selected_indices_hard = torch.argmax(gate_logits, dim=1).tolist()
            if DEBUG_EXPERT_SELECTION:
                print(f"\nINFERENCE - EXPERT SELECTION ANALYSIS")
                print(
                    f"Gate probabilities: {gate_probs_for_selection_output.cpu().numpy()}"
                )
                print(
                    f"Selected experts: {[self.path_names[idx] for idx in selected_indices_hard]}"
                )
                for i in range(min(batch_size, 3)):
                    probs = gate_probs_for_selection_output[i].cpu().numpy()
                    selected_idx = selected_indices_hard[i]
                    selected_name = self.path_names[selected_idx]
                    print(f"\nSample {i + 1} Decision Analysis:")
                    print(f"   Question: {original_questions_batch[i][:80]}...")
                    print(f"   Expert Probabilities:")
                    for j, (name, prob) in enumerate(zip(self.path_names, probs)):
                        marker = "*" if j == selected_idx else "  "
                        cost = self.path_costs_tensor[j].item()
                        print(f"     {marker} {name}: {prob:.4f} (cost: {cost:.2f}s)")
                    print(
                        f"   Selected: {selected_name} (confidence: {probs[selected_idx]:.1%})"
                    )
                    if selected_idx == torch.argmin(self.path_costs_tensor).item():
                        print(f"   Reason: Chose cheapest expert")
                    elif probs[selected_idx] > 0.8:
                        print(f"   Reason: High confidence selection")
                    else:
                        print(f"   Reason: Balanced cost-performance trade-off")
            else:
                print(
                    f"INFO: Gate probabilities: {gate_probs_for_selection_output.cpu().numpy()}"
                )
                print(
                    f"INFO: Selected experts: {[self.path_names[idx] for idx in selected_indices_hard]}"
                )
            for i in range(batch_size):
                selected_idx = selected_indices_hard[i]
                current_text_intermediate_state = {
                    k: v[i : i + 1].clone() for k, v in text_intermediate_state.items()
                }
                current_vlm_intermediate_state = {}
                if (
                    hasattr(self.expert_vlm, "__class__")
                    and "Ovis2" in self.expert_vlm.__class__.__name__
                ):
                    for k, v in vlm_intermediate_state.items():
                        if isinstance(v, list) and len(v) > i:
                            current_vlm_intermediate_state[k] = v[i]
                        else:
                            print(f"WARN: Ovis2 state {k} missing for sample {i}")
                            current_vlm_intermediate_state[k] = None
                else:
                    for k, v in vlm_intermediate_state.items():
                        current_vlm_intermediate_state[k] = v[i : i + 1].clone()
                generated_text = "[Error: Path Gen Failed]"
                try:
                    if selected_idx == 0:
                        text_output = self.expert_text.continue_generation_from_state(
                            current_text_intermediate_state, **generation_kwargs
                        )
                        all_generated_texts_for_return[i] = (
                            text_output[0]
                            if isinstance(text_output, list)
                            else text_output
                        )
                    elif selected_idx == 1:
                        vlm_output = self.expert_vlm.continue_generation_from_state(
                            current_vlm_intermediate_state, **generation_kwargs
                        )
                        all_generated_texts_for_return[i] = (
                            vlm_output[0]
                            if isinstance(vlm_output, list)
                            else vlm_output
                        )
                    elif (
                        self.use_late_fusion
                        and selected_idx == 2
                        and self.fusion_expert_api
                    ):
                        with torch.no_grad():
                            e1_text_list = (
                                self.expert_text.continue_generation_from_state(
                                    current_text_intermediate_state, **generation_kwargs
                                )
                            )
                            e1_text = (
                                e1_text_list[0]
                                if isinstance(e1_text_list, list)
                                else e1_text_list
                            )
                            e2_text_list = (
                                self.expert_vlm.continue_generation_from_state(
                                    current_vlm_intermediate_state, **generation_kwargs
                                )
                            )
                            e2_text = (
                                e2_text_list[0]
                                if isinstance(e2_text_list, list)
                                else e2_text_list
                            )
                        current_category = (
                            categories_batch[i]
                            if categories_batch is not None
                            else None
                        )
                        all_generated_texts_for_return[
                            i
                        ] = self.fusion_expert_api.generate_full(
                            original_questions_batch[i],
                            raw_table_batch[i],
                            e1_text,
                            e2_text,
                            dataset_category=current_category,
                            **generation_kwargs,
                        )
                    else:
                        all_generated_texts_for_return[i] = generated_text
                except Exception as e:
                    print(f"ERROR gen text item {i}, idx {selected_idx}: {e}")
                    all_generated_texts_for_return[i] = f"[Gen Error: {e}]"
        return {
            "generated_text": all_generated_texts_for_return,
            "gate_logits": gate_logits,
            "gate_probabilities_for_selection": gate_probs_for_selection_output,
            "selected_indices": torch.argmax(
                gate_probs_for_selection_output, dim=1
            ).tolist()
            if gate_probs_for_selection_output is not None
            else [0] * batch_size,
            "task_loss_for_gate": task_loss,
            "resource_loss": resource_loss,
            "total_loss_for_gate": total_loss_for_gate,
        }


def _extract_json_answer_from_text(text_content):
    if not text_content or not isinstance(text_content, str):
        return None
    json_string = None
    match_md = re.search(
        "```json\\s*(\\{.*?\\})\\s*```", text_content, re.DOTALL | re.IGNORECASE
    )
    if match_md:
        json_string = match_md.group(1).strip()
    else:
        match_direct = re.search("(\\{.*?\\})(?:\\s|$)", text_content, re.DOTALL)
    if match_direct:
        json_string = match_direct.group(1).strip()
    if json_string:
        try:
            parsed_json = json.loads(json_string)
            answer_value = parsed_json.get("answer")
            if isinstance(answer_value, list):
                return json.dumps(answer_value)
            elif answer_value is not None:
                return str(answer_value)
            return None
        except json.JSONDecodeError:
            pass
    ans_marker = "Answer:"
    idx = text_content.rfind(ans_marker)
    if idx != -1:
        return text_content[idx + len(ans_marker) :].strip()
    return text_content
