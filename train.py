import os
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, ".env")
loaded_env = load_dotenv(dotenv_path=dotenv_path)
print(f"INFO (train.py): .env loaded: {loaded_env}")
hf_token_check = os.getenv("HF_TOKEN")
print(
    f"INFO (train.py): HF_TOKEN value after explicit load: {('Set' if hf_token_check else 'Not Set')}"
)
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import math
import json
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, get_scheduler

DEBUG_COST_CALCULATION = False
DEBUG_BATCH_LIMIT = 3
from models.dynamic_selector import DynamicExpertSelector
from project_config.config import cfg
from data.dataloader import (
    create_data_loader,
    create_dummy_data,
    create_dummy_images,
    ImageNotFoundError,
)
from evaluate import evaluate

skipped_data_log = []


def custom_collate_fn(batch_list):
    valid_batch = [item for item in batch_list if item is not None]
    if not valid_batch:
        return None
    batched = {}
    keys_to_list = [
        "raw_table",
        "question",
        "prompt_for_text_expert",
        "prompt_for_vlm_expert",
        "image_path_for_vlm",
        "category",
        "target_text_str",
        "question_id",
        "image",
    ]
    keys_to_stack = ["target_ids", "target_attention_mask"]
    for key in keys_to_list:
        batched[key] = [item[key] for item in valid_batch]
    for key in keys_to_stack:
        batched[key] = torch.stack([item[key] for item in valid_batch])
    return batched


def create_data_loader_with_error_handling(
    data_path,
    batch_size,
    shuffle,
    num_workers,
    target_tokenizer_name,
    max_seq_len_target,
    table_image_dir,
    filter_dataset_name=None,
):
    dataset = create_data_loader(
        data_path,
        batch_size,
        shuffle,
        num_workers,
        target_tokenizer_name,
        max_seq_len_target,
        table_image_dir,
        filter_dataset_name,
        return_dataset=True,
    )
    if dataset is None:
        return None
    original_getitem = dataset.__getitem__

    def safe_getitem(idx):
        try:
            return original_getitem(idx)
        except ImageNotFoundError as e:
            global skipped_data_log
            skipped_data_log.append(
                {
                    "question_id": e.question_id,
                    "image": e.image_file,
                    "image_path": e.image_path,
                    "error_message": e.message,
                }
            )
            print(f"WARN: Skipping data point: {e.message}")
            return None
        except Exception as e:
            print(f"ERROR in __getitem__ at idx {idx}: {e}")
            return None

    dataset.__getitem__ = safe_getitem
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available() and num_workers > 0,
    )
    return dataloader


def train_gate_online_scoring_epoch(
    model, dataloader, optimizer_gate, device, epoch_num_logging, lr_scheduler=None
):
    model.train()
    model.gating_network.train()
    for expert_module in model.base_experts:
        for param in expert_module.parameters():
            param.requires_grad = False
    if model.fusion_expert_api:
        pass
    total_loss_accum, task_loss_accum, resource_loss_accum = (0.0, 0.0, 0.0)
    epoch_path_counts = {i: 0 for i in range(model.num_paths)}
    num_batches_processed = 0
    skipped_batches = 0
    dataset_size = len(dataloader.dataset)
    batch_loss_history = []
    gate_logits_temp = cfg["TRAINING"]["GATE_LOGITS_TEMPERATURE"]
    score_temp = cfg["TRAINING"]["GATE_TARGET_SCORE_TEMP"]
    accumulation_steps = cfg["TRAINING"].get("GRADIENT_ACCUMULATION_STEPS", 1)
    optimizer_gate.zero_grad()
    progress_bar = tqdm(
        dataloader,
        desc=f"OnlineScore Epoch {epoch_num_logging + 1} GateTemp={gate_logits_temp:.2f} ScoreTemp={score_temp:.2f}",
        leave=False,
    )
    for i, batch in enumerate(progress_bar):
        if batch is None:
            skipped_batches += 1
            continue
        try:
            output_dict = model(
                raw_table_batch=batch["raw_table"],
                original_questions_batch=batch["question"],
                prompts_text_batch=batch["prompt_for_text_expert"],
                prompts_vlm_batch=batch["prompt_for_vlm_expert"],
                image_paths_vlm_batch=batch["image_path_for_vlm"],
                target_texts_str_batch=batch["target_text_str"],
                categories_batch=batch["category"],
                is_training=True,
                **cfg["GENERATION"],
            )
            task_loss = output_dict["task_loss_for_gate"]
            resource_loss_value = output_dict["resource_loss"]
            total_loss = output_dict["total_loss_for_gate"]
            if output_dict["gate_probabilities_for_selection"] is not None:
                with torch.no_grad():
                    selected_indices_batch = output_dict["selected_indices"]
                    for sel_idx in selected_indices_batch:
                        epoch_path_counts[sel_idx] = (
                            epoch_path_counts.get(sel_idx, 0) + 1
                        )
            if total_loss is None or torch.isnan(total_loss):
                print(f"WARN: NaN/None total_loss batch {i}. Skipping.")
                continue
            if accumulation_steps > 1:
                total_loss = total_loss / accumulation_steps
            total_loss.backward()
            if (i + 1) % accumulation_steps == 0 or i + 1 == len(dataloader):
                if cfg["TRAINING"]["GRAD_CLIP_VALUE"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.gating_network.parameters(),
                        cfg["TRAINING"]["GRAD_CLIP_VALUE"],
                    )
                optimizer_gate.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer_gate.zero_grad()
            batch_loss_history.append(
                total_loss.item() * accumulation_steps
                if accumulation_steps > 1
                else total_loss.item()
            )
            total_loss_accum += (
                total_loss.item() * accumulation_steps
                if accumulation_steps > 1
                else total_loss.item()
            )
            if task_loss is not None:
                task_loss_accum += task_loss.item()
            if resource_loss_value is not None:
                resource_loss_accum += resource_loss_value.item()
            num_batches_processed += 1
            if (i + 1) % cfg["TRAINING"]["LOG_INTERVAL"] == 0 or i == len(
                dataloader
            ) - 1:
                if num_batches_processed > 0:
                    avg_total_loss = total_loss_accum / num_batches_processed
                    avg_task_loss = (
                        task_loss_accum / num_batches_processed
                        if task_loss is not None
                        else 0
                    )
                    avg_res_loss = (
                        resource_loss_accum / num_batches_processed
                        if resource_loss_value is not None
                        else 0
                    )
                    current_lr = optimizer_gate.param_groups[0]["lr"]
                    progress_bar.set_postfix(
                        {
                            "Avg Loss": f"{avg_total_loss:.4f}",
                            "KLDiv L": f"{avg_task_loss:.4f}",
                            "Res L": f"{avg_res_loss:.4f}",
                            "LR": f"{current_lr:.2e}",
                            "Skipped": skipped_batches,
                        }
                    )
        except Exception as e:
            print(f"\nERROR train batch {i}: {e}")
            import traceback

            traceback.print_exc()
    if num_batches_processed == 0:
        return (float("inf"), epoch_path_counts, 0, 0, [])
    final_total_loss = total_loss_accum / num_batches_processed
    avg_task_loss_final = (
        task_loss_accum / num_batches_processed if task_loss_accum > 0 else 0
    )
    avg_resource_loss_final = (
        resource_loss_accum / num_batches_processed if resource_loss_accum > 0 else 0
    )
    print(
        f"\nEpoch {epoch_num_logging + 1} Summary - Loss: {final_total_loss:.4f} (Task: {avg_task_loss_final:.4f}, Resource: {avg_resource_loss_final:.4f})"
    )
    total_selections = sum(epoch_path_counts.values())
    if total_selections > 0:
        path_names = ["TextExpert", "VLMExpert", "Fusion"][: len(epoch_path_counts)]
        selection_stats = []
        for i, (path_idx, count) in enumerate(epoch_path_counts.items()):
            percentage = count / total_selections * 100
            expert_name = (
                path_names[path_idx]
                if path_idx < len(path_names)
                else f"Path{path_idx}"
            )
            selection_stats.append(f"{expert_name}: {count}({percentage:.1f}%)")
        print(f"Expert Selection: {', '.join(selection_stats)}")
    processed_count = num_batches_processed
    skipped_count = skipped_batches
    total_count = processed_count + skipped_count
    skip_percent = skipped_count / total_count * 100 if total_count > 0 else 0
    print(
        f"Epoch {epoch_num_logging + 1} Train Summary: AvgLoss={final_total_loss:.4f}, PathSel (from gate's soft pred): {epoch_path_counts}"
    )
    print(
        f"Training Statistics: Processed: {processed_count}, Skipped: {skipped_count}, Total: {total_count}, Skip%: {skip_percent:.2f}%"
    )
    if skipped_data_log:
        skipped_log_path = os.path.join(
            cfg["TRAINING"]["CHECKPOINT_DIR"],
            f"skipped_data_epoch_{epoch_num_logging + 1}.json",
        )
        with open(skipped_log_path, "w") as f:
            json.dump(skipped_data_log, f, indent=2)
        print(
            f"Saved log of {len(skipped_data_log)} skipped data points to {skipped_log_path}"
        )
        sample_size = min(5, len(skipped_data_log))
        skipped_qids = [item["question_id"] for item in skipped_data_log[:sample_size]]
        print(
            f"Sample of skipped question_ids: {skipped_qids}"
            + (
                f" ... and {len(skipped_data_log) - sample_size} more"
                if len(skipped_data_log) > sample_size
                else ""
            )
        )
    return (
        final_total_loss,
        epoch_path_counts,
        num_batches_processed,
        skipped_batches,
        batch_loss_history,
    )


def plot_intra_epoch_loss(batch_losses, epoch_num, phase_name, checkpoint_dir):
    if not batch_losses:
        print(f"WARN: No batch losses to plot for epoch {epoch_num + 1}.")
        return
    plt.figure(figsize=(15, 7))
    plt.plot(batch_losses, label="Per-Batch Training Loss", alpha=0.6)
    if len(batch_losses) >= 20:
        moving_avg = pd.Series(batch_losses).rolling(window=20, min_periods=5).mean()
        plt.plot(
            moving_avg,
            label="20-Batch Moving Average",
            color="orange",
            linewidth=2,
            linestyle="--",
        )
    plt.xlabel("Training Step (Batch) in Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Intra-Epoch Training Loss (Phase: {phase_name}, Global Epoch: {epoch_num + 1})"
    )
    plt.legend()
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    if len(batch_losses) > 1:
        loss_series = pd.Series(batch_losses)
        q1 = loss_series.quantile(0.05)
        q3 = loss_series.quantile(0.95)
        iqr = q3 - q1
        plt.ylim(max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"training_loss_epoch_{epoch_num + 1}.png")
    try:
        plt.savefig(save_path, dpi=150)
        print(f"Intra-epoch loss plot saved to {save_path}")
    except Exception as e:
        print(f"ERROR: Failed to save intra-epoch loss plot to {save_path}: {e}")
    plt.close()


def plot_training_progress(
    train_loss_history, val_metric_history, val_epochs, checkpoint_dir
):
    if not train_loss_history:
        print("WARN: No training history to plot.")
        return
    epochs = range(1, len(train_loss_history) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = "tab:red"
    ax1.set_xlabel("Global Epochs")
    ax1.set_ylabel("Training Loss", color=color)
    ax1.plot(
        epochs,
        train_loss_history,
        color=color,
        marker="o",
        linestyle="-",
        label="Training Loss",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_title("Training Progress: Loss and Validation Accuracy")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if val_metric_history and val_epochs:
        ax2 = ax1.twinx()
        color = "tab:blue"
        val_metric_name = "Validation Accuracy"
        ax2.set_ylabel(val_metric_name, color=color)
        ax2.plot(
            val_epochs,
            val_metric_history,
            color=color,
            marker="s",
            linestyle="--",
            label=val_metric_name,
        )
        ax2.tick_params(axis="y", labelcolor=color)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    lines, labels = ax1.get_legend_handles_labels()
    if val_metric_history and val_epochs:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax1.legend(loc="best")
    plt.grid(True)
    save_path = os.path.join(checkpoint_dir, "training_progress.png")
    try:
        plt.savefig(save_path)
        print(f"Training progress plot saved to {save_path}")
    except Exception as e:
        print(f"ERROR: Failed to save plot to {save_path}: {e}")
    plt.close(fig)


def main():
    device = torch.device(cfg["TRAINING"]["DEVICE"])
    print(f"Using device: {device}")
    for data_key in ["TRAIN_PATH", "VAL_PATH"]:
        path = cfg["DATA"].get(data_key)
        if not path:
            if data_key == "VAL_PATH":
                print("INFO: VAL_PATH is not set. Skipping validation.")
                continue
            else:
                print(f"ERROR: Required data path {data_key} is not set in config.")
                sys.exit(1)
        if not os.path.exists(path):
            print(f"ERROR: Data file {path} not found; update {data_key} in config.")
            sys.exit(1)
    img_dir = cfg["DATA"].get("TABLE_IMAGE_DIR")
    if img_dir and (not os.path.exists(img_dir)):
        print(f"ERROR: Table image directory {img_dir} not found.")
        sys.exit(1)
    target_tokenizer_name_for_dataloader = cfg["MODEL"]["TEXT_EXPERT_ID"]
    try:
        _ = AutoTokenizer.from_pretrained(target_tokenizer_name_for_dataloader)
    except Exception:
        print(
            f"WARN: Could not load tokenizer '{target_tokenizer_name_for_dataloader}'. Defaulting to 'gpt2' for target tokenization."
        )
        target_tokenizer_name_for_dataloader = "gpt2"
    val_loader = None
    if cfg["DATA"].get("VAL_PATH"):
        print(f"\n--- Creating Validation Loader ---")
        val_loader = create_data_loader_with_error_handling(
            cfg["DATA"]["VAL_PATH"],
            batch_size=cfg["EVALUATION"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=0,
            target_tokenizer_name=target_tokenizer_name_for_dataloader,
            max_seq_len_target=cfg["DATA"]["MAX_SEQ_LEN_LM_TARGET"],
            table_image_dir=cfg["DATA"]["TABLE_IMAGE_DIR"],
        )
        if not val_loader:
            print("WARN: VAL_PATH provided but DataLoader creation failed.")
    print("\n--- Initializing Model ---")
    try:
        model = DynamicExpertSelector(
            gate_hidden_dim=cfg["MODEL"]["GATE_HIDDEN_DIM"],
            use_late_fusion=cfg["MODEL"]["USE_LATE_FUSION"],
        )
    except Exception as e:
        print(f"CRITICAL ERROR initializing model: {e}")
        return
    model.to(device)
    print(
        f"\nCost Configuration - Lambda: {cfg['TRAINING']['LAMBDA_RESOURCE_LOSS']}, Expert Costs: {[f'{name}:{cost:.2f}s' for name, cost in zip(model.path_names, model.path_costs_tensor.cpu().tolist())]}"
    )
    print("\n--- Setting up Gate Optimizer ---")
    optimizer_gate = optim.AdamW(
        model.gating_network.parameters(),
        lr=cfg["TRAINING"]["LEARNING_RATE_GATE"],
        weight_decay=cfg["TRAINING"]["WEIGHT_DECAY"],
    )
    checkpoint_dir = cfg["TRAINING"]["CHECKPOINT_DIR"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    training_phases_config = cfg["TRAINING"].get("TRAINING_PHASES", [])
    if not training_phases_config:
        print(
            "WARN: TRAINING_PHASES not defined. Defaulting to full TRAIN_PATH for TRAINING['EPOCHS'] epochs."
        )
        total_epochs_for_full_dataset = cfg["TRAINING"].get("EPOCHS", 3)
        training_phases_config = [
            {"dataset_name_filter": None, "epochs": total_epochs_for_full_dataset}
        ]
    total_update_steps = 0
    accumulation_steps = cfg["TRAINING"].get("GRADIENT_ACCUMULATION_STEPS", 1)
    if training_phases_config:
        print("Calculating total training steps for LR scheduler...")
        for phase_config in training_phases_config:
            dataset_name_filter = phase_config.get("dataset_name_filter")
            num_epochs_for_phase = phase_config.get("epochs", 1)
            temp_dataset = create_data_loader(
                cfg["DATA"]["TRAIN_PATH"],
                batch_size=cfg["TRAINING"]["BATCH_SIZE"],
                shuffle=False,
                num_workers=0,
                target_tokenizer_name=target_tokenizer_name_for_dataloader,
                max_seq_len_target=cfg["DATA"]["MAX_SEQ_LEN_LM_TARGET"],
                table_image_dir=img_dir,
                filter_dataset_name=dataset_name_filter,
                return_dataset=True,
            )
            if temp_dataset and len(temp_dataset) > 0:
                num_batches_per_epoch = math.ceil(
                    len(temp_dataset) / cfg["TRAINING"]["BATCH_SIZE"]
                )
                num_updates_per_epoch = math.ceil(
                    num_batches_per_epoch / accumulation_steps
                )
                total_update_steps += num_updates_per_epoch * num_epochs_for_phase
                print(
                    f"  - Phase '{dataset_name_filter or 'Pre-generated Mixed Dataset'}': {len(temp_dataset)} samples, {num_updates_per_epoch} updates/epoch * {num_epochs_for_phase} epochs = {num_updates_per_epoch * num_epochs_for_phase} updates"
                )
            else:
                print(
                    f"  - WARNING: Could not load or empty dataset for phase '{dataset_name_filter or 'Pre-generated Mixed Dataset'}'. It will not be counted for LR scheduler."
                )
    lr_scheduler = None
    if total_update_steps > 0:
        warmup_steps = int(
            cfg["TRAINING"].get("LR_WARMUP_RATIO", 0.05) * total_update_steps
        )
        lr_scheduler = get_scheduler(
            name=cfg["TRAINING"].get("LR_SCHEDULER_TYPE", "linear"),
            optimizer=optimizer_gate,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )
        print(
            f"LR Scheduler enabled: {cfg['TRAINING'].get('LR_SCHEDULER_TYPE', 'linear')} with {warmup_steps} warmup steps over {total_update_steps} total updates."
        )
    else:
        print("No training steps calculated. LR Scheduler will not be used.")
    global_epoch_counter = 0
    resume_checkpoint_path = cfg["TRAINING"].get("RESUME_CHECKPOINT_PATH")
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"\n--- Resuming Training from Checkpoint: {resume_checkpoint_path} ---")
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.gating_network.load_state_dict(checkpoint["model_state_dict"])
                print("INFO: Gating network state loaded from checkpoint.")
            else:
                print(
                    "WARN: 'model_state_dict' not found in checkpoint. Gating network weights not loaded."
                )
            if "optimizer_gate_state_dict" in checkpoint:
                optimizer_gate.load_state_dict(checkpoint["optimizer_gate_state_dict"])
                print("INFO: Optimizer state loaded from checkpoint.")
            else:
                print(
                    "WARN: 'optimizer_gate_state_dict' not found in checkpoint. Optimizer state not reloaded."
                )
            global_epoch_counter = checkpoint.get("epoch", 0)
            print(f"INFO: Resuming from global epoch {global_epoch_counter}.")
            if lr_scheduler and "lr_scheduler_state_dict" in checkpoint:
                try:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                    print("INFO: LR Scheduler state loaded from checkpoint.")
                except Exception as e:
                    print(f"WARN: Could not load LR scheduler state: {e}")
        except Exception as e:
            print(
                f"ERROR: Failed to load checkpoint '{resume_checkpoint_path}'. Starting from scratch. Error: {e}"
            )
            global_epoch_counter = 0
    elif resume_checkpoint_path:
        print(
            f"WARN: resume_checkpoint_path was set to '{resume_checkpoint_path}' but file not found. Starting from scratch."
        )
    else:
        print("\n--- Starting Training from Scratch ---")
    train_loss_history = []
    val_metric_history = []
    val_epochs = []
    best_val_metric = -float("inf")
    global skipped_data_log
    total_processed = 0
    total_skipped = 0
    print("\n--- Starting Training By Dataset Phase Specified in Config ---")
    for phase_idx, phase_config in enumerate(training_phases_config):
        dataset_name_filter = phase_config.get("dataset_name_filter")
        num_epochs_for_phase = phase_config.get("epochs", 3)
        phase_name_log = dataset_name_filter or "Pre-generated Mixed Dataset"
        print(
            f"\n===== Training Phase {phase_idx + 1}: {phase_name_log}, Epochs: {num_epochs_for_phase} ====="
        )
        current_phase_train_loader = create_data_loader_with_error_handling(
            cfg["DATA"]["TRAIN_PATH"],
            batch_size=cfg["TRAINING"]["BATCH_SIZE"],
            shuffle=True,
            num_workers=0,
            target_tokenizer_name=target_tokenizer_name_for_dataloader,
            max_seq_len_target=cfg["DATA"]["MAX_SEQ_LEN_LM_TARGET"],
            table_image_dir=img_dir,
            filter_dataset_name=dataset_name_filter,
        )
        if not current_phase_train_loader:
            print(f"WARN: No data for phase '{phase_name_log}'. Skipping.")
            continue
        skipped_data_log = []
        for epoch_in_phase in range(num_epochs_for_phase):
            print(
                f"\n--- Phase '{phase_name_log}' - Epoch {epoch_in_phase + 1}/{num_epochs_for_phase} (Global Epoch {global_epoch_counter + 1}) ---"
            )
            (
                epoch_train_loss,
                _,
                processed_in_epoch,
                skipped_in_epoch,
                batch_losses,
            ) = train_gate_online_scoring_epoch(
                model,
                current_phase_train_loader,
                optimizer_gate,
                device,
                global_epoch_counter,
                lr_scheduler,
            )
            train_loss_history.append(epoch_train_loss)
            plot_intra_epoch_loss(
                batch_losses, global_epoch_counter, phase_name_log, checkpoint_dir
            )
            total_processed += processed_in_epoch
            total_skipped += skipped_in_epoch
            if val_loader:
                print(
                    f"\n--- Validating Gate after Global Epoch {global_epoch_counter + 1} ---"
                )
                val_results = evaluate(
                    model,
                    val_loader,
                    device,
                    cfg["GENERATION"],
                    epoch_num=global_epoch_counter + 1,
                    silent=False,
                )
                primary_metric_val = val_results.get("Overall_Accuracy", -1.0)
                if primary_metric_val < 0:
                    print(
                        f"WARN: Could not find 'Overall_Accuracy' in validation results. Available keys: {list(val_results.keys())}. Cannot determine best model."
                    )
                else:
                    print(f"Validation Accuracy: {primary_metric_val:.4f}")
                    val_metric_history.append(primary_metric_val)
                    val_epochs.append(global_epoch_counter + 1)
                    if primary_metric_val > best_val_metric:
                        best_val_metric = primary_metric_val
                        print(
                            f"*** New best validation accuracy: {best_val_metric:.4f}. Saving best model. ***"
                        )
                        best_model_path = os.path.join(
                            checkpoint_dir, "best_model_gate.pth"
                        )
                        save_content_for_best = {
                            "epoch": global_epoch_counter + 1,
                            "model_state_dict": model.gating_network.state_dict(),
                            "optimizer_gate_state_dict": optimizer_gate.state_dict(),
                            "lr_scheduler_state_dict": lr_scheduler.state_dict()
                            if lr_scheduler
                            else None,
                            "train_loss": epoch_train_loss,
                            "val_accuracy": best_val_metric,
                            "gate_input_dim": model.gate_input_dim,
                            "config": cfg,
                            "path_costs": model.path_costs_tensor.cpu().tolist(),
                        }
                        torch.save(save_content_for_best, best_model_path)
                        print(f"Best model checkpoint saved to {best_model_path}")
            else:
                print(
                    "INFO: No validation loader, skipping per-epoch validation and best model saving."
                )
            latest_model_path = os.path.join(checkpoint_dir, f"latest_model_gate.pth")
            torch.save(
                {
                    "epoch": global_epoch_counter + 1,
                    "model_state_dict": model.gating_network.state_dict(),
                    "optimizer_gate_state_dict": optimizer_gate.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict()
                    if lr_scheduler
                    else None,
                    "gate_input_dim": model.gate_input_dim,
                },
                latest_model_path,
            )
            print(f"Latest model checkpoint saved to {latest_model_path}")
            global_epoch_counter += 1
    print("\n--- Training Complete ---")
    print(
        f"Training data summary: processed {total_processed}, skipped {total_skipped}, total {total_processed + total_skipped}"
    )
    final_save_path = os.path.join(
        checkpoint_dir, f"final_model_epoch{global_epoch_counter}_gate.pth"
    )
    torch.save(
        {
            "epoch": global_epoch_counter,
            "model_state_dict": model.gating_network.state_dict(),
            "optimizer_gate_state_dict": optimizer_gate.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict()
            if lr_scheduler
            else None,
            "gate_input_dim": model.gate_input_dim,
            "config": cfg,
            "path_costs": model.path_costs_tensor.cpu().tolist(),
        },
        final_save_path,
    )
    print(f"Final model state saved to {final_save_path}")
    plot_training_progress(
        train_loss_history, val_metric_history, val_epochs, checkpoint_dir
    )


if __name__ == "__main__":
    main()
