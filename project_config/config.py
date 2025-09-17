import os
from pathlib import Path
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = "checkpoints/LAMBDA_RESOURCE_LOSS_0.15/best_model_gate.pth"
DATA = {
    "TRAIN_PATH": DATA_DIR / "your_data_path" / "train.jsonl",
    "VAL_PATH": DATA_DIR / "your_data_path" / "val.jsonl",
    "TEST_PATH": DATA_DIR / "your_data_path" / "test.jsonl",
    "MAX_SEQ_LEN_LM_TARGET": 128,
    "TABLE_IMAGE_DIR": DATA_DIR / "your_data_path" / "table_images" / "train",
    "TEST_TABLE_IMAGE_DIR": DATA_DIR / "your_data_path" / "table_images" / "test",
    "INFERENCE_OUTPUT_DIR": OUTPUT_DIR / "main",
    "INFERENCE_DATASET_FILTERS": [
        "WTQ",
        "TabFact",
        "InfoTabs",
        "TABMWP",
        "TAT-QA",
        "HiTab",
        "FeTaQA",
    ],
    "MIXED_DATASETS": [
        "WTQ",
        "TabFact",
        "InfoTabs",
        "TABMWP",
        "TAT-QA",
        "HiTab",
        "FeTaQA",
    ],
    "SAMPLES_PER_DATASET": 2000,
}
MODEL = {
    "USE_LATE_FUSION": True,
    "GATE_HIDDEN_DIM": 256,
    "TEXT_EXPERT_ID": "tablegpt/TableGPT2-7B",
    "VLM_EXPERT_ID": "AIDC-AI/Ovis2-8B",
    "VLM_EXPERT_TYPE": "Ovis2",
    "FUSION_EXPERT_MODEL_ID": "gemini-2.0-flash",
    "FUSION_MAX_NEW_TOKENS": 512,
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "GPT2_GATE_FEATURE_LAYERS": 0,
    "QUESTION_EMBED_MODEL_FOR_GATE_ID": "sentence-transformers/all-MiniLM-L6-v2",
    "EXPERT_COSTS": {"TextExpert": 0.73, "VLMExpert": 0.81, "Fusion": 0.96},
}
MODEL["NUM_PATHS"] = 3 if MODEL["USE_LATE_FUSION"] else 2
TRAINING = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EPOCHS": 1,
    "EPOCHS_PER_DATASET": 1,
    "BATCH_SIZE": 8,
    "GRADIENT_ACCUMULATION_STEPS": 4,
    "LEARNING_RATE_GATE": 0.0001,
    "LR_SCHEDULER_TYPE": "cosine",
    "LR_WARMUP_RATIO": 0.05,
    "OPTIMIZER": "AdamW",
    "WEIGHT_DECAY": 0.01,
    "GATE_LOGITS_TEMPERATURE": 1.0,
    "GATE_TARGET_SCORE_TEMP": 0.3,
    "LAMBDA_RESOURCE_LOSS": 0.15,
    "FREEZE_EXPERTS": True,
    "TRAINING_PHASES": [
        {
            "dataset_name_filter": None,
            "epochs": 1,
            "metric_for_gate_training": "Accuracy",
            "description": "Mixed dataset training",
        }
    ],
    "DEFAULT_TRAINING_METRIC": "Accuracy",
    "CHECKPOINT_DIR": CHECKPOINT_DIR / "Mixed_Dataset_Training",
    "LOG_INTERVAL": 50,
    "GRAD_CLIP_VALUE": 1.0,
    "SAVE_AFTER_EACH_DATASET_PHASE": True,
    "RESUME_CHECKPOINT_PATH": None,
    "INFERENCE_CHECKPOINT": CHECKPOINT_DIR
    / "Mixed_Dataset_Training"
    / "best_model_gate.pth",
}
GENERATION = {
    "MAX_NEW_TOKENS": 64,
    "NUM_BEAMS": 1,
    "TEMPERATURE": 0.0,
    "DO_SAMPLE": False,
    "TOP_K": 1,
    "TOP_P": 1.0,
    "REPETITION_PENALTY": 1.2,
    "LENGTH_PENALTY": 1.2,
}
EVALUATION = {"BATCH_SIZE": 1, "METRICS": ["BLEU", "ROUGE-L"]}
cfg = {
    "DATA": DATA,
    "MODEL": MODEL,
    "TRAINING": TRAINING,
    "GENERATION": GENERATION,
    "EVALUATION": EVALUATION,
}
cfg["DATA"]["GATE_INPUT_DIM"] = None
