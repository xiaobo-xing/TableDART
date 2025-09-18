# TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding

## 1. Environment Setup
**Install Dependencies**
- `conda create -n tabledart python=3.10`
- `conda activate tabledart`
- `pip install -r requirements.txt`

**Configuration**
- Copy the provided credential template: `cp env.txt .env`, then fill in keys such as `GEMINI_API_KEY` and `HF_TOKEN`.
- Edit `project_config/config.py` such as dataset paths, table image directories, output folders, API models, and checkpoints，etc.

## 2. Data Preparation
1. **Data Download.** Download the publicly available MMTab dataset. For a direct and fair comparison, our work uses the same training and test datasets as established in the prior works, such as HIPPO and Table-LLaVA.
  

2. **Create mixed training/validation splits**. Regenerate the mixed data with:
   ```bash
   python data/create_mixed_dataset.py \
     --input_file data/your_data_path/train_data.jsonl \
     --output_dir data/your_data_path \
     --samples_per_dataset 2000 \
     --val_ratio 0.15
   ```
   - The script pulls dataset names from `cfg["DATA"]["MIXED_DATASETS"]` and writes `mixed_train.jsonl`, `mixed_val.jsonl`, and `dataset_summary.json` under `data/processed_datasets/`
   - Sampling uses a fixed random seed (42) in code for reproducibility. 
  
    After generate or directly use our mixed dataset, point `TRAIN_PATH` and `VAL_PATH` in `project_config/config.py` to the generated files.

## 3. Cost Vector Measurement 
Measure the cost vector before training, our measured result is provided at `cost_measurement/expert_costs.json`. Or you can execute via:
  ```bash
  python cost_measurement/measure_expert_costs.py \
    --save_results \
    --output_file cost_measurement/expert_costs.json
  ```
  - The script loads the test split and table images configured in `project_config/config.py`, benchmarks each model, logs latency/TTFT/throughput statistics, and writes `cost_measurement/expert_costs.json`. 
  - Copy the reported values into `cfg["MODEL"]["EXPERT_COSTS"]` for training and inference.

## 4. Training
Start to train the gating network by running: `python train.py`
  - Checkpoints and plots will appear in `checkpoints/Mixed_Dataset_Training/` (configurable via `cfg["TRAINING"]["CHECKPOINT_DIR"])`.
  - **⭐️ Our trained checkpoint is provided** at `checkpoints/LAMBDA_RESOURCE_LOSS_0.15/best_model_gate.pth`. 
    - (Optional) Update `cfg["TRAINING"]["INFERENCE_CHECKPOINT"]` to your checkpoint path so evaluation scripts pick it up automatically.

## 5. Inference
- Standard run: `python inference.py`
- With efficiency measurement: `python inference.py --measure_efficiency`
  - Outputs are written to `cfg["DATA"]["INFERENCE_OUTPUT_DIR"]` (default `output/main/`).

## 6. Evaluation
Run `evaluation/MMTab_evaluation.py` to evaluate the performance.