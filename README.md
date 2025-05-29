# KotlinDecompiler

Toolkit for analyzing and comparing Kotlin decompilation and re-Kotlin conversion methods using structural, entropy, and
LM-based metrics.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Install JDK and Kotlin compiler (JVM target 23 recommended).
3. Prepare a working environment (GPU recommended for model inference).

## Repository Structure & Scripts

### Data Collection

* **`collect/bytecode/download_datasets.py`**
  Downloads `KExercises` and `KStack-clean` datasets.

    * Output: `originals/` directories with Kotlin `.kt` files.

* **`collect/process_models/compile_models.py`**
  Compiles `.kt` files to bytecode (`bytecode/`).

    * Uses `kotlinc` and fallback Gradle projects if needed.
    * Logs errors to `compile_errors.log`.

* **`collect/bytecode/bytecode_pair_collector.py`**
  Pairs `.kt` files with their disassembled bytecode (`javap`).

    * Output: `pairs.jsonl` in dataset root.

* **`collect/bytecode/merge_datasets.py`**
  Merges datasets, splits into train/test JSON files.

* **`distribution.py`**
  Builds token and bigram language models from datasets.

    * Output: `unigram.json`, `bigram.json`, `left.json`.

### Model Inference

* **`collect/process_models/process_model.py`**
  Runs selected AI model (`transformers`) to convert bytecode to Kotlin.

    * Input: `pairs.jsonl`.
    * Output: JSONL file with model outputs per `kt_path`.

* **`collect/process_models/merge_all_jsonl_with_hf.py`**
  Merges original data with all model outputs (JSONL).

### Metrics Computation

* **`collect/metrics/metrics_for_models.py`**
  Computes structural, entropy, and LM metrics for all outputs.

    * Input: merged JSONL and allowed paths JSON.
    * Output: CSV file with metrics per model.

* **`collect/metrics/metrics_collector.py`**
  Provides methods to compute metrics (`structural`, `entropy`, `lm_metrics`).

### Analysis

* **`analysis/tests_J2K.py`**
  Counts successful J2K conversions and compiles for each test.

* **`analysis/tests_ChatGPT.py`**
  Same as above, but for ChatGPT outputs.

* **`analysis/best_models.py`**
  Ranks models by metric distance to original code.

### Visualization

* **`charts/build_charts.py`**
  Generates bar charts and heatmaps for metrics comparisons.

### Model Training

* **`model_train/train.py`**
  Fine-tunes models with LoRA on bytecode-to-Kotlin task.

* **`model_train/find_hyperparameters.py`**
  Hyperparameter tuning with Optuna.

* **`model_train/merge.py`**
  Merges LoRA adapters with base models.

### Utilities

* **`dim_reduction/feature_selection.py`**
  Removes low-variance and highly correlated metrics.

## Recommended Pipeline

1. `download_datasets.py`
2. `compile_models.py`
3. `bytecode_pair_collector.py`
4. `merge_datasets.py` (optional)
5. `distribution.py`
6. `process_model.py <model_name>` for each model
7. `merge_all_jsonl_with_hf.py`
8. `metrics_for_models.py`
9. (optional) `tests_J2K.py`, `tests_ChatGPT.py`
10. (optional) `best_models.py`
11. (optional) `build_charts.py`

## Input/Output Locations

* **Originals:** `dataset/originals/`
* **Bytecode:** `dataset/bytecode/`
* **Bytecode pairs:** `pairs.jsonl`
* **Model outputs:** `*.jsonl`
* **Merged metrics:** `metrics_results.csv`
* **Charts:** `charts/`

## Notes

* GPU is recommended for AI model inference.
* Language models (`unigram`, `bigram`, `left`) must be built before metrics.
* Fine-tuning (`train.py`) is optional and requires GPU.

