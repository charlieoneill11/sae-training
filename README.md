# SAE Factory

This repository provides a fully reproducible pipeline for training JumpReLU Sparse AutoEncoders (SAEs) on activations extracted from large-scale language models. The codebase includes data processing, activation buffering, model training with a custom JumpReLU nonlinearity, evaluation, and utilities for pushing both datasets and trained models to the Hugging Face Hub.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Training the SAE](#training-the-sae)
  - [Evaluation and Pushing to Hugging Face](#evaluation-and-pushing-to-hugging-face)
  - [Example Notebook](#example-notebook)
- [Reproducibility and Configuration](#reproducibility-and-configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

The goal of this project is to learn sparse representations from hidden activations of a pretrained language model using a custom autoencoder architecture with a **JumpReLU** activation function. The pipeline:
- Processes various medical question answering (QA) datasets.
- Uses Hugging Face Datasets to load and format data.
- Extracts activations from a large-scale language model (e.g., `google/gemma-2-2b`) via a custom `ActivationBuffer`.
- Trains a JumpReLU SAE using specialized training routines (see `JumpReluTrainer` in `jumprelu.py`).
- Logs progress via console output and optionally weights and metrics via WandB.
- Saves checkpoints locally and optionally pushes the final model to the Hugging Face Hub.

---

## Features

- **Unified Dataset Formatting:**  
  Processes multiple medical QA datasets (e.g., MedQA, MedMCQA, PubMedQA, MMLU variants) into a consistent format for training and evaluation.

- **Activation Buffering:**  
  Implements a memory-efficient `ActivationBuffer` that streams activations from a language model in batches, ensuring efficient training even with large-scale data.

- **Customized JumpReLU SAE:**  
  Uses a custom autograd function (`JumpReLUFunction`) that applies a thresholded nonlinearity designed to yield sparse representations.

- **Flexible Training Pipeline:**  
  Supports hyperparameter tuning via configurations defined in the demo configuration file. The training loop handles random seed initialization, learning rate scheduling (with warmup and decay phases), and gradient normalization.

- **Hugging Face Integration:**  
  Loads datasets with the `datasets` library and pushes both combined datasets and trained models to the Hugging Face Hub.

- **Notebook Example:**  
  An example Jupyter Notebook (`train_sae.ipynb`) demonstrates how to load activations from the Hugging Face Hub and train a JumpReLU SAE interactively.

---

## File Structure

- **`dataset.py`**  
  Contains functions to format medical QA datasets into a unified text representation, concatenates them, and pushes the combined dataset to the Hugging Face Hub.

- **`dictionary_learning/`**  
  - **`demo.py`** – Main execution script for training the SAE. Initializes the language model, configures the activation buffer, and triggers the training loop via `run_training()`.
  - **`jumprelu.py`** – Implements the custom JumpReLU nonlinearity and the `JumpReluTrainer` class which trains the SAE.
  - **`training.py`** – Defines training routines including logging, checkpointing, and evaluation.
  - **`trainer.py`** – Contains a base class for SAE trainers and helper functions such as unit norm normalization.
  - **`dictionary.py`** – Defines the dictionary classes for autoencoders, including the JumpReLU-based variant.
  - **`demo_config.py`** – Holds hyperparameters and configuration details (e.g., number of tokens, batch sizes, learning rates, dictionary widths, and random seeds).
  - **`buffer.py`** – Implements the `ActivationBuffer` class for efficient streaming and batching of activations extracted from the language model.
  - **`utils.py`** – Provides utility functions for dataset manipulation, converting Hugging Face datasets to generators, and pushing models to the Hub.

- **`train_sae.ipynb`**  
  An example notebook that demonstrates how to load activations from the Hugging Face Hub and perform training on a JumpReLU SAE.

---

## Installation

### Prerequisites

- Python 3.8 or above.
- A CUDA-enabled GPU is recommended for training, though CPU mode is supported.
- [PyTorch](https://pytorch.org/) (with CUDA support if available)
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/) and [datasets](https://huggingface.co/docs/datasets)
- Additional dependencies such as `pandas`, `tqdm`, and optionally `wandb` (Weights & Biases) for logging.

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/jumprelu-sae.git
   cd jumprelu-sae
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   # Run the startup script
   bash startup.sh
   ```

---

## Usage

### Data Processing

The `dataset.py` script processes multiple medical QA datasets, reformats them, and combines them into a single dataset. To run the data processing pipeline:

```bash
python dataset.py
```
This will:
- Download datasets from Hugging Face:
  * [MedQA](https://huggingface.co/datasets/openlifescienceai/medqa)
  * [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) 
  * [College medicine subset of MMLU](https://huggingface.co/datasets/openlifescienceai/mmlu_college_medicine)
  * [MMLU clinical knowledge](https://huggingface.co/datasets/openlifescienceai/mmlu_clinical_knowledge)
  * [MMLU professional medicine](https://huggingface.co/datasets/openlifescienceai/mmlu_professional_medicine)
  * [PubmedQA](https://huggingface.co/datasets/openlifescienceai/pubmedqa)
- Apply formatting functions (e.g., `format_medqa`, `format_medmcqa`, `format_pubmed_qa`, and a generic formatter for MMLU).
- Concatenate the formatted datasets.
- Push the combined dataset to the Hugging Face Hub under a specified repository (e.g., `charlieoneill/medical-qa-combined`).

### Training the SAE

The main training routine is implemented in `dictionary_learning/demo.py`. This script configures the language model, initializes the activation buffer, and trains the JumpReLU SAE using configurations specified in `demo_config.py`.

Run the training with:

```bash
python dictionary_learning/demo.py
```

Key steps performed by the training script:
- **Model Initialization:** Loads a pretrained language model (e.g., `"google/gemma-2-2b"`) using the `nnsight.LanguageModel` class. The model is moved to the desired device (GPU or CPU) and cast to the configured dtype.
- **Activation Buffer:** An `ActivationBuffer` instance is created to feed batches of activations from the model’s internal submodule (e.g., a residual stream).
- **Trainer Configuration:** Using hyperparameter combinations from `demo_config.py`, multiple trainer configurations (e.g., different dictionary widths, learning rates, random seeds) may be generated.
- **Training Loop:** The SAE is trained over a defined number of tokens. Each training step computes a reconstruction loss combined with a sparsity penalty and then updates using Adam with a custom learning rate schedule.
- **Checkpointing & Evaluation:** Checkpoints (SAE state dictionaries) are saved, and the final model can be pushed to the Hugging Face Hub if configured.

### Evaluation and Pushing to Hugging Face

During and after training, evaluations are performed to compute metrics such as reconstruction error and sparsity level. The final configuration and evaluation results are stored as JSON files in the output directory. If the repository ID is provided in the configuration, the final model and checkpoints will automatically be pushed to the Hugging Face Hub.

---

## Reproducibility and Configuration

- **Random Seeds:**  
  Both Python’s and PyTorch’s random seeds are set in the training routines (see `demo_config.py` and within training functions) to ensure reproducibility.
  
- **Configuration File:**  
  All key parameters (batch sizes, learning rates, number of tokens, dictionary sizes, warmup steps, etc.) are defined in `demo_config.py`. Adjust these parameters to reproduce experiments or perform hyperparameter sweeps.

- **Environment Variables:**  
  To optimize GPU memory usage, the training script sets:
  ```python
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  ```
  Additionally, for multiprocessing support with WandB, the start method is enforced:
  ```python
  mp.set_start_method("spawn", force=True)
  ```

---

## Troubleshooting

- **CUDA Out of Memory:**  
  If you encounter CUDA memory errors, consider lowering the batch size or adjusting the activation buffer size. Verify that the environment variable `PYTORCH_CUDA_ALLOC_CONF` is correctly set to handle expandable memory segments.

- **Dataset Streaming Issues:**  
  The code uses retry parameters in the `datasets` library to handle intermittent internet issues:
  ```python
  config.STREAMING_READ_MAX_RETRIES = 100
  config.STREAMING_READ_RETRY_INTERVAL = 20
  ```

- **Hugging Face Authentication:**  
  Make sure you have a valid Hugging Face token. Replace the token in the training scripts:
  ```python
  HF_TOKEN = "your_token_here"
  login(token=HF_TOKEN)
  ```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgments

- This project leverages open-source libraries such as PyTorch, Hugging Face Datasets, and Hugging Face Hub.
- Special thanks to the contributors behind the `nnsight` package and QA dataset repositories.
- We welcome contributions and feedback to further enhance this training pipeline.

---

Happy Training!