#!/usr/bin/env python3
"""
Script to perform “dark matter” analysis of SAE reconstruction error,
following the methodology of the original paper. We compute three metrics:
  1. SAE Error Norm Prediction: Optimal linear probe from x to ||SaeError(x)||².
  2. SAE Error Vector Prediction: Optimal linear transform from x to SaeError(x).
  3. Nonlinear FVU: Fraction of variance unexplained by SAE(x) plus the linear prediction.

The script uses tokenized activations from layer 20 of the Gemma model and filters out
tokens before position 200 in each context.
"""

import os
import torch
import numpy as np
import einops
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download, login
from datasets import load_dataset
from nnsight import LanguageModel
from utils import load_dictionary, get_submodule  # Assumes these are defined

# ----------------------------- Configuration -----------------------------
# Model and repository parameters
HF_TOKEN = "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"  # Replace with your token.
hf_repo_id = "charlieoneill/gemma-medicine-sae"  # Repo for the SAE
MODEL_FILENAME = "sparse_autoencoder.pth"
INPUT_DIM = 768
HIDDEN_DIM = 22 * INPUT_DIM
CONFIG_PATH = "config.yaml"

# Analysis parameters
TAKE_SIZE = 10000        # Number of tokenized contexts to process
BATCH_SIZE = 16          # Batch size for processing contexts
LAYER = 20               # Transformer layer from which to extract activations
TRAIN_SIZE_DESIRED = 150000  # Use 150k training samples if available; else use 60% split
DEBUG = False
tracer_kwargs = {'scan': True, 'validate': True} if DEBUG else {'scan': False, 'validate': False}

# ----------------------------- Helper Functions -----------------------------
def load_sparse_autoencoder(ae_path: str, device: str = 'cuda:0', dtype: torch.dtype = torch.float32):
    dictionary, config = load_dictionary(ae_path, device)
    dictionary = dictionary.to(dtype=dtype)
    return dictionary, config

def load_transformer_model(model_name: str = 'google/gemma-2-2b', device: str = 'cuda:0', dtype: torch.dtype = torch.float32) -> LanguageModel:
    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    return model

def load_tokenized_data(max_length: int = 1024, take_size: int = TAKE_SIZE) -> torch.Tensor:
    """
    Loads and tokenizes the 'charlieoneill/medical-qa-combined' dataset.
    """
    def tokenize_and_concatenate(dataset, tokenizer, max_length=1024, column_name="text", add_bos_token=True):
        # Remove all columns except the text column.
        for key in dataset.features:
            if key != column_name:
                dataset = dataset.remove_columns(key)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        seq_len = max_length - 1 if add_bos_token else max_length

        def tokenize_function(examples):
            text = examples[column_name]
            full_text = tokenizer.eos_token.join(text)
            # Split full_text into 20 chunks
            num_chunks = 20
            chunk_length = (len(full_text) - 1) // num_chunks + 1
            chunks = [full_text[i * chunk_length: (i + 1) * chunk_length] for i in range(num_chunks)]
            tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
            tokens = tokens[tokens != tokenizer.pad_token_id]
            num_tokens = len(tokens)
            num_batches = num_tokens // seq_len
            tokens = tokens[: seq_len * num_batches]
            tokens = einops.rearrange(tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len)
            if add_bos_token:
                prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
                tokens = np.concatenate([prefix, tokens], axis=1)
            return {"tokens": tokens}

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=[column_name])
        return tokenized_dataset

    transformer_model = load_transformer_model()
    dataset = load_dataset('charlieoneill/medical-qa-combined', split='train', streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    tokenized_dataset = tokenize_and_concatenate(dataset, transformer_model.tokenizer, max_length=max_length)
    tokenized_dataset = tokenized_dataset.shuffle(42)
    tokenized_dataset = tokenized_dataset.take(take_size)
    tokens = np.stack([x['tokens'] for x in tokenized_dataset])
    tokens_torch = torch.tensor(tokens)
    return tokens_torch

def compute_activations_and_error(sae, transformer_model, tokens: torch.Tensor, layer: int, device: str = 'cuda:0') -> tuple:
    """
    For each tokenized context, compute the activation x from the given transformer layer,
    obtain the SAE reconstruction, and compute the error:
      SaeError(x) = x - SAE(x)
    Additionally, filter out activations for tokens before position 200.
    Returns:
      X_all: Flattened activations (only tokens from position >= 200) of shape (N, d)
      E_all: Corresponding SAE errors of shape (N, d)
    """
    sae.eval()
    transformer_model.eval()
    
    all_X = []
    all_E = []
    max_length = tokens.shape[1]  # original sequence length

    submodule = get_submodule(transformer_model, layer)

    for i in tqdm(range(0, tokens.shape[0], BATCH_SIZE), desc="Computing activations and errors"):
        batch_tokens = tokens[i: i+BATCH_SIZE]
        texts = [transformer_model.tokenizer.decode(toks, skip_special_tokens=True) for toks in batch_tokens]
        with torch.no_grad():
            with transformer_model.trace(texts, **tracer_kwargs):
                x = submodule.output.save()  # shape: (batch, pos, d)
            x = x.value
            if isinstance(x, tuple):
                x = x[0]
            # Optionally remove extra tokens (e.g. BOS) if present.
            if x.shape[1] > max_length:
                x = x[:, 1:max_length+1, :]
            # Filter: only consider tokens after position 20.
            if x.shape[1] > 20:
                x = x[:, 20:, :]
            # Flatten: (batch * pos, d)
            X = einops.rearrange(x, "b pos d -> (b pos) d")
            # Compute SAE reconstruction and error.
            x_hat, _ = sae(x, output_features=True)
            x_hat_flat = einops.rearrange(x_hat, "b pos d -> (b pos) d")
            E = X - x_hat_flat

            all_X.append(X.cpu().numpy())
            all_E.append(E.cpu().numpy())
    
    X_all = np.concatenate(all_X, axis=0)
    E_all = np.concatenate(all_E, axis=0)
    return X_all, E_all

def augment_with_bias(X: np.ndarray) -> np.ndarray:
    """ Append a column of ones to X for bias terms. """
    ones = np.ones((X.shape[0], 1))
    return np.concatenate([X, ones], axis=1)

def r2_score_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Compute R^2 for scalar predictions. """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def r2_score_vector(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the R^2 per dimension and return the average R^2.
    y_true and y_pred have shape (N, d).
    """
    r2s = []
    d = y_true.shape[1]
    for j in range(d):
        ss_res = np.sum((y_true[:, j] - y_pred[:, j]) ** 2)
        ss_tot = np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2)
        r2s.append(1 - ss_res / ss_tot)
    return np.mean(r2s)

# ----------------------------- Regression Metrics -----------------------------
def compute_error_norm_probe(X_train, y_train, X_test, y_test):
    """
    Solve for a* in R^(d+1) such that a^T * [X;1] approximates ||SaeError(x)||^2.
    Returns the test R^2.
    """
    X_aug_train = augment_with_bias(X_train)
    X_aug_test = augment_with_bias(X_test)
    # Solve least squares: a = argmin ||X_aug_train * a - y_train||^2.
    a, _, _, _ = np.linalg.lstsq(X_aug_train, y_train, rcond=None)
    y_pred = X_aug_test.dot(a)
    r2 = r2_score_scalar(y_test, y_pred)
    return r2, a

def compute_error_vector_probe(X_train, E_train, X_test, E_test):
    """
    Solve for b* in R^(d+1 x d) mapping augmented X to E.
    Returns the average R^2 (across dimensions) on the test set.
    """
    X_aug_train = augment_with_bias(X_train)
    X_aug_test = augment_with_bias(X_test)
    b, _, _, _ = np.linalg.lstsq(X_aug_train, E_train, rcond=None)
    E_pred = X_aug_test.dot(b)
    r2 = r2_score_vector(E_test, E_pred)
    return r2, b, E_pred

def compute_fvu_nonlinear(X_test, E_test, b, X_aug_test):
    """
    Compute the FVU_nonlinear:
      FVU_nonlinear = 1 - R^2(x, SAE(x) + b*·x)
    Note that SAE(x) = x - SaeError(x) so that:
      SAE(x) + b*·x = (x - E) + (predicted error)
    """
    # Recompute predicted error on test set.
    E_pred = X_aug_test.dot(b)
    # Recover SAE(x) from E_test: SAE(x) = x - E_test.
    SAE_test = X_test - E_test
    x_pred = SAE_test + E_pred
    # Compute overall R^2 for x.
    ss_res = np.sum((X_test - x_pred) ** 2)
    ss_tot = np.sum((X_test - np.mean(X_test, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    fvu_nonlinear = 1 - r2
    return fvu_nonlinear, r2

# ----------------------------- Main Analysis -----------------------------
def main():
    torch.set_grad_enabled(False)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Log in to Hugging Face.
    login(token=HF_TOKEN)
    print("Downloading SAE repository from Hugging Face...")
    local_repo = snapshot_download(repo_id=hf_repo_id, repo_type="model")
    print(f"Repository downloaded to: {local_repo}")

    # Define run folder and trainer folders.
    run_folder = os.path.join(local_repo, "._run3_google_gemma-2-2b_jump_relu", "resid_post_layer_20")
    if not os.path.exists(run_folder):
        raise FileNotFoundError(f"Run folder not found: {run_folder}")
    trainer_folders = [os.path.join(run_folder, d) for d in os.listdir(run_folder)
                       if d.startswith("trainer_") and os.path.isdir(os.path.join(run_folder, d))]
    trainer_folders.sort()

    # Load models.
    sae, _ = load_sparse_autoencoder(trainer_folders[0], device='cuda:0', dtype=torch.float32)
    transformer_model = load_transformer_model(model_name='google/gemma-2-2b', device='cuda:0', dtype=torch.float32)

    # Load tokenized data.
    print("Loading tokenized data...")
    tokens = load_tokenized_data(max_length=128, take_size=TAKE_SIZE)
    print(f"Tokenized data shape: {tokens.shape}")

    # Compute activations and SAE error.
    print("Computing activations and SAE error...")
    X, E = compute_activations_and_error(sae, transformer_model, tokens, layer=LAYER, device='cuda:0')
    print(f"Activation matrix shape: {X.shape}")
    print(f"SAE error matrix shape: {E.shape}")

    # Prepare targets:
    # For error norm prediction, target is the squared norm of SaeError for each activation.
    y = np.sum(E**2, axis=1)

    # Split data into training and test sets.
    N = X.shape[0]
    if N >= TRAIN_SIZE_DESIRED:
        train_size = TRAIN_SIZE_DESIRED
    else:
        train_size = int(0.6 * N)
    print(f"Using {train_size} training samples and {N - train_size} test samples.")
    X_train, X_test = X[:train_size], X[train_size:]
    E_train, E_test = E[:train_size], E[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # --- Metric 1: SAE Error Norm Prediction ---
    print("Computing optimal linear probe for SAE error norm prediction...")
    r2_norm, a = compute_error_norm_probe(X_train, y_train, X_test, y_test)
    print(f"R^2 for SAE error norm prediction: {r2_norm:.4f}")

    # --- Metric 2: SAE Error Vector Prediction ---
    print("Computing optimal linear transform for SAE error vector prediction...")
    r2_vector, b, E_pred_test = compute_error_vector_probe(X_train, E_train, X_test, E_test)
    print(f"Average R^2 for SAE error vector prediction: {r2_vector:.4f}")

    # --- Metric 3: Nonlinear FVU ---
    # For computing FVU_nonlinear, we need the augmented test input.
    X_aug_test = augment_with_bias(X_test)
    fvu_nonlinear, r2_nonlinear = compute_fvu_nonlinear(X_test, E_test, b, X_aug_test)
    print(f"R^2 for predicting x from SAE(x)+linear probe: {r2_nonlinear:.4f}")
    print(f"Nonlinear FVU (1 - R^2): {fvu_nonlinear:.4f}")

if __name__ == "__main__":
    main()