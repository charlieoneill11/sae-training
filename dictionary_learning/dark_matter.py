#!/usr/bin/env python3
"""
Script to analyze the linear predictability of the reconstruction error ("dark matter")
in sparse autoencoders (SAEs), following the framework of Engels et al. (2024).

For an activation x, the SAE produces a reconstruction x̂ = SAE(x) with error
    e(x) = x - x̂.
We then solve for the optimal linear mapping b* that best predicts e(x) from x via:
    b* = argmin_b ||b·x - e(x)||_2^2,
and compute the goodness of fit via:
    R^2 = 1 - (||e(x) - b*·x||_2^2 / ||e(x)||_2^2).
A high R^2 indicates that much of the error is linearly predictable (“dark matter”).
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
INPUT_DIM = 768
HIDDEN_DIM = 22 * INPUT_DIM
CONFIG_PATH = "config.yaml"
HF_TOKEN = "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"  # Replace with your Hugging Face token
hf_repo_id = "charlieoneill/gemma-medicine-sae"  # Repo where the SAE is hosted

# Analysis parameters
TAKE_SIZE = 1_000  # Number of tokenized sequences to process
BATCH_SIZE = 16   # Batch size for processing tokens
LAYER = 20        # Transformer layer to extract activations from

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

def load_tokenized_data(max_length: int = 256, batch_size: int = 64, take_size: int = TAKE_SIZE) -> torch.Tensor:
    """
    Loads and tokenizes the 'charlieoneill/medical-qa-combined' dataset.
    """
    def tokenize_and_concatenate(dataset, tokenizer, streaming=False, max_length=1024, column_name="text", add_bos_token=True):
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
    tokenized_dataset = tokenize_and_concatenate(dataset, transformer_model.tokenizer, max_length=max_length, streaming=True)
    tokenized_dataset = tokenized_dataset.shuffle(42)
    tokenized_dataset = tokenized_dataset.take(take_size)
    tokens = np.stack([x['tokens'] for x in tokenized_dataset])
    tokens_torch = torch.tensor(tokens)
    return tokens_torch

def compute_activations_and_error(sae, transformer_model, tokens: torch.Tensor, layer: int, device: str = 'cuda:0') -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each tokenized sequence, compute the activation x from the given transformer layer,
    obtain the SAE reconstruction x̂, and compute the error e(x) = x - x̂.
    Returns flattened X and E as numpy arrays.
    """
    sae.eval()
    transformer_model.eval()

    all_X = []
    all_E = []
    max_length = tokens.shape[1] # sequence length

    # Get the transformer submodule for the specified layer
    submodule = get_submodule(transformer_model, layer)

    for i in tqdm(range(0, tokens.shape[0], BATCH_SIZE), desc="Computing activations and errors"):
        batch_tokens = tokens[i:i+BATCH_SIZE]
        # Decode tokens to text for tracing
        texts = [transformer_model.tokenizer.decode(toks, skip_special_tokens=True) for toks in batch_tokens]
        with torch.no_grad():
            with transformer_model.trace(texts, **tracer_kwargs):
                x = submodule.output.save() # shape: (batch, pos, d)

            x = x.value
            if isinstance(x, tuple):
                x = x[0]
            # Optionally remove extra tokens (e.g. BOS) if present
            if x.shape[1] > max_length:
                x = x[:, 1:max_length+1, :]
            # Flatten activations: (batch * pos, d)
            X = einops.rearrange(x, "b pos d -> (b pos) d")
            # Compute SAE reconstruction
            x_hat, _ = sae(x, output_features=True)
            x_hat_flat = einops.rearrange(x_hat, "b pos d -> (b pos) d")
            # Reconstruction error
            E = X - x_hat_flat

            all_X.append(X.cpu().numpy())
            all_E.append(E.cpu().numpy())

    X = np.concatenate(all_X, axis=0)
    E = np.concatenate(all_E, axis=0)
    return X, E

def compute_optimal_linear_predictor(X: np.ndarray, E: np.ndarray) -> tuple:
    """
    Solves for b* that minimizes ||X b - E||_F^2 via least squares,
    returning b* and the predicted error E_pred = X b*.
    """
    b_opt = np.linalg.pinv(X) @ E
    E_pred = X @ b_opt
    return b_opt, E_pred

def compute_r2(E: np.ndarray, E_pred: np.ndarray) -> float:
    """
    Computes the coefficient of determination:
        R^2 = 1 - ||E - E_pred||_F^2 / ||E||_F^2.
    """
    error_norm = np.linalg.norm(E - E_pred, ord='fro')**2
    total_norm = np.linalg.norm(E, ord='fro')**2
    return 1 - error_norm / total_norm

# ----------------------------- Main Analysis -----------------------------
def main():
    torch.set_grad_enabled(False)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Log in to Hugging Face.
    login(token=HF_TOKEN)

    print("Downloading repository from Hugging Face...")
    local_repo = snapshot_download(repo_id=hf_repo_id, repo_type="model")
    print(f"Repository downloaded to: {local_repo}")

    # Define the run folder used during training.
    run_folder = os.path.join(local_repo, "._run3_google_gemma-2-2b_jump_relu", "resid_post_layer_20")
    if not os.path.exists(run_folder):
        raise FileNotFoundError(f"Run folder not found: {run_folder}")

    # List and sort trainer folders.
    trainer_folders = [
        os.path.join(run_folder, d)
        for d in os.listdir(run_folder)
        if d.startswith("trainer_") and os.path.isdir(os.path.join(run_folder, d))
    ]
    trainer_folders.sort()

    # Load the SAE and transformer model.
    sae, _ = load_sparse_autoencoder(trainer_folders[0], device='cuda:0', dtype=torch.float32)
    transformer_model = load_transformer_model(model_name='google/gemma-2-2b', device='cuda:0', dtype=torch.float32)

    # Load tokenized data.
    print("Loading tokenized data...")
    tokens = load_tokenized_data(take_size=TAKE_SIZE)
    print(f"Tokenized data shape: {tokens.shape}")

    # Compute activations and reconstruction error.
    print("Computing activations and reconstruction error...")
    X, E = compute_activations_and_error(sae, transformer_model, tokens, layer=LAYER, device='cuda:0')
    print(f"Activation matrix shape: {X.shape}")
    print(f"Error matrix shape: {E.shape}")

    # Solve for the optimal linear predictor.
    print("Computing optimal linear predictor for reconstruction error...")
    b_opt, E_pred = compute_optimal_linear_predictor(X, E)

    # Compute the coefficient of determination R^2.
    r2 = compute_r2(E, E_pred)
    print(f"Coefficient of determination (R^2) for predicting reconstruction error: {r2:.4f}")

if __name__ == "__main__":
    main()