#!/usr/bin/env python3
"""
Script to analyze sparse autoencoder features by retrieving top-k activating examples,
obtaining top-k and bottom-k boosted logits, formatting prompts, and getting responses
from an AI interpreter.

Requirements:
- torch
- huggingface_hub
- einops
- numpy
- yaml
- nnsight
- datasets
- tqdm
- openai
- IPython
"""

import torch
from huggingface_hub import hf_hub_download, snapshot_download, login
import einops
import numpy as np
import yaml
from tqdm.auto import tqdm
from nnsight import LanguageModel  # <-- Changed: use nnsight instead of transformer_lens
from datasets import load_dataset
from openai import OpenAI, AzureOpenAI
import re
import html
import os
import gc
import json
import random
from itertools import cycle

from utils import load_dictionary, get_submodule  # Assumes get_submodule(model, layer) is defined

# ----------------------------- Configuration -----------------------------

# Parameters
REPO_NAME = "charlieoneill/sparse-coding"  # Hugging Face repo name
MODEL_FILENAME = "sparse_autoencoder.pth"  # Model file name in the repo
INPUT_DIM = 768  # Example input dimension
HIDDEN_DIM = 22 * INPUT_DIM  # Projection up parameter * input_dim
SCORES_PATH = "scores.npy"  # Path to the saved scores
FEATURE_INDICES = [x for x in range(10)]
feature_indices = FEATURE_INDICES
TOP_K = 10  # Number of top activating examples
BOTTOM_K = 10  # Number of bottom boosted logits

# OpenAI Configuration
CONFIG_PATH = "config.yaml"  # Path to your config file containing API keys

# Define tracer kwargs for nnsight (as in the provided snippet)
DEBUG = False
tracer_kwargs = {'scan': True, 'validate': True} if DEBUG else {'scan': False, 'validate': False}

# ----------------------------- Helper Functions -----------------------------

def load_sparse_autoencoder(ae_path: str, device: str = 'cuda:0', dtype: torch.dtype = torch.float32):
    dictionary, config = load_dictionary(ae_path, device)
    dictionary = dictionary.to(dtype=dtype)
    return dictionary, config
    
def load_transformer_model(model_name: str = 'google/gemma-2-2b', device: str = 'cuda:0', dtype: torch.dtype = torch.float32) -> LanguageModel:
    # Create an nnsight LanguageModel instance instead of a HookedTransformer.
    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    return model

def load_scores(scores_path: str) -> np.ndarray:
    scores = np.load(scores_path)
    return scores

def load_tokenized_data(max_length: int = 256, batch_size: int = 64, take_size: int = 102400) -> torch.Tensor:
    def tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=1024,
        column_name="text",
        add_bos_token=True,
    ):
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
            chunks = [
                full_text[i * chunk_length : (i + 1) * chunk_length]
                for i in range(num_chunks)
            ]
            tokens = tokenizer(chunks, return_tensors="np", padding=True)[
                "input_ids"
            ].flatten()
            tokens = tokens[tokens != tokenizer.pad_token_id]
            num_tokens = len(tokens)
            num_batches = num_tokens // seq_len
            tokens = tokens[: seq_len * num_batches]
            tokens = einops.rearrange(
                tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
            )
            if add_bos_token:
                prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
                tokens = np.concatenate([prefix, tokens], axis=1)
            return {"tokens": tokens}

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name]
        )
        return tokenized_dataset

    transformer_model = load_transformer_model()
    dataset = load_dataset('charlieoneill/medical-qa-combined', split='train', streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    tokenized_owt = tokenize_and_concatenate(dataset, transformer_model.tokenizer, max_length=max_length, streaming=True)
    tokenized_owt = tokenized_owt.shuffle(42)
    tokenized_owt = tokenized_owt.take(take_size)
    owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
    owt_tokens_torch = torch.tensor(owt_tokens)
    return owt_tokens_torch

def compute_scores(sae, transformer_model: LanguageModel, owt_tokens_torch: torch.Tensor, layer: int, feature_indices: list, device: str = 'cuda:0') -> np.ndarray:
    sae.eval()

    scores = []
    batch_size = 16
    max_length = owt_tokens_torch.shape[1]
    for i in tqdm(range(0, owt_tokens_torch.shape[0], batch_size), desc="Computing scores"):
        with torch.no_grad():
            # Convert tokenized inputs (tensor) back to text
            texts = [transformer_model.tokenizer.decode(tokens, skip_special_tokens=True) 
                     for tokens in owt_tokens_torch[i : i + batch_size]]
            # batch = owt_tokens_torch[i : i + batch_size]
            # Get the submodule corresponding to the desired layer (using a helper from utils)
            submodule = get_submodule(transformer_model, layer)
            # Use nnsight’s trace method to capture activations from the submodule
            with transformer_model.trace(texts, **tracer_kwargs):
                x = submodule.output.save()  # Shape: (batch, pos, d_model)
            x = x.value
            if isinstance(x, tuple):
                x = x[0]
            # Remove the BOS token
            if x.shape[1] > max_length:
                x = x[:, 1:max_length+1, :]
            X = einops.rearrange(x, "batch pos d_model -> (batch pos) d_model")
            cur_scores = sae.encode(X)[:, feature_indices]
            x_hat, f = sae(x, output_features=True)
            l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
            total_variance = torch.var(x, dim=0).sum()
            residual_variance = torch.var(x - x_hat, dim=0).sum()
            frac_variance_explained = (1 - residual_variance / total_variance)
            l0 = (f != 0).float().sum(dim=-1).mean()
            print(f"L2 loss: {l2_loss}, L0 loss: {l0}, Fraction of variance explained: {frac_variance_explained}")
            cur_scores_reshaped = einops.rearrange(cur_scores, "(b pos) n -> b n pos", pos=owt_tokens_torch.shape[1])
            scores.append(cur_scores_reshaped.cpu().numpy().astype(np.float16))

    scores = np.concatenate(scores, axis=0)
    np.save(SCORES_PATH, scores)
    return scores

def get_top_k_indices(scores: np.ndarray, feature_index: int, k: int = TOP_K) -> np.ndarray:
    """ 
    Get the indices of the examples where the feature activates the most.
    scores is shape (batch, feature, pos), so we index with feature_index.
    """
    feature_scores = scores[:, feature_index, :]
    top_k_indices = feature_scores.argsort()[-k:][::-1]
    return top_k_indices

def get_topk_bottomk_logits(feature_index: int, sae, transformer_model: LanguageModel, k: int = TOP_K) -> tuple:
    feature_vector = sae.decoder.weight.data[:, feature_index]
    W_U = transformer_model.W_U  # Assuming the nnsight LanguageModel exposes this attribute
    logits = einops.einsum(W_U, feature_vector, "d_model vocab, d_model -> vocab")
    top_k_logits = logits.topk(k).indices
    bottom_k_logits = logits.topk(k, largest=False).indices
    top_k_tokens = [transformer_model.to_string(x.item()) for x in top_k_logits]
    bottom_k_tokens = [transformer_model.to_string(x.item()) for x in bottom_k_logits]
    return top_k_tokens, bottom_k_tokens

def highlight_scores_in_html(token_strs: list, scores: list, seq_idx: int, max_color: str = "#ff8c00", zero_color: str = "#ffffff", show_score: bool = True) -> tuple:
    if len(token_strs) != len(scores):
        print(f"Length mismatch between tokens and scores (len(tokens)={len(token_strs)}, len(scores)={len(scores)})") 
        return "", ""
    scores_min = min(scores)
    scores_max = max(scores)
    if scores_max - scores_min == 0:
        scores_normalized = np.zeros_like(scores)
    else:
        scores_normalized = (np.array(scores) - scores_min) / (scores_max - scores_min)
    max_color_vec = np.array(
        [int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)]
    )
    zero_color_vec = np.array(
        [int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)]
    )
    color_vecs = np.einsum("i, j -> ij", scores_normalized, max_color_vec) + np.einsum(
        "i, j -> ij", 1 - scores_normalized, zero_color_vec
    )
    color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]
    if show_score:
        tokens_html = "".join(
            [
                f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}<span class='feature_val'> ({scores[i]:.2f})</span></span>"""
                for i, token_str in enumerate(token_strs)
            ]
        )
        clean_text = " | ".join(
            [f"{token_str} ({scores[i]:.2f})" for i, token_str in enumerate(token_strs)]
        )
    else:
        tokens_html = "".join(
            [
                f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}</span>"""
                for i, token_str in enumerate(token_strs)
            ]
        )
        clean_text = " | ".join(token_strs)
    head = """
    <style>
        span.token {
            font-family: monospace;
            border-style: solid;
            border-width: 1px;
            border-color: #dddddd;
        }
        span.feature_val {
            font-size: smaller;
            color: #555555;
        }
    </style>
    """
    return head + tokens_html, convert_clean_text(clean_text)

def convert_clean_text(clean_text: str, k: int = 1, tokens_left: int = 30, tokens_right: int = 5) -> str:
    token_score_pairs = clean_text.split(" | ")
    if token_score_pairs:
        token_score_pairs = token_score_pairs[1:]
    tokens_with_scores = []
    token_score_pattern = re.compile(r"^(.+?) \((\d+\.\d+)\)$")
    for token_score in token_score_pairs:
        match = token_score_pattern.match(token_score.strip())
        if match:
            token = match.group(1)
            score = float(match.group(2))
            tokens_with_scores.append((token, score))
        else:
            token = token_score.split(' (')[0].strip()
            tokens_with_scores.append((token, 0.0))
    sorted_tokens = sorted(tokens_with_scores, key=lambda x: x[1], reverse=True)
    top_k_tokens = [token for token, score in sorted_tokens if score > 0][:k]
    top_k_indices = [i for i, (token, score) in enumerate(tokens_with_scores) if token in top_k_tokens and score > 0]
    windows = []
    for idx in top_k_indices:
        start = max(0, idx - tokens_left)
        end = min(len(tokens_with_scores) - 1, idx + tokens_right)
        windows.append((start, end))
    merged_windows = []
    for window in sorted(windows, key=lambda x: x[0]):
        if not merged_windows:
            merged_windows.append(window)
        else:
            last_start, last_end = merged_windows[-1]
            current_start, current_end = window
            if current_start <= last_end + 1:
                merged_windows[-1] = (last_start, max(last_end, current_end))
            else:
                merged_windows.append(window)
    selected_indices = set()
    for start, end in merged_windows:
        selected_indices.update(range(start, end + 1))
    converted_tokens = []
    for i, (token, score) in enumerate(tokens_with_scores):
        if i in selected_indices:
            if token in top_k_tokens and score > 0:
                token = f"<<{token}>>"
            converted_tokens.append(token)
    converted_text = " ".join(converted_tokens)
    return converted_text


"""
Main function to execute the analysis for all defined feature indices.
"""
torch.set_grad_enabled(False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HF_TOKEN = "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"  # Replace with your token.
login(token=HF_TOKEN)

# The Hugging Face repo ID where the trained SAEs were pushed.
hf_repo_id = "charlieoneill/gemma-medicine-sae"

print("Downloading repository from Hugging Face...")
local_repo = snapshot_download(repo_id=hf_repo_id, repo_type="model")
print(f"Repository downloaded to: {local_repo}")

# Define the run folder that was used during training.
run_folder = os.path.join(local_repo, "._run3_google_gemma-2-2b_jump_relu", "resid_post_layer_20")
if not os.path.exists(run_folder):
    raise FileNotFoundError(f"Run folder not found: {run_folder}")

# List all trainer folders (trainer_0 to trainer_5) within the run folder.
trainer_folders = [
    os.path.join(run_folder, d)
    for d in os.listdir(run_folder)
    if d.startswith("trainer_") and os.path.isdir(os.path.join(run_folder, d))
]
trainer_folders.sort()  # Ensure proper order

# Load models
sae, config = load_sparse_autoencoder(trainer_folders[0], device='cuda:0', dtype=torch.float32)
transformer_model = load_transformer_model(model_name='google/gemma-2-2b', device='cuda:0', dtype=torch.float32)

# Load or compute scores
try:
    scores = load_scores(SCORES_PATH)
except FileNotFoundError:
    print(f"Scores file not found at {SCORES_PATH}. Computing scores...")
    owt_tokens_torch = load_tokenized_data(take_size=10_000)
    layer = 20
    device = 'cuda:0'
    scores = compute_scores(sae, transformer_model, owt_tokens_torch, layer, FEATURE_INDICES, device=device)

# Load tokenized data
owt_tokens_torch = load_tokenized_data(take_size=10_000)

# 1) Count number of non-zero entries
nonzero_count = np.count_nonzero(scores)

# 2) Number of features with any non-zero activation
features_nonzero = np.sum((scores != 0).any(axis=(0, 2)))

# 3) Mean number of examples an activating feature activates on
examples_per_feature = np.sum((scores != 0).any(axis=2), axis=0)
mean_examples = np.mean(examples_per_feature[examples_per_feature > 0])

print("Number of non-zero entries:", nonzero_count)
print("Number of features with non-zero activations:", features_nonzero)
print("Mean number of examples an activating feature activates on:", mean_examples)
