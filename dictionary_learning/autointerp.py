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
import torch as t
from huggingface_hub import hf_hub_download, snapshot_download, login
import einops
import numpy as np
import yaml
from tqdm.auto import tqdm
from nnsight import LanguageModel  # use nnsight instead of transformer_lens
from datasets import load_dataset
from openai import OpenAI, AzureOpenAI
import re
import html
import os
import gc
import json
import random
from itertools import cycle, islice
from tqdm.auto import tqdm

import demo_config  # Assumes a demo configuration module is available.
from utils import load_dictionary, get_submodule, hf_dataset_to_generator
from buffer import ActivationBuffer  # Import the activation buffer from buffer.py

# ----------------------------- Configuration -----------------------------

# Parameters
REPO_NAME = "charlieoneill/sparse-coding"      # Hugging Face repo name
MODEL_FILENAME = "sparse_autoencoder.pth"       # Model file name in the repo
INPUT_DIM = 768                                 # Example input dimension
HIDDEN_DIM = 22 * INPUT_DIM                     # Projection up parameter * input_dim
SCORES_PATH = "scores.npy"                      # Path to the saved scores
FEATURE_INDICES = [x for x in range(10)]
feature_indices = FEATURE_INDICES
TOP_K = 10                                    # Number of top activating examples
BOTTOM_K = 10                                 # Number of bottom boosted logits

# OpenAI Configuration
CONFIG_PATH = "config.yaml"                   # Path to your config file containing API keys

# Define tracer kwargs for nnsight
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

def load_scores(scores_path: str) -> np.ndarray:
    scores = np.load(scores_path)
    return scores

def get_top_k_indices(scores: np.ndarray, feature_index: int, k: int = TOP_K) -> np.ndarray:
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
    max_color_vec = np.array([int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)])
    zero_color_vec = np.array([int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)])
    color_vecs = np.einsum("i, j -> ij", scores_normalized, max_color_vec) + np.einsum("i, j -> ij", 1 - scores_normalized, zero_color_vec)
    color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]
    if show_score:
        tokens_html = "".join(
            [f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}<span class='feature_val'> ({scores[i]:.2f})</span></span>"""
             for i, token_str in enumerate(token_strs)]
        )
        clean_text = " | ".join([f"{token_str} ({scores[i]:.2f})" for i, token_str in enumerate(token_strs)])
    else:
        tokens_html = "".join(
            [f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}</span>"""
             for i, token_str in enumerate(token_strs)]
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

# ----------------------------- Main Execution -----------------------------

torch.set_grad_enabled(False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HF_TOKEN = "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"  # Replace with your token.
login(token=HF_TOKEN)

# Download the Hugging Face repository with the trained SAEs.
hf_repo_id = "charlieoneill/gemma-medicine-sae"
print("Downloading repository from Hugging Face...")
local_repo = snapshot_download(repo_id=hf_repo_id, repo_type="model")
print(f"Repository downloaded to: {local_repo}")

# Define the run folder used during training.
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

print(f"Scores file not found at {SCORES_PATH}. Computing scores using activation buffer...")
    
# --- Adjusted Buffer Construction (as in run_sae_training) ---
model_name = 'google/gemma-2-2b'
# Get parameters from demo_config
context_length = 256 #demo_config.LLM_CONFIG[model_name].context_length  # e.g. 256
llm_batch_size = 64 #demo_config.LLM_CONFIG[model_name].llm_batch_size
sae_batch_size = 8192 #demo_config.LLM_CONFIG[model_name].sae_batch_size
dtype = demo_config.LLM_CONFIG[model_name].dtype

# Compute number of buffer inputs from desired total tokens (default 250,000 tokens)
buffer_tokens = 250_000
num_buffer_inputs = buffer_tokens // context_length
print(f"Buffer will hold {num_buffer_inputs} examples (total {buffer_tokens} tokens).")

# Training/usage parameters (for example, if used for training these would determine steps)
# For this analysis script we simply use the buffer to stream activations.
io = "out"
activation_dim = config["trainer"]["activation_dim"]
layer = 20 #config["trainer"]["layer"]
submodule = get_submodule(transformer_model, layer)

generator = hf_dataset_to_generator("charlieoneill/medical-qa-combined")

activation_buffer = ActivationBuffer(
    generator,
    transformer_model,
    submodule,
    n_ctxs=num_buffer_inputs,
    ctx_len=context_length,
    refresh_batch_size=llm_batch_size,
    out_batch_size=sae_batch_size,
    io=io,
    d_submodule=activation_dim,
    device='cuda:0'
)
# --- End Adjusted Buffer Construction ---

# Now process a target number of tokens. For example, we want to process 49 million tokens.
target_tokens = 49_000_000
tokens_processed = 0
batches_processed = 0
batch_counter = 0
scores_list = []
# Total next calls = len(dataset) / (sae_batch_size // context_length)
total_examples = 10_000
total_next_calls = total_examples // (sae_batch_size // context_length)

# while tokens_processed < target_tokens:
for i in tqdm(range(total_next_calls)):
        gc.collect()
        torch.cuda.empty_cache()
        x = next(activation_buffer)  # x shape: (token_examples, d_submodule)
        #print(f"X shape: {x.shape}")
        token_examples = x.shape[0]
        d_submodule = x.shape[1]
        tokens_processed += token_examples
        batches_processed += token_examples // context_length
        # Compute SAE scores. Insert a singleton “position” dimension so that SAE output is (batch, pos, d_submodule).
        cur_scores = sae.encode(x)[:, feature_indices] # (token_examples, num_features) #.unsqueeze(0)[:, :, feature_indices]
        #print(f"Cur scores shape: {cur_scores.shape}")
        # Reshape to (batch, pos, feature) - there are token_examples // context_length batches
        # So the reshape should have shape (token_examples // context_length, context_length, feature)
        cur_scores = einops.rearrange(
            cur_scores,
            "(batch pos) feature -> batch pos feature",
            batch=token_examples // context_length,
            pos=context_length
        )
        #print(f"Cur scores reshaped shape: {cur_scores.shape}")
        # Print total batches, total tokens
        #print(f"Total batches: {batches_processed}, total tokens: {tokens_processed}")

        scores_list.append(cur_scores.cpu().numpy().astype(np.float16))
        batch_counter += 1
        
scores = np.concatenate(scores_list, axis=0)
print(f"Finished processing {tokens_processed} tokens in {batch_counter} batches.")
print(f"Scores shape: {scores.shape}")
# Optionally, you can save the scores with: np.save(SCORES_PATH, scores)
np.save("scores.npy", scores)

# Compute some summary statistics on the computed scores.
nonzero_count = np.count_nonzero(scores)
features_nonzero = np.sum((scores != 0).any(axis=(0, 1)))
# Reduce the position dimension to compute activations per example.
examples_per_feature = (scores != 0).any(axis=1)
mean_examples = np.mean(examples_per_feature.sum(axis=0)[examples_per_feature.sum(axis=0) > 0])

print("Number of non-zero entries:", nonzero_count)
print("Number of features with non-zero activations:", features_nonzero)
print("Mean number of examples an activating feature activates on:", mean_examples)

# # ----------------------------- Configuration -----------------------------

# # Parameters
# REPO_NAME = "charlieoneill/sparse-coding"  # Hugging Face repo name
# MODEL_FILENAME = "sparse_autoencoder.pth"  # Model file name in the repo
# INPUT_DIM = 768  # Example input dimension
# HIDDEN_DIM = 22 * INPUT_DIM  # Projection up parameter * input_dim
# SCORES_PATH = "scores.npy"  # Path to the saved scores
# FEATURE_INDICES = [x for x in range(10)]
# feature_indices = FEATURE_INDICES
# TOP_K = 10  # Number of top activating examples
# BOTTOM_K = 10  # Number of bottom boosted logits

# # OpenAI Configuration
# CONFIG_PATH = "config.yaml"  # Path to your config file containing API keys

# # Define tracer kwargs for nnsight (as in the provided snippet)
# DEBUG = False
# tracer_kwargs = {'scan': True, 'validate': True} if DEBUG else {'scan': False, 'validate': False}

# # ----------------------------- Helper Functions -----------------------------

# def load_sparse_autoencoder(ae_path: str, device: str = 'cuda:0', dtype: torch.dtype = torch.float32):
#     dictionary, config = load_dictionary(ae_path, device)
#     dictionary = dictionary.to(dtype=dtype)
#     return dictionary, config
    
# def load_transformer_model(model_name: str = 'google/gemma-2-2b', device: str = 'cuda:0', dtype: torch.dtype = torch.float32) -> LanguageModel:
#     # Create an nnsight LanguageModel instance instead of a HookedTransformer.
#     model = LanguageModel(model_name, dispatch=True, device_map=device)
#     model = model.to(dtype=dtype)
#     return model

# def load_scores(scores_path: str) -> np.ndarray:
#     scores = np.load(scores_path)
#     return scores

# def load_tokenized_data(max_length: int = 256, batch_size: int = 64, take_size: int = 102400) -> torch.Tensor:
#     def tokenize_and_concatenate(
#         dataset,
#         tokenizer,
#         streaming=False,
#         max_length=max_length,
#         column_name="text",
#         add_bos_token=True,
#     ):
#         for key in dataset.features:
#             if key != column_name:
#                 dataset = dataset.remove_columns(key)
#         if tokenizer.pad_token is None:
#             tokenizer.add_special_tokens({"pad_token": "<PAD>"})
#         seq_len = max_length - 1 if add_bos_token else max_length

#         def tokenize_function(examples):
#             text = examples[column_name]
#             full_text = tokenizer.eos_token.join(text)
#             num_chunks = 20
#             chunk_length = (len(full_text) - 1) // num_chunks + 1
#             chunks = [
#                 full_text[i * chunk_length : (i + 1) * chunk_length]
#                 for i in range(num_chunks)
#             ]
#             tokens = tokenizer(chunks, return_tensors="np", padding=True)[
#                 "input_ids"
#             ].flatten()
#             tokens = tokens[tokens != tokenizer.pad_token_id]
#             num_tokens = len(tokens)
#             num_batches = num_tokens // seq_len
#             tokens = tokens[: seq_len * num_batches]
#             tokens = einops.rearrange(
#                 tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
#             )
#             if add_bos_token:
#                 prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
#                 tokens = np.concatenate([prefix, tokens], axis=1)
#             return {"tokens": tokens}

#         tokenized_dataset = dataset.map(
#             tokenize_function, batched=True, remove_columns=[column_name]
#         )
#         return tokenized_dataset

#     transformer_model = load_transformer_model()
#     dataset = load_dataset('charlieoneill/medical-qa-combined', split='train', streaming=True)
#     dataset = dataset.shuffle(seed=42, buffer_size=10_000)
#     tokenized_owt = tokenize_and_concatenate(dataset, transformer_model.tokenizer, max_length=max_length, streaming=True)
#     tokenized_owt = tokenized_owt.shuffle(42)
#     tokenized_owt = tokenized_owt.take(take_size)
#     owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
#     owt_tokens_torch = torch.tensor(owt_tokens)
#     return owt_tokens_torch.cpu()

# def compute_scores(sae, transformer_model: LanguageModel, owt_tokens_torch: torch.Tensor, layer: int, feature_indices: list, device: str = 'cuda:0') -> np.ndarray:
#     sae.eval()

#     scores = []
#     batch_size = 128
#     max_length = owt_tokens_torch.shape[1]
#     submodule = get_submodule(transformer_model, layer)
#     for i in tqdm(range(0, owt_tokens_torch.shape[0], batch_size), desc="Computing scores"):
#         gc.collect()
#         torch.cuda.empty_cache()
#         with torch.no_grad():
#             # Convert tokenized inputs (tensor) back to text
#             texts = [transformer_model.tokenizer.decode(tokens, skip_special_tokens=True) 
#                      for tokens in owt_tokens_torch[i : i + batch_size]]
#             # batch = owt_tokens_torch[i : i + batch_size]
#             # Use nnsight’s trace method to capture activations from the submodule
#             with transformer_model.trace(texts, **tracer_kwargs):
#                 x = submodule.output.save()  # Shape: (batch, pos, d_model)
#                 submodule.output.stop()
#             x = x.value
#             if isinstance(x, tuple):
#                 x = x[0]
#             # Remove the BOS token
#             if x.shape[1] > max_length:
#                 x = x[:, 1:max_length+1, :]
#             cur_scores = sae.encode(x)[:, feature_indices]
#             scores.append(cur_scores.cpu().numpy().astype(np.float16))

#     scores = np.concatenate(scores, axis=0)
#     np.save(SCORES_PATH, scores)
#     return scores

# def get_top_k_indices(scores: np.ndarray, feature_index: int, k: int = TOP_K) -> np.ndarray:
#     """ 
#     Get the indices of the examples where the feature activates the most.
#     scores is shape (batch, feature, pos), so we index with feature_index.
#     """
#     feature_scores = scores[:, feature_index, :]
#     top_k_indices = feature_scores.argsort()[-k:][::-1]
#     return top_k_indices

# def get_topk_bottomk_logits(feature_index: int, sae, transformer_model: LanguageModel, k: int = TOP_K) -> tuple:
#     feature_vector = sae.decoder.weight.data[:, feature_index]
#     W_U = transformer_model.W_U  # Assuming the nnsight LanguageModel exposes this attribute
#     logits = einops.einsum(W_U, feature_vector, "d_model vocab, d_model -> vocab")
#     top_k_logits = logits.topk(k).indices
#     bottom_k_logits = logits.topk(k, largest=False).indices
#     top_k_tokens = [transformer_model.to_string(x.item()) for x in top_k_logits]
#     bottom_k_tokens = [transformer_model.to_string(x.item()) for x in bottom_k_logits]
#     return top_k_tokens, bottom_k_tokens

# def highlight_scores_in_html(token_strs: list, scores: list, seq_idx: int, max_color: str = "#ff8c00", zero_color: str = "#ffffff", show_score: bool = True) -> tuple:
#     if len(token_strs) != len(scores):
#         print(f"Length mismatch between tokens and scores (len(tokens)={len(token_strs)}, len(scores)={len(scores)})") 
#         return "", ""
#     scores_min = min(scores)
#     scores_max = max(scores)
#     if scores_max - scores_min == 0:
#         scores_normalized = np.zeros_like(scores)
#     else:
#         scores_normalized = (np.array(scores) - scores_min) / (scores_max - scores_min)
#     max_color_vec = np.array(
#         [int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)]
#     )
#     zero_color_vec = np.array(
#         [int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)]
#     )
#     color_vecs = np.einsum("i, j -> ij", scores_normalized, max_color_vec) + np.einsum(
#         "i, j -> ij", 1 - scores_normalized, zero_color_vec
#     )
#     color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]
#     if show_score:
#         tokens_html = "".join(
#             [
#                 f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}<span class='feature_val'> ({scores[i]:.2f})</span></span>"""
#                 for i, token_str in enumerate(token_strs)
#             ]
#         )
#         clean_text = " | ".join(
#             [f"{token_str} ({scores[i]:.2f})" for i, token_str in enumerate(token_strs)]
#         )
#     else:
#         tokens_html = "".join(
#             [
#                 f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}</span>"""
#                 for i, token_str in enumerate(token_strs)
#             ]
#         )
#         clean_text = " | ".join(token_strs)
#     head = """
#     <style>
#         span.token {
#             font-family: monospace;
#             border-style: solid;
#             border-width: 1px;
#             border-color: #dddddd;
#         }
#         span.feature_val {
#             font-size: smaller;
#             color: #555555;
#         }
#     </style>
#     """
#     return head + tokens_html, convert_clean_text(clean_text)

# def convert_clean_text(clean_text: str, k: int = 1, tokens_left: int = 30, tokens_right: int = 5) -> str:
#     token_score_pairs = clean_text.split(" | ")
#     if token_score_pairs:
#         token_score_pairs = token_score_pairs[1:]
#     tokens_with_scores = []
#     token_score_pattern = re.compile(r"^(.+?) \((\d+\.\d+)\)$")
#     for token_score in token_score_pairs:
#         match = token_score_pattern.match(token_score.strip())
#         if match:
#             token = match.group(1)
#             score = float(match.group(2))
#             tokens_with_scores.append((token, score))
#         else:
#             token = token_score.split(' (')[0].strip()
#             tokens_with_scores.append((token, 0.0))
#     sorted_tokens = sorted(tokens_with_scores, key=lambda x: x[1], reverse=True)
#     top_k_tokens = [token for token, score in sorted_tokens if score > 0][:k]
#     top_k_indices = [i for i, (token, score) in enumerate(tokens_with_scores) if token in top_k_tokens and score > 0]
#     windows = []
#     for idx in top_k_indices:
#         start = max(0, idx - tokens_left)
#         end = min(len(tokens_with_scores) - 1, idx + tokens_right)
#         windows.append((start, end))
#     merged_windows = []
#     for window in sorted(windows, key=lambda x: x[0]):
#         if not merged_windows:
#             merged_windows.append(window)
#         else:
#             last_start, last_end = merged_windows[-1]
#             current_start, current_end = window
#             if current_start <= last_end + 1:
#                 merged_windows[-1] = (last_start, max(last_end, current_end))
#             else:
#                 merged_windows.append(window)
#     selected_indices = set()
#     for start, end in merged_windows:
#         selected_indices.update(range(start, end + 1))
#     converted_tokens = []
#     for i, (token, score) in enumerate(tokens_with_scores):
#         if i in selected_indices:
#             if token in top_k_tokens and score > 0:
#                 token = f"<<{token}>>"
#             converted_tokens.append(token)
#     converted_text = " ".join(converted_tokens)
#     return converted_text


# """
# Main function to execute the analysis for all defined feature indices.
# """
# torch.set_grad_enabled(False)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# HF_TOKEN = "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"  # Replace with your token.
# login(token=HF_TOKEN)

# # The Hugging Face repo ID where the trained SAEs were pushed.
# hf_repo_id = "charlieoneill/gemma-medicine-sae"

# print("Downloading repository from Hugging Face...")
# local_repo = snapshot_download(repo_id=hf_repo_id, repo_type="model")
# print(f"Repository downloaded to: {local_repo}")

# # Define the run folder that was used during training.
# run_folder = os.path.join(local_repo, "._run3_google_gemma-2-2b_jump_relu", "resid_post_layer_20")
# if not os.path.exists(run_folder):
#     raise FileNotFoundError(f"Run folder not found: {run_folder}")

# # List all trainer folders (trainer_0 to trainer_5) within the run folder.
# trainer_folders = [
#     os.path.join(run_folder, d)
#     for d in os.listdir(run_folder)
#     if d.startswith("trainer_") and os.path.isdir(os.path.join(run_folder, d))
# ]
# trainer_folders.sort()  # Ensure proper order

# # Load models
# sae, config = load_sparse_autoencoder(trainer_folders[0], device='cuda:0', dtype=torch.float32)
# transformer_model = load_transformer_model(model_name='google/gemma-2-2b', device='cuda:0', dtype=torch.float32)

# # Load or compute scores
# try:
#     scores = load_scores(SCORES_PATH)
# except FileNotFoundError:
#     print(f"Scores file not found at {SCORES_PATH}. Computing scores...")
#     owt_tokens_torch = load_tokenized_data(take_size=10_000)
#     layer = 20
#     device = 'cuda:0'
#     scores = compute_scores(sae, transformer_model, owt_tokens_torch, layer, FEATURE_INDICES, device=device)

# # Load tokenized data
# owt_tokens_torch = load_tokenized_data(take_size=10_000)

# # 1) Count number of non-zero entries
# nonzero_count = np.count_nonzero(scores)

# # 2) Number of features with any non-zero activation
# features_nonzero = np.sum((scores != 0).any(axis=(0, 2)))

# # 3) Mean number of examples an activating feature activates on
# examples_per_feature = np.sum((scores != 0).any(axis=2), axis=0)
# mean_examples = np.mean(examples_per_feature[examples_per_feature > 0])

# print("Number of non-zero entries:", nonzero_count)
# print("Number of features with non-zero activations:", features_nonzero)
# print("Mean number of examples an activating feature activates on:", mean_examples)