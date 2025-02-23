# #!/usr/bin/env python3
# """
# Script to perform “dark matter” analysis of SAE reconstruction error,
# following the methodology of the original paper. We compute three metrics:
#   1. SAE Error Norm Prediction: Optimal linear probe from x to ||SaeError(x)||².
#   2. SAE Error Vector Prediction: Optimal linear transform from x to SaeError(x).
#   3. Nonlinear FVU: Fraction of variance unexplained by SAE(x) plus the linear prediction.

# The analysis is performed for all trainers across three SAE configurations:
#   - ._run3_google_gemma-2-2b_jump_relu
#   - ._run3_google_gemma-2-9b_width16384_jump_relu
#   - ._run3_google_gemma-2-9b_width32768_jump_relu

# For each trainer, we extract width and target L0 from the config file,
# compute the metrics on activations from transformer layer 20 (filtered to tokens beyond a fixed position),
# and save the results to a JSON file as we go.
# """

# import os
# import torch
# import numpy as np
# import einops
# import json
# from tqdm.auto import tqdm
# from huggingface_hub import snapshot_download, login
# from datasets import load_dataset
# from nnsight import LanguageModel
# import gc
# from utils import load_dictionary, get_submodule 

# # ----------------------------- Configuration -----------------------------
# HF_TOKEN = "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"  # Replace with your token.
# hf_repo_id = "charlieoneill/gemma-medicine-sae"       # Repository for the SAE

# TAKE_SIZE = 1_000             # Number of tokenized contexts to process
# BATCH_SIZE = 16               # Batch size for processing contexts
# LAYER = 20                    # Transformer layer from which to extract activations
# TRAIN_SIZE_DESIRED = 150000   # Use 150k training samples if available; else use 60% split
# DEBUG = False
# tracer_kwargs = {'scan': True, 'validate': True} if DEBUG else {'scan': False, 'validate': False}

# RUN_FOLDER_NAMES = [
#     "._run3_google_gemma-2-2b_jump_relu",
#     "._run3_google_gemma-2-9b_width16384_jump_relu",
#     "._run3_google_gemma-2-9b_width32768_jump_relu",
# ]

# output_file = "dark_matter_results.json"

# # ----------------------------- Helper Functions -----------------------------
# def load_sparse_autoencoder(ae_path: str, device: str = 'cuda:0', dtype: torch.dtype = torch.float32):
#     dictionary, config = load_dictionary(ae_path, device)
#     dictionary = dictionary.to(dtype=dtype)
#     return dictionary, config

# def load_transformer_model(model_name: str = 'google/gemma-2-2b', device: str = 'cuda:0', dtype: torch.dtype = torch.float32) -> LanguageModel:
#     model = LanguageModel(model_name, dispatch=True, device_map=device)
#     model = model.to(dtype=dtype)
#     return model

# def load_tokenized_data(max_length: int = 1024, take_size: int = TAKE_SIZE) -> torch.Tensor:
#     def tokenize_and_concatenate(dataset, tokenizer, max_length=1024, column_name="text", add_bos_token=True):
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
#             chunks = [full_text[i * chunk_length: (i + 1) * chunk_length] for i in range(num_chunks)]
#             tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
#             tokens = tokens[tokens != tokenizer.pad_token_id]
#             num_tokens = len(tokens)
#             num_batches = num_tokens // seq_len
#             tokens = tokens[: seq_len * num_batches]
#             tokens = einops.rearrange(tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len)
#             if add_bos_token:
#                 prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
#                 tokens = np.concatenate([prefix, tokens], axis=1)
#             return {"tokens": tokens}

#         tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=[column_name])
#         return tokenized_dataset

#     transformer_model = load_transformer_model()
#     dataset = load_dataset('charlieoneill/medical-qa-combined', split='train', streaming=True)
#     dataset = dataset.shuffle(seed=42, buffer_size=10_000)
#     tokenized_dataset = tokenize_and_concatenate(dataset, transformer_model.tokenizer, max_length=max_length)
#     tokenized_dataset = tokenized_dataset.shuffle(42)
#     tokenized_dataset = tokenized_dataset.take(take_size)
#     tokens = np.stack([x['tokens'] for x in tokenized_dataset])
#     return torch.tensor(tokens)

# def compute_activations_and_error(sae, transformer_model, tokens: torch.Tensor, 
#                                   layer: int, batch_size: int = BATCH_SIZE, device: str = 'cuda:0') -> tuple:
#     sae.eval()
#     transformer_model.eval()
    
#     all_X = []
#     all_E = []
#     max_length = tokens.shape[1]
#     submodule = get_submodule(transformer_model, layer)

#     for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Computing activations and errors"):
#         batch_tokens = tokens[i: i+batch_size]
#         texts = [transformer_model.tokenizer.decode(toks, skip_special_tokens=True) for toks in batch_tokens]
#         with torch.no_grad():
#             with transformer_model.trace(texts, **tracer_kwargs):
#                 x = submodule.output.save()  # shape: (batch, pos, d)
#             x = x.value
#             if isinstance(x, tuple):
#                 x = x[0]
#             if x.shape[1] > max_length:
#                 x = x[:, 1:max_length+1, :]
#             if x.shape[1] > 20:
#                 x = x[:, 20:, :]
#             X = einops.rearrange(x, "b pos d -> (b pos) d")
#             x_hat, _ = sae(x, output_features=True)
#             x_hat_flat = einops.rearrange(x_hat, "b pos d -> (b pos) d")
#             E = X - x_hat_flat

#             all_X.append(X.cpu().numpy())
#             all_E.append(E.cpu().numpy())
    
#     X_all = np.concatenate(all_X, axis=0)
#     E_all = np.concatenate(all_E, axis=0)
#     del X, E
#     torch.cuda.empty_cache()
#     gc.collect()
#     return X_all, E_all

# def augment_with_bias(X: np.ndarray) -> np.ndarray:
#     ones = np.ones((X.shape[0], 1))
#     return np.concatenate([X, ones], axis=1)

# def r2_score_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     return 1 - ss_res / ss_tot

# def r2_score_vector(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     r2s = []
#     d = y_true.shape[1]
#     for j in range(d):
#         ss_res = np.sum((y_true[:, j] - y_pred[:, j]) ** 2)
#         ss_tot = np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2)
#         r2s.append(1 - ss_res / ss_tot)
#     return np.mean(r2s)

# def compute_error_norm_probe(X_train, y_train, X_test, y_test):
#     X_aug_train = augment_with_bias(X_train)
#     X_aug_test = augment_with_bias(X_test)
#     a, _, _, _ = np.linalg.lstsq(X_aug_train, y_train, rcond=None)
#     y_pred = X_aug_test.dot(a)
#     r2 = r2_score_scalar(y_test, y_pred)
#     return r2, a

# def compute_error_vector_probe(X_train, E_train, X_test, E_test):
#     X_aug_train = augment_with_bias(X_train)
#     X_aug_test = augment_with_bias(X_test)
#     b, _, _, _ = np.linalg.lstsq(X_aug_train, E_train, rcond=None)
#     E_pred = X_aug_test.dot(b)
#     r2 = r2_score_vector(E_test, E_pred)
#     return r2, b, E_pred

# def compute_fvu_nonlinear(X_test, E_test, b, X_aug_test):
#     E_pred = X_aug_test.dot(b)
#     SAE_test = X_test - E_test
#     x_pred = SAE_test + E_pred
#     ss_res = np.sum((X_test - x_pred) ** 2)
#     ss_tot = np.sum((X_test - np.mean(X_test, axis=0)) ** 2)
#     r2 = 1 - ss_res / ss_tot
#     fvu_nonlinear = 1 - r2
#     return fvu_nonlinear, r2

# # ----------------------------- Main Analysis -----------------------------
# def main():
#     torch.set_grad_enabled(False)
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#     login(token=HF_TOKEN)
#     print("Downloading SAE repository from Hugging Face...")
#     local_repo = snapshot_download(repo_id=hf_repo_id, repo_type="model")
#     print(f"Repository downloaded to: {local_repo}")

#     print("Loading tokenized data...")
#     tokens = load_tokenized_data(max_length=128, take_size=TAKE_SIZE)
#     print(f"Tokenized data shape: {tokens.shape}")

#     # Load previously saved results, if any.
#     results_list = []
#     processed_trainers = set()
#     if os.path.exists(output_file):
#         with open(output_file, "r") as f:
#             results_list = json.load(f)
#         processed_trainers = {(res["run_folder"], res["trainer_dir"]) for res in results_list}
#         print(f"Resuming from {len(processed_trainers)} processed trainers.")
#     else:
#         print("No previous results found; starting from scratch.")

#     for run_folder_name in RUN_FOLDER_NAMES:
#         run_folder = os.path.join(local_repo, run_folder_name, "resid_post_layer_20")
#         if not os.path.exists(run_folder):
#             print(f"Run folder not found: {run_folder}")
#             continue
#         trainer_dirs = [os.path.join(run_folder, d) for d in os.listdir(run_folder)
#                         if d.startswith("trainer_") and os.path.isdir(os.path.join(run_folder, d))]
#         trainer_dirs.sort()
#         print(f"Found {len(trainer_dirs)} trainers in {run_folder_name}.")

#         for trainer_dir in trainer_dirs:
#             trainer_key = (run_folder_name, os.path.basename(trainer_dir))
#             if trainer_key in processed_trainers:
#                 print(f"Skipping already processed trainer: {trainer_dir}")
#                 continue

#             config_path = os.path.join(trainer_dir, "config.json")
#             if not os.path.exists(config_path):
#                 print(f"Config not found for {trainer_dir}. Skipping.")
#                 continue
#             with open(config_path, "r") as f:
#                 config_data = json.load(f)
#             width = config_data["trainer"]["dict_size"]
#             target_l0 = config_data["trainer"]["target_l0"]
#             lm_name = config_data["trainer"]["lm_name"]
#             activation_dim = config_data["trainer"]["activation_dim"]

#             print(f"Processing {trainer_dir} (LM: {lm_name}, width: {width}, target_l0: {target_l0})")

#             sae, _ = load_sparse_autoencoder(trainer_dir, device='cuda:0', dtype=torch.float32)

#             if "2b" in run_folder_name:
#                 transformer_model = load_transformer_model(model_name='google/gemma-2-2b', device='cuda:0', dtype=torch.float32)
#                 batch_size = 16
#             else:
#                 transformer_model = load_transformer_model(model_name='google/gemma-2-9b', device='cuda:0', dtype=torch.float32)
#                 batch_size = 8

#             X, E = compute_activations_and_error(sae, transformer_model, tokens, layer=LAYER, batch_size=batch_size, device='cuda:0')
#             del transformer_model
#             print(f"Activation matrix shape: {X.shape}, SAE error matrix shape: {E.shape}")

#             y = np.sum(E**2, axis=1)
#             N = X.shape[0]
#             train_size = TRAIN_SIZE_DESIRED if N >= TRAIN_SIZE_DESIRED else int(0.6 * N)
#             print(f"Using {train_size} training samples and {N - train_size} test samples.")
#             X_train, X_test = X[:train_size], X[train_size:]
#             E_train, E_test = E[:train_size], E[train_size:]
#             y_train, y_test = y[:train_size], y[train_size:]

#             r2_norm, _ = compute_error_norm_probe(X_train, y_train, X_test, y_test)
#             r2_vector, b, _ = compute_error_vector_probe(X_train, E_train, X_test, E_test)
#             X_aug_test = augment_with_bias(X_test)
#             fvu_nonlinear, r2_nonlinear = compute_fvu_nonlinear(X_test, E_test, b, X_aug_test)

#             result = {
#                 "run_folder": run_folder_name,
#                 "trainer_dir": os.path.basename(trainer_dir),
#                 "lm_name": lm_name,
#                 "width": width,
#                 "target_l0": target_l0,
#                 "activation_dim": activation_dim,
#                 "r2_error_norm": r2_norm,
#                 "r2_error_vector": r2_vector,
#                 "r2_nonlinear": r2_nonlinear,
#                 "fvu_nonlinear": fvu_nonlinear,
#                 "num_training_samples": int(train_size),
#                 "num_test_samples": int(N - train_size)
#             }
#             results_list.append(result)
#             processed_trainers.add(trainer_key)
#             print(f"Results for {trainer_dir}: {result}")

#             with open(output_file, "w") as f:
#                 json.dump(results_list, f, indent=4)

#             # Free memory
#             del X, E, y, X_train, X_test, E_train, E_test, y_train, y_test, X_aug_test, b, r2_norm, r2_vector, r2_nonlinear, fvu_nonlinear
#             torch.cuda.empty_cache()
#             gc.collect()

#     print(f"All results have been saved incrementally to {output_file}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Script to perform “dark matter” analysis of gemmascope SAE reconstruction error,
following the methodology of the original paper. We compute three metrics:
  1. SAE Error Norm Prediction: Optimal linear probe from x to ||SaeError(x)||².
  2. SAE Error Vector Prediction: Optimal linear transform from x to SaeError(x).
  3. Nonlinear FVU: Fraction of variance unexplained by SAE(x) plus the linear prediction.

This script iterates over gemmascope SAE configurations for layer 20 from two model sizes:
  - For 2b: widths 16k and 65k.
  - For 9b: widths 16k and 32k.
Within each configuration, all available target L0 values are evaluated.
Results for each SAE are saved to a JSON file as we go.
"""

import os
import json
import torch
import numpy as np
import einops
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download, login
from datasets import load_dataset
from nnsight import LanguageModel
import gc
from dictionary import JumpReluAutoEncoder  # Gemmascope SAE loader
from utils import get_submodule  # Assumes this is defined

# ----------------------------- Configuration -----------------------------
HF_TOKEN = "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"  # Replace with your token.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

TAKE_SIZE = 1_000             # Number of tokenized contexts to process.
BATCH_SIZE = 16               # Batch size for processing contexts.
LAYER = 20                    # Transformer layer from which to extract activations.
TRAIN_SIZE_DESIRED = 150_000  # Use 150k training samples if available; else use 60% split.
DEBUG = False
tracer_kwargs = {'scan': False, 'validate': False} if not DEBUG else {'scan': True, 'validate': True}

MODEL_SIZES = ["2b", "9b"]
WIDTHS_BY_MODEL = {
    "2b": ["16k", "65k"],
    "9b": ["16k", "32k"]
}

def get_repo_id(model_size, model_width):
    #return f"google/gemma-scope-{model_size}-pt-res/layer_20/width_{model_width}"
    return f"google/gemma-scope-{model_size}-pt-res"


output_file = "dark_matter_results_gemmascope.json"

# ----------------------------- Helper Functions -----------------------------
def load_transformer_model(model_name: str, device: str = "cuda:0", dtype: torch.dtype = torch.float32) -> LanguageModel:
    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    return model

def load_tokenized_data(max_length: int = 1024, take_size: int = TAKE_SIZE) -> torch.Tensor:
    def tokenize_and_concatenate(dataset, tokenizer, max_length=1024, column_name="text", add_bos_token=True):
        for key in dataset.features:
            if key != column_name:
                dataset = dataset.remove_columns(key)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        seq_len = max_length - 1 if add_bos_token else max_length

        def tokenize_function(examples):
            text = examples[column_name]
            full_text = tokenizer.eos_token.join(text)
            # Split full_text into 20 chunks.
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

    transformer_model = load_transformer_model("google/gemma-2-2b", device=device, dtype=dtype)
    dataset = load_dataset("charlieoneill/medical-qa-combined", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    tokenized_dataset = tokenize_and_concatenate(dataset, transformer_model.tokenizer, max_length=max_length)
    tokenized_dataset = tokenized_dataset.shuffle(42)
    tokenized_dataset = tokenized_dataset.take(take_size)
    tokens = np.stack([x["tokens"] for x in tokenized_dataset])
    tokens_torch = torch.tensor(tokens)
    return tokens_torch

def compute_activations_and_error(sae, transformer_model, tokens: torch.Tensor, 
                                  layer: int, batch_size: int = BATCH_SIZE, device: str = "cuda:0") -> tuple:
    sae.eval()
    transformer_model.eval()
    
    all_X = []
    all_E = []
    max_length = tokens.shape[1]
    submodule = get_submodule(transformer_model, layer)

    for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Computing activations and errors"):
        batch_tokens = tokens[i : i + batch_size]
        texts = [transformer_model.tokenizer.decode(toks, skip_special_tokens=True) for toks in batch_tokens]
        with torch.no_grad():
            with transformer_model.trace(texts, **tracer_kwargs):
                x = submodule.output.save()  # shape: (batch, pos, d)
            x = x.value
            if isinstance(x, tuple):
                x = x[0]
            if x.shape[1] > max_length:
                x = x[:, 1:max_length+1, :]
            if x.shape[1] > 20:
                x = x[:, 20:, :]
            X = einops.rearrange(x, "b pos d -> (b pos) d")
            x_hat, _ = sae(x, output_features=True)
            x_hat_flat = einops.rearrange(x_hat, "b pos d -> (b pos) d")
            E = X - x_hat_flat

            all_X.append(X.cpu().numpy())
            all_E.append(E.cpu().numpy())
    X_all = np.concatenate(all_X, axis=0)
    E_all = np.concatenate(all_E, axis=0)
    return X_all, E_all

def augment_with_bias(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1))
    return np.concatenate([X, ones], axis=1)

def r2_score_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def r2_score_vector(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r2s = []
    d = y_true.shape[1]
    for j in range(d):
        ss_res = np.sum((y_true[:, j] - y_pred[:, j]) ** 2)
        ss_tot = np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2)
        r2s.append(1 - ss_res / ss_tot)
    return np.mean(r2s)

def compute_error_norm_probe(X_train, y_train, X_test, y_test):
    X_aug_train = augment_with_bias(X_train)
    X_aug_test = augment_with_bias(X_test)
    a, _, _, _ = np.linalg.lstsq(X_aug_train, y_train, rcond=None)
    y_pred = X_aug_test.dot(a)
    r2 = r2_score_scalar(y_test, y_pred)
    return r2, a

def compute_error_vector_probe(X_train, E_train, X_test, E_test):
    X_aug_train = augment_with_bias(X_train)
    X_aug_test = augment_with_bias(X_test)
    b, _, _, _ = np.linalg.lstsq(X_aug_train, E_train, rcond=None)
    E_pred = X_aug_test.dot(b)
    r2 = r2_score_vector(E_test, E_pred)
    return r2, b, E_pred

def compute_fvu_nonlinear(X_test, E_test, b, X_aug_test):
    E_pred = X_aug_test.dot(b)
    SAE_test = X_test - E_test
    x_pred = SAE_test + E_pred
    ss_res = np.sum((X_test - x_pred) ** 2)
    ss_tot = np.sum((X_test - np.mean(X_test, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    fvu_nonlinear = 1 - r2
    return fvu_nonlinear, r2

# ----------------------------- Main Analysis -----------------------------
def main():
    torch.set_grad_enabled(False)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    login(token=HF_TOKEN)

    # Load previously saved results, if any.
    results_list = []
    processed_configs = set()  # keys: (model_size, width, l0)
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results_list = json.load(f)
        processed_configs = {(res["model_size"], res["width"], res["l0"]) for res in results_list}
        print(f"Resuming from {len(processed_configs)} processed configurations.")
    else:
        print("No previous results found; starting from scratch.")

    print("Loading tokenized data...")
    tokens = load_tokenized_data(max_length=128, take_size=TAKE_SIZE)
    print(f"Tokenized data shape: {tokens.shape}")

    for model_size in MODEL_SIZES:

        widths = WIDTHS_BY_MODEL[model_size]
        for width in widths:
            repo_id = get_repo_id(model_size, width)
            print(f"Downloading repository {repo_id} ...")
            # local_repo = snapshot_download(repo_id=repo_id, repo_type="model")
            local_repo = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[f"layer_{LAYER}/width_{width}/average_l0_*/*"]
            )

            print(f"Repository downloaded to: {local_repo}")
            width_dir = os.path.join(local_repo, f"layer_{LAYER}", f"width_{width}")
            if not os.path.exists(width_dir):
                print(f"Width directory not found: {width_dir}")
                continue
            for l0_dir in os.listdir(width_dir):
                if not l0_dir.startswith("average_l0_"):
                    continue
                l0_value_str = l0_dir.split("average_l0_")[-1]
                try:
                    l0_value = int(l0_value_str)
                except ValueError:
                    print(f"Could not parse l0 value from {l0_dir}. Skipping.")
                    continue

                config_key = (model_size, width, l0_value)
                if config_key in processed_configs:
                    print(f"Skipping already processed config: {config_key}")
                    continue

                sae_dir = os.path.join(width_dir, l0_dir)
                print(f"Processing {repo_id} | width: {width} | l0: {l0_value}")
                try:
                    sae = JumpReluAutoEncoder.from_npz(
                        repo_id, LAYER, width, l0_value, dtype=dtype, device=device
                    )
                except Exception as e:
                    print(f"Error loading SAE from {sae_dir}: {e}")
                    continue

                # X, E = compute_activations_and_error(sae, transformer_model, tokens, layer=LAYER, device=device)
                # print(f"Activation shape: {X.shape}, SAE error shape: {E.shape}")

                if model_size == "2b":
                    batch_size = 16
                    model_name = f"google/gemma-2-{model_size}"
                    transformer_model = load_transformer_model(model_name, device=device, dtype=dtype)
                    X, E = compute_activations_and_error(sae, transformer_model, tokens, layer=LAYER, batch_size=batch_size, device='cuda:0')
                else:
                    batch_size = 8
                    model_name = f"google/gemma-2-{model_size}"
                    transformer_model = load_transformer_model(model_name, device=device, dtype=dtype)
                    X, E = compute_activations_and_error(sae, transformer_model, tokens, layer=LAYER, batch_size=batch_size, device='cuda:0')

                del transformer_model

                y = np.sum(E**2, axis=1)
                N = X.shape[0]
                train_size = TRAIN_SIZE_DESIRED if N >= TRAIN_SIZE_DESIRED else int(0.6 * N)
                X_train, X_test = X[:train_size], X[train_size:]
                E_train, E_test = E[:train_size], E[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                r2_norm, _ = compute_error_norm_probe(X_train, y_train, X_test, y_test)
                r2_vector, b, _ = compute_error_vector_probe(X_train, E_train, X_test, E_test)
                X_aug_test = augment_with_bias(X_test)
                fvu_nonlinear, r2_nonlinear = compute_fvu_nonlinear(X_test, E_test, b, X_aug_test)

                result = {
                    "model_size": model_size,
                    "repo_id": repo_id,
                    "width": width,
                    "l0": l0_value,
                    "r2_error_norm": r2_norm,
                    "r2_error_vector": r2_vector,
                    "r2_nonlinear": r2_nonlinear,
                    "fvu_nonlinear": fvu_nonlinear,
                    "num_training_samples": int(train_size),
                    "num_test_samples": int(N - train_size)
                }
                results_list.append(result)
                processed_configs.add(config_key)
                print(f"Results for {repo_id} | width {width} | l0 {l0_value}: {result}")

                # Incrementally save results.
                with open(output_file, "w") as f:
                    json.dump(results_list, f, indent=4)

                del X, E, y, X_train, X_test, E_train, E_test, y_train, y_test, X_aug_test, b, r2_norm, r2_vector, r2_nonlinear, fvu_nonlinear
                torch.cuda.empty_cache()
                gc.collect()

    print(f"All results have been saved incrementally to {output_file}")

if __name__ == "__main__":
    main()
