# #!/usr/bin/env python
# """
# Script to train a JumpReLU sparse autoencoder (SAE) using precomputed activations.

# This script:
#   - Loads a tensor of activations from Hugging Face Hub.
#   - Wraps the activations in a simple Dataset/DataLoader so that batches are produced.
#   - Configures the training (using JumpReluTrainer and JumpReluAutoEncoder) and calls trainSAE.
# """

# import torch
# from torch.utils.data import Dataset, DataLoader
# import einops
# from huggingface_hub import hf_hub_download, HfApi

# # Import the JumpReLU SAE and trainer from the repository.
# from dictionary import JumpReluAutoEncoder
# from jumprelu import JumpReluTrainer
# from training import trainSAE

# # Flush cache
# torch.cuda.empty_cache()

# #!/usr/bin/env python
# """
# Script to train a JumpReLU sparse autoencoder (SAE) using precomputed activations.

# This script:
#   - Loads a tensor of activations from Hugging Face Hub.
#   - Wraps the activations in a simple Dataset/DataLoader so that batches are produced.
#   - Configures the training (using JumpReluTrainer and JumpReluAutoEncoder) and calls trainSAE.
  
# Note: This script assumes that the repository's modules (e.g. dictionary_learning, trainers,
# and training) are available in your PYTHONPATH.
# """

# #####################################
# # 1. Load Precomputed Activations   #
# #####################################


# # Device selection.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# try:
#     # Download the activation tensor (expected shape: [10000, 128, 2304])
#     repo_id = "charlieoneill/gemma-medicine-sae"  # adjust if needed
#     activation_file = hf_hub_download(repo_id=repo_id, filename="10000_128.pt")
#     activations = torch.load(activation_file)
#     print(f"Loaded activations with shape: {activations.shape}")
# except Exception as e:
#     print(f"Error loading activations: {e}")
#     # Fall back to random data (for debugging/testing)
#     activations = torch.randn(10000, 128, 2304)
#     print(f"Using random activations with shape: {activations.shape}")

# # --- Optional: Flatten the sequence dimension ---
# # Uncomment these lines if you wish to treat every token independently.
# activations = einops.rearrange(activations, "b s d -> (b s) d")
# print(f"Flattened activations shape: {activations.shape}")

# # Shuffle activations along first dimension
# activations = activations[torch.randperm(activations.size(0))]

# #####################################
# # 2. Create a DataLoader for Activations  #
# #####################################

# class ActivationDataset(Dataset):
#     def __init__(self, activations: torch.Tensor):
#         """
#         activations: a torch.Tensor of shape (N, d_model) OR (N, seq_len, d_model)
#         Keep activations on CPU until needed
#         """
#         self.activations = activations.pin_memory() if torch.cuda.is_available() else activations

#     def __len__(self):
#         return self.activations.size(0)

#     def __getitem__(self, idx):
#         # Only transfer the specific item when requested
#         return self.activations[idx]

# # Create the dataset without moving to GPU
# dataset = ActivationDataset(activations)  # activations stays on CPU
# dataset.config = {"d_submodule": activations.shape[-1]}

# # Create a DataLoader with memory-efficient settings
# batch_size = 1024
# data_loader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     drop_last=True,
#     pin_memory=True,  # This helps with faster CPU->GPU transfers
#     num_workers=4,    # Adjust based on your CPU cores
#     prefetch_factor=2 # Each worker will prefetch 2 batches
# )

# # Define an infinite iterator over the DataLoader.
# def infinite_loader(loader):
#     while True:
#         for batch in loader:
#             yield batch

# data_iterator = infinite_loader(data_loader)

# #####################################
# # 3. Configure Training Parameters  #
# #####################################

# # Activation dimension (d_model) is the last dimension of the activations.
# d_model = activations.shape[-1]  # e.g. 2304
# # Set dictionary (latent) size; here we use 16 * d_model (adjust as needed)
# dictionary_size = 16384 #16 * d_model
# # Total training steps.
# total_steps = 10_000

# print(f"Training for {total_steps} steps with batch size {batch_size} ({total_steps*batch_size} total tokens)")

# # Trainer configuration dictionary.
# trainer_cfg = {
#     "trainer": JumpReluTrainer,           # Use the JumpReLU trainer
#     "dict_class": JumpReluAutoEncoder,      # Use the JumpReLU autoencoder
#     "steps": total_steps,                   # required: total training steps
#     "activation_dim": d_model,
#     "dict_size": dictionary_size,
#     "layer": 0,                           # arbitrary value (you can choose a meaningful layer if you wish)
#     "lm_name": "precomputed",             # arbitrary value (since no language model is used)
#     "lr": 7e-5,
#     "device": device,
#     "bandwidth": 0.001,
#     "sparsity_penalty": 20.0,
#     "warmup_steps": 1000,
#     "sparsity_warmup_steps": 2000,
#     "decay_start": None,
#     "target_l0": 60.0,
#     "wandb_name": "JumpRelu",
# }

# #####################################
# # 4. Train the JumpReLU SAE          #
# #####################################

# # The trainSAE function expects:
# #   - data: an iterator that yields batches of activations,
# #   - trainer_configs: a list of trainer configuration dicts,
# #   - steps: total training steps, etc.
# ae = trainSAE(
#     data=data_iterator,             # our infinite iterator over activation batches
#     trainer_configs=[trainer_cfg],
#     steps=total_steps,
#     use_wandb=False,                # set to True to use Weights & Biases logging
#     log_steps=1000,                  # log every 100 steps
#     normalize_activations=True, #False,    # set True if you wish to normalize activations
#     verbose=True,
#     device=device,
#     save_dir="./"
# )

# print("Training completed!")

#!/usr/bin/env python
import math
import torch
import einops
from datasets import load_dataset

# Import our ActivationBuffer from activations.py
from activations import ActivationBuffer
# Import the gemma model
from sae_lens import HookedSAETransformer
# Import SAE training components
from dictionary import JumpReluAutoEncoder
from jumprelu import JumpReluTrainer
from training import trainSAE

# ------------------------------
# 1. Load the HF Dataset & Determine Total Examples
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the dataset in non-streaming mode so we can get its length.
ds = load_dataset("charlieoneill/medical-qa-combined", split="train", streaming=False)
num_examples = len(ds)
print(f"Total examples in dataset: {num_examples}")

# Create a finite generator that yields texts from the dataset.
def finite_generator(dataset):
    for example in dataset:
        yield example["text"]

data_generator = finite_generator(ds)

# ------------------------------
# 2. Load gemma-2-2b and Create the ActivationBuffer
# ------------------------------
print("Loading gemma-2-2b model...")
model_hooked = HookedSAETransformer.from_pretrained("gemma-2-2b", device=device)
torch.cuda.empty_cache()

# Parameters for tokenization/activation extraction.
ctx_len = 256        # maximum sequence length per text
activation_dim = 2304  # gemma-2-2b's model dimension
# Choose your training batch size (number of contexts yielded per __next__)
out_batch_size = 1024

# Instantiate the ActivationBuffer.
buffer = ActivationBuffer(
    data=data_generator,
    model=model_hooked,
    submodule=None,        # not used in our current setup
    d_submodule=activation_dim,
    n_ctxs=100,            # buffer capacity (number of contexts stored)
    ctx_len=ctx_len,
    refresh_batch_size=16, # how many texts to process at a time during refresh
    out_batch_size=out_batch_size,
    device=device,
    remove_bos=False
)

# Initial fill of the buffer.
print("Refreshing activation buffer...")
try:
    buffer.refresh()
except StopIteration:
    print("Reached end of dataset during initial buffer fill.")

# ------------------------------
# 3. Compute Total Training Steps
# ------------------------------
# Each text in the dataset produces one activation context.
# With out_batch_size contexts per batch, the total number of batches is:
total_steps = math.ceil(num_examples / out_batch_size)
print(f"Total training steps (batches): {total_steps}")

# ------------------------------
# 4. Create a Finite Iterator Over the Buffer
# ------------------------------
def finite_buffer_iterator(buffer):
    """Yield batches from the buffer until the underlying data generator is exhausted."""
    while True:
        try:
            yield next(buffer)
        except StopIteration:
            break

# The ActivationBuffer yields tensors of shape (B, ctx_len, activation_dim).
data_iterator = finite_buffer_iterator(buffer)

# ------------------------------
# 5. Flatten Each Batch (Optional)
# ------------------------------
# Your old training code flattened the sequence dimension so that each token is independent.
# This converts a batch from shape (B, 256, 2304) to ((B*256), 2304).
def flatten_batch(batch):
    return einops.rearrange(batch, "b s d -> (b s) d")

def flattened_data_iterator(data_iter):
    for batch in data_iter:
        yield flatten_batch(batch)

flattened_iterator = flattened_data_iterator(data_iterator)

# ------------------------------
# 6. Configure Trainer Parameters & Train SAE
# ------------------------------
d_model = activation_dim
dictionary_size = 16384  # e.g. choose as needed

trainer_cfg = {
    "trainer": JumpReluTrainer,           # SAE trainer class
    "dict_class": JumpReluAutoEncoder,      # SAE model class
    "steps": total_steps,                   # total number of training steps (batches)
    "activation_dim": d_model,
    "dict_size": dictionary_size,
    "layer": 0,                           # (choose an appropriate layer if needed)
    "lm_name": "precomputed",             # arbitrary name since no LM is used here
    "lr": 7e-5,
    "device": device,
    "bandwidth": 0.001,
    "sparsity_penalty": 20.0,
    "warmup_steps": 1000,
    "sparsity_warmup_steps": 2000,
    "decay_start": None,
    "target_l0": 60.0,
    "wandb_name": "JumpRelu",
}

print(f"Training for {total_steps} steps with batch size {out_batch_size} (pre-flattening)")

ae = trainSAE(
    data=flattened_iterator,         # iterator over flattened activation batches
    trainer_configs=[trainer_cfg],
    steps=total_steps,
    use_wandb=False,                 # set to True to enable Weights & Biases logging
    log_steps=1000,
    normalize_activations=True,      # or False, as desired
    verbose=True,
    device=device,
    save_dir="./"
)

print("Training completed!")