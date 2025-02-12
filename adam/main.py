#!/usr/bin/env python
"""
Script to train a JumpReLU sparse autoencoder (SAE) using precomputed activations.

This script:
  - Loads a tensor of activations from Hugging Face Hub.
  - Wraps the activations in a simple Dataset/DataLoader so that batches are produced.
  - Configures the training (using JumpReluTrainer and JumpReluAutoEncoder) and calls trainSAE.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import einops
from huggingface_hub import hf_hub_download, HfApi

# Import the JumpReLU SAE and trainer from the repository.
from dictionary import JumpReluAutoEncoder
from jumprelu import JumpReluTrainer
from training import trainSAE

# Flush cache
torch.cuda.empty_cache()

#!/usr/bin/env python
"""
Script to train a JumpReLU sparse autoencoder (SAE) using precomputed activations.

This script:
  - Loads a tensor of activations from Hugging Face Hub.
  - Wraps the activations in a simple Dataset/DataLoader so that batches are produced.
  - Configures the training (using JumpReluTrainer and JumpReluAutoEncoder) and calls trainSAE.
  
Note: This script assumes that the repository's modules (e.g. dictionary_learning, trainers,
and training) are available in your PYTHONPATH.
"""

#####################################
# 1. Load Precomputed Activations   #
#####################################


# Device selection.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    # Download the activation tensor (expected shape: [10000, 128, 2304])
    repo_id = "charlieoneill/gemma-medicine-sae"  # adjust if needed
    activation_file = hf_hub_download(repo_id=repo_id, filename="10000_128.pt")
    activations = torch.load(activation_file)
    print(f"Loaded activations with shape: {activations.shape}")
except Exception as e:
    print(f"Error loading activations: {e}")
    # Fall back to random data (for debugging/testing)
    activations = torch.randn(10000, 128, 2304)
    print(f"Using random activations with shape: {activations.shape}")

# --- Optional: Flatten the sequence dimension ---
# Uncomment these lines if you wish to treat every token independently.
activations = einops.rearrange(activations, "b s d -> (b s) d")
print(f"Flattened activations shape: {activations.shape}")

# Shuffle activations along first dimension
activations = activations[torch.randperm(activations.size(0))]

#####################################
# 2. Create a DataLoader for Activations  #
#####################################

class ActivationDataset(Dataset):
    def __init__(self, activations: torch.Tensor):
        """
        activations: a torch.Tensor of shape (N, d_model) OR (N, seq_len, d_model)
        Keep activations on CPU until needed
        """
        self.activations = activations.pin_memory() if torch.cuda.is_available() else activations

    def __len__(self):
        return self.activations.size(0)

    def __getitem__(self, idx):
        # Only transfer the specific item when requested
        return self.activations[idx]

# Create the dataset without moving to GPU
dataset = ActivationDataset(activations)  # activations stays on CPU
dataset.config = {"d_submodule": activations.shape[-1]}

# Create a DataLoader with memory-efficient settings
batch_size = 1024
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,  # This helps with faster CPU->GPU transfers
    num_workers=4,    # Adjust based on your CPU cores
    prefetch_factor=2 # Each worker will prefetch 2 batches
)

# Define an infinite iterator over the DataLoader.
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

data_iterator = infinite_loader(data_loader)

#####################################
# 3. Configure Training Parameters  #
#####################################

# Activation dimension (d_model) is the last dimension of the activations.
d_model = activations.shape[-1]  # e.g. 2304
# Set dictionary (latent) size; here we use 16 * d_model (adjust as needed)
dictionary_size = 16384 #16 * d_model
# Total training steps.
total_steps = 10_000

print(f"Training for {total_steps} steps with batch size {batch_size} ({total_steps*batch_size} total tokens)")

# Trainer configuration dictionary.
trainer_cfg = {
    "trainer": JumpReluTrainer,           # Use the JumpReLU trainer
    "dict_class": JumpReluAutoEncoder,      # Use the JumpReLU autoencoder
    "steps": total_steps,                   # required: total training steps
    "activation_dim": d_model,
    "dict_size": dictionary_size,
    "layer": 0,                           # arbitrary value (you can choose a meaningful layer if you wish)
    "lm_name": "precomputed",             # arbitrary value (since no language model is used)
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

#####################################
# 4. Train the JumpReLU SAE          #
#####################################

# The trainSAE function expects:
#   - data: an iterator that yields batches of activations,
#   - trainer_configs: a list of trainer configuration dicts,
#   - steps: total training steps, etc.
ae = trainSAE(
    data=data_iterator,             # our infinite iterator over activation batches
    trainer_configs=[trainer_cfg],
    steps=total_steps,
    use_wandb=False,                # set to True to use Weights & Biases logging
    log_steps=1000,                  # log every 100 steps
    normalize_activations=True, #False,    # set True if you wish to normalize activations
    verbose=True,
    device=device,
    save_dir="./"
)

print("Training completed!")