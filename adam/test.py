from nnsight import LanguageModel
from buffer import ActivationBuffer
from dictionary import AutoEncoder, JumpReluAutoEncoder
from standard import StandardTrainer
from jumprelu import JumpReluTrainer
from training import trainSAE
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")    
model_name = "EleutherAI/pythia-70m-deduped" # can be any Huggingface model

model = LanguageModel(
    model_name,
    device_map=device,
)
submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data must be an iterator that outputs strings
data = iter(
    [
        "This is some example data",
        "In real life, for training a dictionary",
        "you would need much more data than this",
    ]
)
buffer = ActivationBuffer(
    data=data,
    model=model,
    submodule=submodule,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=3e4,  # you can set this higher or lower dependong on your available memory
    device=device,
)  # buffer will yield batches of tensors of dimension = submodule's output dimension

trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-3,
    "device": device,
    "steps": 10_000,
    "layer": 2,
    "lm_name": model_name,
    # "warmup_steps": 1000,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    steps=10_000,
    data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
)