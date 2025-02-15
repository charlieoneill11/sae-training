import torch as t
from nnsight import LanguageModel
import itertools
import os
import random
import json
import torch.multiprocessing as mp
import time
import huggingface_hub
from datasets import config, load_dataset

import demo_config
# from buffer import ActivationBuffer
# from evaluation import evaluate
# from training import trainSAE
# import utils as utils

from utils import hf_dataset_to_generator
from buffer import ActivationBuffer
from evaluation import evaluate
from training import trainSAE
import utils as utils

from huggingface_hub import login

HF_TOKEN = "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"  # Replace with your token.
login(token=HF_TOKEN)

# Configuration parameters that were previously command line arguments
CONFIG = {
    "save_dir": "./run3",
    "use_wandb": False,
    "dry_run": False,
    "save_checkpoints": False,
    "layers": [20],  # Can be modified to include multiple layers
    "model_name": "google/gemma-2-2b",
    "architectures": ["jump_relu"],  # Can include multiple architectures
    "device": "cuda:0",
    "hf_repo_id": "charlieoneill/gemma-medicine-sae"  # Set this if you want to push to HuggingFace
}

def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    """
    Creates a generator from a Hugging Face dataset. Each iteration yields the 'text' field.
    """
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    def gen():
        for x in dataset:
            yield x["text"]
    return gen()

def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
):
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    # model and data parameters
    context_length = demo_config.LLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    num_buffer_inputs = buffer_tokens // context_length
    print(f"buffer_size: {num_buffer_inputs}, buffer_size_in_tokens: {buffer_tokens}")

    log_steps = 100  # Log the training on wandb or print to console every log_steps
    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train

    if save_checkpoints:
        desired_checkpoints = t.logspace(-3, 0, 4).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    submodule = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    generator = hf_dataset_to_generator("charlieoneill/medical-qa-combined") #"monology/pile-uncopyrighted")

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=num_buffer_inputs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
    )

    trainer_configs = demo_config.get_trainer_configs(
        architectures,
        learning_rates,
        random_seeds,
        activation_dim,
        dictionary_widths,
        model_name,
        device,
        layer,
        submodule_name,
        steps,
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            wandb_project=demo_config.wandb_project,
            normalize_activations=True,
            verbose=True,
            autocast_dtype=t.bfloat16,
        )

@t.no_grad()
def eval_saes(
    model_name: str,
    ae_paths: list[str],
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
) -> dict:
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    context_length = demo_config.LLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    sae_batch_size = loss_recovered_batch_size * context_length
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)

    buffer_size = n_inputs
    io = "out"
    n_batches = n_inputs // loss_recovered_batch_size

    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i > n_inputs * 5:
            break

    eval_results = {}

    for ae_path in ae_paths:
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} as eval results already exist")
                continue

        dictionary, config = utils.load_dictionary(ae_path, device)
        dictionary = dictionary.to(dtype=model.dtype)

        layer = config["trainer"]["layer"]
        submodule = utils.get_submodule(model, layer)

        activation_dim = config["trainer"]["activation_dim"]

        activation_buffer = ActivationBuffer(
            iter(input_strings),
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
        )

        eval_results = evaluate(
            dictionary,
            activation_buffer,
            context_length,
            loss_recovered_batch_size,
            io=io,
            device=device,
            n_batches=n_batches,
        )

        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
        }
        eval_results["hyperparameters"] = hyperparameters

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

    # return the final eval_results for testing purposes
    return eval_results

def push_to_huggingface(save_dir: str, repo_id: str):
    api = huggingface_hub.HfApi()
    
    # Create repo if it doesn't exist
    if not huggingface_hub.repo_exists(repo_id=repo_id, repo_type="model"):
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True  # Set to False if you want a public repository
        )
    
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=save_dir,
    )


# Main execution code
def run_training():
    hf_repo_id = CONFIG["hf_repo_id"]

    # if hf_repo_id:
    #     assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # For wandb to work with multiprocessing
    mp.set_start_method("spawn", force=True)

    # Rarely I have internet issues on cloud GPUs and then the streaming read fails
    config.STREAMING_READ_MAX_RETRIES = 100
    config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()

    save_dir = f"{CONFIG['save_dir']}_{CONFIG['model_name']}_{'_'.join(CONFIG['architectures'])}".replace("/", "_")

    for layer in CONFIG['layers']:
        run_sae_training(
            model_name=CONFIG['model_name'],
            layer=layer,
            save_dir=save_dir,
            device=CONFIG['device'],
            architectures=CONFIG['architectures'],
            num_tokens=demo_config.num_tokens,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=demo_config.dictionary_widths,
            learning_rates=demo_config.learning_rates,
            dry_run=CONFIG['dry_run'],
            use_wandb=CONFIG['use_wandb'],
            save_checkpoints=CONFIG['save_checkpoints'],
        )

    ae_paths = utils.get_nested_folders(save_dir)
    print(f"Total time: {time.time() - start_time}")

    if hf_repo_id:
        push_to_huggingface(save_dir, hf_repo_id)

run_training()