# import os
# import json
# import torch as t
# from huggingface_hub import snapshot_download, login
# from demo import eval_saes
# import demo_config
# import utils  # Ensure this contains load_dictionary

# def main():
#     # The Hugging Face repo ID where the trained SAEs were pushed.
#     hf_repo_id = "charlieoneill/gemma-medicine-sae"

#     # Download the repository snapshot locally.
#     print("Downloading repository from Hugging Face...")
#     local_repo = snapshot_download(repo_id=hf_repo_id, repo_type="model")
#     print(f"Repository downloaded to: {local_repo}")

#     # Define the run folder that was used during training.
#     # This should be "._run3_google_gemma-2-2b_jump_relu/resid_post_layer_20".
#     run_folder = os.path.join(local_repo, "._run3_google_gemma-2-2b_jump_relu", "resid_post_layer_20")
#     if not os.path.exists(run_folder):
#         raise FileNotFoundError(f"Run folder not found: {run_folder}")

#     # List all trainer folders (trainer_0 to trainer_5) within the run folder.
#     trainer_folders = [
#         os.path.join(run_folder, d)
#         for d in os.listdir(run_folder)
#         if d.startswith("trainer_") and os.path.isdir(os.path.join(run_folder, d))
#     ]
#     trainer_folders.sort()  # Ensure proper order

#     # Evaluation parameters.
#     model_name = "google/gemma-2-2b"
#     device = "cuda:0" if t.cuda.is_available() else "cpu"
#     n_inputs = 200  # Adjust as needed
#     overwrite_prev_results = True

#     combined_results = {}

#     # Evaluate each SAE in the trainer folders.
#     for trainer_path in trainer_folders:
#         trainer_name = os.path.basename(trainer_path)
#         print(f"Evaluating SAE in folder: {trainer_path}")
#         eval_result = eval_saes(
#             model_name=model_name,
#             ae_paths=[trainer_path],
#             n_inputs=n_inputs,
#             device=device,
#             overwrite_prev_results=overwrite_prev_results,
#             transcoder=False  # Set to True if you need to evaluate both input and output activations.
#         )
#         combined_results[trainer_name] = eval_result
#         print(f"Results for {trainer_name}: {eval_result}\n")
    
#     # Extract additional details (layer, SAE width) from one trainer's config.
#     # We assume all trainers were run with the same settings.
#     _, config = utils.load_dictionary(trainer_folders[0], device)
#     layer = config["trainer"]["layer"]
#     width = config["trainer"]["activation_dim"]

#     # Create an output filename that reflects the evaluation settings.
#     output_filename = f"eval_results_layer{layer}_width{width}_ninputs{n_inputs}.json"
#     with open(output_filename, "w") as f:
#         json.dump(combined_results, f, indent=2)
#     print(f"Combined evaluation results saved to {output_filename}")

# if __name__ == "__main__":
#     HF_TOKEN = "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"  # Replace with your token.
#     login(token=HF_TOKEN)
#     main()

import os
import json
import torch as t
import numpy as np
from itertools import cycle
from huggingface_hub import login
from nnsight import LanguageModel
from buffer import ActivationBuffer
from demo import evaluate, hf_dataset_to_generator  # Your existing evaluation and dataset functions
import demo_config
import utils  # for get_submodule

# Import JumpReluAutoEncoder from your dictionary module.
from dictionary import JumpReluAutoEncoder

def main():
    # Login to Hugging Face.
    HF_TOKEN = "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"  # Replace with your token.
    login(token=HF_TOKEN)

    # Evaluation parameters.
    model_name = "google/gemma-2-2b"  # LLM used to extract activations.
    device = "cuda:0" if t.cuda.is_available() else "cpu"
    n_inputs = 200  # Number of inputs for evaluation.
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    # The list of L0 values we want to evaluate.
    l0_values = [139, 22, 294, 38, 71]
    layer = 20
    width = "16k"
    hf_repo_id = "google/gemma-scope-2b-pt-res"  # Repository with gemma‑scope SAEs.

    # Create the language model (LLM) and get the appropriate submodule.
    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    submodule = utils.get_submodule(model, layer)

    # Evaluation settings.
    context_length = demo_config.LLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    sae_batch_size = loss_recovered_batch_size * context_length
    n_batches = n_inputs // loss_recovered_batch_size

    print(f"context_length: {context_length}, llm_batch_size: {llm_batch_size}, sae_batch_size: {sae_batch_size}")

    # Build a list of input texts.
    generator = hf_dataset_to_generator(
        dataset_name="charlieoneill/medical-qa-combined",
        split="train",
        streaming=True
    )
    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i >= n_inputs * 5:
            break

    combined_results = {}
    # Loop over each L0 value.
    for l0 in l0_values:
        print(f"\nEvaluating gemma‑scope SAE for L0={l0} ...")
        # Load the SAE using the new from_npz method.
        sae = JumpReluAutoEncoder.from_npz(hf_repo_id, layer, width, l0, dtype=dtype, device=device)
        
        # Build an ActivationBuffer using a cycling iterator.
        activation_buffer = ActivationBuffer(
            cycle(input_strings),
            model,
            submodule,
            n_ctxs=n_inputs,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io="out",
            d_submodule=sae.activation_dim,
            device=device,
        )
        
        # Run evaluation.
        eval_results = evaluate(
            sae,  # The SAE acts as the dictionary.
            activation_buffer,
            max_len=context_length,
            batch_size=loss_recovered_batch_size,
            io="out",
            device=device,
            n_batches=n_batches,
        )
        # Add hyperparameters and model info.
        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
            "l0": l0,
            "layer": layer,
            "width": width,
        }
        eval_results["hyperparameters"] = hyperparameters
        combined_results[f"l0_{l0}"] = eval_results
        print(f"Results for L0={l0}: {eval_results}\n")

    # Save all evaluation results to a single JSON file.
    l0_str = "_".join(str(x) for x in l0_values)
    output_filename = f"eval_results_gemmascope_layer{layer}_width{width}_l0s_{l0_str}_ninputs{n_inputs}.json"
    with open(output_filename, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"Combined evaluation results saved to {output_filename}")

if __name__ == "__main__":
    main()