import os
import json
import torch as t
from huggingface_hub import snapshot_download, login
from demo import eval_saes
import demo_config
import utils  # Ensure this contains load_dictionary

def main():
    # The Hugging Face repo ID where the trained SAEs were pushed.
    hf_repo_id = "charlieoneill/gemma-medicine-sae"

    # Download the repository snapshot locally.
    print("Downloading repository from Hugging Face...")
    local_repo = snapshot_download(repo_id=hf_repo_id, repo_type="model")
    print(f"Repository downloaded to: {local_repo}")

    # Define the run folder that was used during training.
    # This should be "._run3_google_gemma-2-2b_jump_relu/resid_post_layer_20".
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

    # Evaluation parameters.
    model_name = "google/gemma-2-2b"
    device = "cuda:0" if t.cuda.is_available() else "cpu"
    n_inputs = 200  # Adjust as needed
    overwrite_prev_results = True

    combined_results = {}

    # Evaluate each SAE in the trainer folders.
    for trainer_path in trainer_folders:
        trainer_name = os.path.basename(trainer_path)
        print(f"Evaluating SAE in folder: {trainer_path}")
        eval_result = eval_saes(
            model_name=model_name,
            ae_paths=[trainer_path],
            n_inputs=n_inputs,
            device=device,
            overwrite_prev_results=overwrite_prev_results,
            transcoder=False,  # Set to True if you need to evaluate both input and output activations.
        )
        combined_results[trainer_name] = eval_result
        print(f"Results for {trainer_name}: {eval_result}\n")
    
    # Extract additional details (layer, SAE width) from one trainer's config.
    # We assume all trainers were run with the same settings.
    _, config = utils.load_dictionary(trainer_folders[0], device)
    layer = config["trainer"]["layer"]
    width = config["trainer"]["activation_dim"]
    
    # Round width to nearest 1000 and convert to k format
    width_rounded = round(width/1000)*1000
    width_k = f"{width_rounded//1000}k"

    # Create an output filename that reflects the evaluation settings.
    output_filename = f"eval_results_layer{layer}_width{width_k}_ninputs{n_inputs}.json"
    with open(output_filename, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"Combined evaluation results saved to {output_filename}")

if __name__ == "__main__":
    HF_TOKEN = "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"  # Replace with your token.
    login(token=HF_TOKEN)
    main()

import os
import json
import torch as t
import numpy as np
from itertools import cycle
from collections import defaultdict
from huggingface_hub import login
from nnsight import LanguageModel
from buffer import ActivationBuffer
from demo import hf_dataset_to_generator
from tqdm.auto import tqdm
from evaluation import loss_recovered
import demo_config
import utils  # for get_submodule

# Import JumpReluAutoEncoder from your dictionary module.
from dictionary import JumpReluAutoEncoder

def evaluate_individual(sae, model, submodule, activation_buffer, context_length, num_texts):
    """
    Evaluate the SAE on individual texts from the dataset.
    For each text, we obtain activations using the same procedure as in the buffer:
      - Use model.trace over a text batch (of size 1)
      - Save the submodule activations and model inputs
      - Apply attention mask filtering (and remove BOS if needed)
    Then, pass the resulting activations through the SAE to compute reconstruction metrics,
    and call loss_recovered on that text.
    Returns a dictionary of averaged metrics.
    """
    stats = defaultdict(float)
    active_features = t.zeros(sae.dict_size, dtype=t.float32, device=model.device)
    lr_orig_accum = 0.0
    lr_recon_accum = 0.0
    lr_zero_accum = 0.0
    frac_recov_accum = 0.0
    count = 0

    # Use the same tracer kwargs as in the ActivationBuffer.
    tracer_kwargs = {'scan': False, 'validate': False}

    for i in tqdm(range(num_texts)):
        # Get one text from the activation buffer.
        # text_batch(batch_size=1) returns a list with one text.
        texts = activation_buffer.text_batch(batch_size=1)
        text = texts[0]
        
        # Use model.trace to extract activations for this text.
        with t.no_grad():
            with model.trace(
                [text],
                **tracer_kwargs,
                invoker_args={"truncation": True, "max_length": context_length},
            ):
                # Depending on the io setting, save the appropriate activations.
                if activation_buffer.io == "in":
                    hidden_states = submodule.inputs[0].save()
                else:
                    hidden_states = submodule.output.save()
                # Also save the model inputs.
                input_obj = model.inputs.save()
                # Stop tracing for the submodule.
                submodule.output.stop()
            # Process the saved outputs.
            attn_mask = input_obj.value[1]["attention_mask"]
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            if activation_buffer.remove_bos:
                hidden_states = hidden_states[:, 1:, :]
                attn_mask = attn_mask[:, 1:]
            # Flatten the tokens using the attention mask.
            x = hidden_states[attn_mask != 0]

        # Pass the extracted activations through the SAE.
        x_hat, f = sae(x, output_features=True)
        # (Optional) Print shapes for debugging.
        #print(f"Text {i}: x shape: {x.shape}, x_hat shape: {x_hat.shape}, f shape: {f.shape}")
        
        # Compute reconstruction metrics.
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()
        
        # Flatten f to accumulate active features.
        features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32)
        active_features += features_BF.sum(dim=0)
        
        # Cosine similarity between x and x_hat.
        x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()
        
        # l2 ratio.
        l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()
        
        # Variance explained.
        total_variance = t.var(x, dim=0).sum()
        residual_variance = t.var(x - x_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
        
        # Relative reconstruction bias (Equation 10).
        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()
        
        # Accumulate these metrics.
        stats["l2_loss"] += l2_loss.item()
        stats["l1_loss"] += l1_loss.item()
        stats["l0"] += l0.item()
        stats["frac_variance_explained"] += frac_variance_explained.item()
        stats["cossim"] += cossim.item()
        stats["l2_ratio"] += l2_ratio.item()
        stats["relative_reconstruction_bias"] += relative_reconstruction_bias.item()
        
        # Compute loss recovered on this individual text.
        # loss_recovered expects a batch of texts, so we pass [text].
        lr_orig, lr_recon, lr_zero = loss_recovered(
            [text],
            model,
            submodule,
            sae,
            max_len=context_length,
            normalize_batch=False,
            io="out"
        )
        frac_recov = (lr_recon - lr_zero) / (lr_orig - lr_zero)
        lr_orig_accum += lr_orig.item()
        lr_recon_accum += lr_recon.item()
        lr_zero_accum += lr_zero.item()
        frac_recov_accum += frac_recov.item()
        
        count += 1

    # Average the metrics over all texts.
    for key in stats:
        stats[key] /= count
    stats["loss_original"] = lr_orig_accum / count
    stats["loss_reconstructed"] = lr_recon_accum / count
    stats["loss_zero"] = lr_zero_accum / count
    stats["frac_recovered"] = frac_recov_accum / count

    # Compute the fraction of dictionary entries that were ever used.
    frac_alive = (active_features != 0).float().sum() / sae.dict_size
    stats["frac_alive"] = frac_alive.item()

    return dict(stats)

def main():
    # Log in to Hugging Face.
    HF_TOKEN = "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"  # Replace with your token.
    login(token=HF_TOKEN)

    # Evaluation parameters.
    model_size = "9b"
    model_name = f"google/gemma-2-{model_size}"  # LLM used to extract activations.
    device = "cuda:0" if t.cuda.is_available() else "cpu"
    n_inputs = 200  # Number of input texts for evaluation.
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    # The list of L0 values we want to evaluate.
    l0_values = [11, 344, 57]
    layer = 20
    width = "32k"
    hf_repo_id = f"google/gemma-scope-{model_size}-pt-res"  # Repository with gemma‑scope SAEs.

    # Create the language model (LLM) and get the appropriate submodule.
    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    submodule = utils.get_submodule(model, layer)

    # Evaluation settings.
    context_length = demo_config.LLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    print(f"context_length: {context_length}, llm_batch_size: {llm_batch_size}")

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
        # Load the SAE using the from_npz method.
        sae = JumpReluAutoEncoder.from_npz(hf_repo_id, layer, width, l0, dtype=dtype, device=device)
        sae.eval()
        
        # Build an ActivationBuffer (we use it only for its text stream and submodule reference).
        activation_buffer = ActivationBuffer(
            cycle(input_strings),
            model,
            submodule,
            n_ctxs=n_inputs,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            # out_batch_size is not used in the individual evaluation.
            out_batch_size=llm_batch_size,
            io="out",
            d_submodule=sae.activation_dim,
            device=device,
            remove_bos=True,
        )
        
        # Evaluate individually for n_inputs texts.
        individual_results = evaluate_individual(sae, model, submodule, activation_buffer, context_length, n_inputs)
        
        # Add hyperparameters and model info.
        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
            "l0": l0,
            "layer": layer,
            "width": width,
        }
        individual_results["hyperparameters"] = hyperparameters
        combined_results[f"l0_{l0}"] = individual_results
        print(f"Results for L0={l0}: {individual_results}\n")

    # Save all evaluation results to a single JSON file.
    output_filename = f"eval_results_gemmascope_layer{layer}_width{width}_model{model_size}_ninputs{n_inputs}.json"
    with open(output_filename, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"Combined evaluation results saved to {output_filename}")

if __name__ == "__main__":
    main()