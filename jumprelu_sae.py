# import torch as t
# from torch import Tensor
# from typing import Any
# from torch.distributions.categorical import Categorical
# from jaxtyping import Float
# import torch.nn as nn
# import torch.nn.functional as F
# import einops

# ### our imports ###
# from base_sae import SAE, SAEConfig


# def rectangle(x: Tensor, width: float = 1.0) -> Tensor:
#     """
#     Returns the rectangle function value, i.e. K(x) = 1[|x| < width/2], as a float.
#     """
#     return (x.abs() < width / 2).float()


# class Heaviside(t.autograd.Function):
#     """
#     Implementation of the Heaviside step function, using straight through estimators for the derivative.

#         forward:
#             H(z,θ,ε) = 1[z > θ]

#         backward:
#             dH/dz := None
#             dH/dθ := -1/ε * K(z/ε)

#             where K is the rectangle kernel function with width 1, centered at 0: K(u) = 1[|u| < 1/2]
#     """

#     @staticmethod
#     def forward(ctx: Any, z: t.Tensor, theta: t.Tensor, eps: float) -> t.Tensor:
#         # Save any necessary information for backward pass
#         ctx.save_for_backward(z, theta)
#         ctx.eps = eps
#         # Compute the output
#         return (z > theta).float()

#     @staticmethod
#     def backward(ctx: Any, grad_output: t.Tensor) -> t.Tensor:
#         # Retrieve saved tensors & values
#         (z, theta) = ctx.saved_tensors
#         eps = ctx.eps
#         # Compute gradient of the loss with respect to z (no STE) and theta (using STE)
#         grad_z = 0.0 * grad_output
#         grad_theta = -(1.0 / eps) * rectangle((z - theta) / eps) * grad_output
#         grad_theta_agg = grad_theta.sum(dim=0)  # note, sum over batch dim isn't strictly necessary

#         # # ! DEBUGGING
#         # # nz = rectangle((z - theta) / eps) > 0
#         # print(f"HEAVISIDE\nNumber of nonzero grads? {(grad_theta.abs() > 1e-6).float().mean():.3f}")
#         # print(f"Average positive (of non-zero grads): {grad_theta[grad_theta.abs() > 1e-6].mean():.3f}")

#         return grad_z, grad_theta_agg, None
    

# class JumpReLU(t.autograd.Function):
#     """
#     Implementation of the JumpReLU function, using straight through estimators for the derivative.

#         forward:
#             J(z,θ,ε) = z * 1[z > θ]

#         backward:
#             dJ/dθ := -θ/ε * K(z/ε)
#             dJ/dz := 1[z > θ]

#             where K is the rectangle kernel function with width 1, centered at 0: K(u) = 1[|u| < 1/2]
#     """

#     @staticmethod
#     def forward(ctx: Any, z: t.Tensor, theta: t.Tensor, eps: float) -> t.Tensor:
#         # Save relevant stuff for backwards pass
#         ctx.save_for_backward(z, theta)
#         ctx.eps = eps
#         # Compute the output
#         return z * (z > theta).float()


#     @staticmethod
#     def backward(ctx: Any, grad_output: t.Tensor) -> t.Tensor:
#         # Retrieve saved tensors & values
#         (z, theta) = ctx.saved_tensors
#         eps = ctx.eps
#         # Compute gradient of the loss with respect to z (no STE) and theta (using STE)
#         grad_z = (z > theta).float() * grad_output
#         grad_theta = -(theta / eps) * rectangle((z - theta) / eps) * grad_output
#         grad_theta_agg = grad_theta.sum(dim=0)  # note, sum over batch dim isn't strictly necessary
#         return grad_z, grad_theta_agg, None
    
# THETA_INIT = 0.1

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Any, Callable, Optional
# from tqdm import tqdm

# class JumpReLUSAE(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         d_sae: int,
#         sparsity_coeff: float = 1.0,
#         threshold_init: float = 0.1,
#         ste_epsilon: float = 1e-2,
#         weight_normalize_eps: float = 1e-8
#     ):
#         """
#         Args:
#             d_model: Dimension of input (and reconstruction).
#             d_sae: Dimension of the latent space.
#             sparsity_coeff: Weighting for the sparsity penalty.
#             threshold_init: Initial threshold value.
#             ste_epsilon: STE epsilon (not used in this simple version).
#         """
#         super().__init__()

#         # Encoder: shape [d_model, d_sae]
#         self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
#         nn.init.kaiming_uniform_(self.W_enc, a=nn.init.calculate_gain('relu'))

#         # Decoder: shape [d_sae, d_model]
#         self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
#         nn.init.kaiming_uniform_(self.W_dec, a=nn.init.calculate_gain('relu'))

#         # Threshold for gating activations (one per latent unit)
#         self.threshold = nn.Parameter(torch.full((d_sae,), threshold_init))
#         self.log_theta = nn.Parameter(torch.full((d_sae,), torch.log(torch.tensor(threshold_init))))

#         # Biases for encoder and decoder
#         self.b_enc = nn.Parameter(torch.zeros(d_sae))
#         self.b_dec = nn.Parameter(torch.zeros(d_model))

#         self.d_model = d_model
#         self.d_sae = d_sae
#         self.sparsity_coeff = sparsity_coeff
#         self.ste_epsilon = ste_epsilon
#         self.weight_normalize_eps = weight_normalize_eps

#     @property
#     def theta(self) -> Float[torch.Tensor, "d_sae"]:
#         return self.log_theta.exp()

#     def encode(self, input_acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Applies a linear transform, ReLU, and gating based on the threshold.
        
#         Args:
#             input_acts: Tensor of shape (..., d_model)
#         Returns:
#             acts: Gated latent activations (shape: (..., d_sae))
#             pre_acts: Pre-activation values.
#         """
#         # pre_acts = input_acts @ self.W_enc + self.b_enc
#         # relu_acts = F.relu(pre_acts)
#         # # The threshold is broadcast along the batch dimensions.
#         # mask = (pre_acts > self.threshold).float()
#         # acts = relu_acts * mask
#         # return acts, pre_acts
#         h_cent = input_acts - self.b_dec

#         pre_acts = (
#             einops.einsum(h_cent, self.W_enc, "batch d_model, d_model d_sae -> batch d_sae") + self.b_enc
#         )
#         acts_relu = F.relu(pre_acts)
#         acts = JumpReLU.apply(acts_relu, self.theta, self.ste_epsilon)
#         return acts, pre_acts
    
#     def decode(self, acts: torch.Tensor) -> torch.Tensor:
#         """
#         Reconstructs the input from latent activations.
        
#         Args:
#             acts: Tensor of shape (..., d_sae)
#         Returns:
#             Reconstructed input of shape (..., d_model)
#         """
#         # return acts @ self.W_dec + self.b_dec
#         h_reconstructed = (
#             einops.einsum(acts, self.W_dec, "batch d_sae, d_sae d_in -> batch d_in") + self.b_dec
#         )
#         return h_reconstructed


#     def forward(self, input_acts: torch.Tensor) -> torch.Tensor:
#         """
#         Full pass: encode then decode.
        
#         Args:
#             input_acts: Tensor of shape (..., d_model)
#         Returns:
#             Reconstruction of the input.
#         """
#         acts, _ = self.encode(input_acts)
#         recon = self.decode(acts)
#         return recon

#     # --- Loss Functions ---
#     def compute_reconstruction_loss(self, input_acts: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
#         """
#         Computes mean-squared error per example.
#         """
#         #return ((recon - input_acts) ** 2).mean(dim=-1)
#         return(recon - input_acts).pow(2).mean(-1)

#     def compute_sparsity_loss(self, pre_acts: torch.Tensor) -> torch.Tensor:
#         """
#         Computes the sparsity loss by measuring the fraction of activations exceeding the threshold.
#         For a 3D tensor (batch, seq_len, d_sae) we average over batch and sequence; for 2D input,
#         we average only over the batch.
#         """
#         # mask = (pre_acts > self.threshold).float()
#         # if pre_acts.dim() == 3:
#         #     return mask.mean(dim=(0, 1)).sum()  # one value per latent dimension
#         # else:
#         #     return mask.mean(dim=0).sum()
#         return Heaviside.apply(pre_acts, self.theta, self.ste_epsilon).sum(-1)

#     def compute_loss(self, input_acts: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
#         """
#         Computes the total loss: reconstruction + sparsity penalty.
        
#         Args:
#             input_acts: Input tensor of shape (batch, d_model) or (batch, seq_len, d_model)
#         Returns:
#             total_loss: Per-example loss (if input is batched) plus the sparsity penalty.
#             loss_dict: Dictionary with individual loss components.
#         """
#         acts, pre_acts = self.encode(input_acts)
#         recon = self.decode(acts)
#         L_reconstruction = self.compute_reconstruction_loss(input_acts, recon)
#         L_sparsity = self.compute_sparsity_loss(pre_acts)
#         # Sum over latent dimensions (if L_sparsity is a vector) and add to reconstruction loss.
#         #print(f"L_sparsity: {L_sparsity}, sparsity coeff: {self.sparsity_coeff}")
#         total_loss = L_reconstruction + self.sparsity_coeff * L_sparsity#.sum()
#         loss_dict = {
#             "L_reconstruction": L_reconstruction,
#             "L_sparsity": L_sparsity,
#             "total_loss": total_loss,
#         }
#         return total_loss, loss_dict

#     # --- Optimization Routine ---
#     def optimize_on_activations(
#         self,
#         activations: torch.Tensor,
#         batch_size: int = 1024,
#         epochs: int = 10,  # Changed from steps to epochs
#         log_freq: int = 100,
#         lr: float = 1e-3,
#         lr_scale: Callable[[int, int], float] = lambda step, total: 1.0,
#         resample_method: Optional[str] = None,
#         resample_freq: int = 2500,
#         resample_window: int = 500,
#         resample_scale: float = 0.5,
#         hidden_sample_size: int = 256,
#     ) -> list[dict[str, Any]]:
#         """
#         Trains the SAE on a tensor of activations for a specified number of epochs.
        
#         Args:
#             activations: Tensor of shape (N, seq_len, d_model)
#             batch_size: Mini-batch size (each example is one token of dimension d_model).
#             epochs: Number of complete passes through the dataset.
#             log_freq: Frequency (in steps) to log progress.
#             lr: Base learning rate.
#             lr_scale: Function to scale the learning rate.
#             resample_method: Optional resampling method (if implemented).
#             resample_freq: Frequency (in steps) to perform resampling.
#             resample_window: Number of steps to consider when resampling.
#             resample_scale: Scale factor used during resampling.
#             hidden_sample_size: Number of examples used for logging.
            
#         Returns:
#             data_log: A list of dictionaries containing logged training information.
#         """
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
#         data_log = []
#         frac_active_list = []

#         # Flatten the activations so that each token is treated as an independent example
#         flat_acts = activations.view(-1, activations.shape[-1])  # shape: (N * seq_len, d_model)
#         total_samples = flat_acts.size(0)
#         steps_per_epoch = (total_samples + batch_size - 1) // batch_size  # ceiling division
#         total_steps = steps_per_epoch * epochs
#         step = 0

#         print(f"Total steps: {total_steps} (steps per epoch: {steps_per_epoch})")

#         for epoch in range(epochs):
#             # Shuffle indices over the flattened examples at the start of each epoch
#             perm = torch.randperm(total_samples)
            
#             for start_idx in range(0, total_samples, batch_size):
#                 # Get batch indices and data
#                 batch_indices = perm[start_idx : start_idx + batch_size]
#                 batch = flat_acts[batch_indices]

#                 # Update the learning rate
#                 step_lr = lr * lr_scale(step, total_steps)
#                 for group in optimizer.param_groups:
#                     group["lr"] = step_lr

#                 # Optimization step
#                 optimizer.zero_grad()
#                 total_loss, loss_dict = self.compute_loss(batch)
#                 #total_loss.mean().backward()
#                 total_loss.mean(0).sum().backward()
#                 optimizer.step()

#                 # Optionally perform resampling of dead latents
#                 if (resample_method is not None) and ((step + 1) % resample_freq == 0):
#                     if len(frac_active_list) >= resample_window:
#                         frac_active_in_window = torch.stack(frac_active_list[-resample_window:], dim=0)
#                         if resample_method == "simple" and hasattr(self, "resample_simple"):
#                             print("Resampling with simple method...")
#                             self.resample_simple(frac_active_in_window, resample_scale)
#                         else:
#                             raise ValueError(f"Resampling method {resample_method} not implemented.")

#                 # Compute the fraction of active latent units
#                 acts, _ = self.encode(batch)
#                 frac_active = (acts.abs() > 1e-8).float().mean(dim=0)
#                 frac_active_list.append(frac_active.detach().cpu())

#                 # Logging
#                 if step % log_freq == 0 or (step + 1 == total_steps):
#                     # with torch.no_grad():
#                     #     h_sample = flat_acts[:hidden_sample_size]
#                     #     acts_sample, _ = self.encode(h_sample)
#                     #     recon_sample = self.decode(acts_sample)
#                         # _, loss_dict_sample = self.compute_loss(h_sample)
                    
#                     log_dict = {
#                         "step": step,
#                         "epoch": epoch,
#                         "lr": step_lr,
#                         "loss": total_loss.mean().item(),
#                         "frac_active_mean": frac_active.mean().item(),
#                         "loss_reconstruction": loss_dict["L_reconstruction"].mean().item(),
#                         "loss_sparsity": loss_dict["L_sparsity"].mean().item(),
#                         # "hidden_sample": h_sample.detach().cpu(),
#                         # "hidden_reconstructed_sample": recon_sample.detach().cpu(),
#                         **{name: param.detach().cpu() for name, param in self.named_parameters()},
#                     }
#                     data_log.append(log_dict)
#                     tqdm.write(
#                         f"Epoch {epoch}/{epochs}, Step {step}: lr={step_lr:.6f}, "
#                         f"recon_loss={loss_dict['L_reconstruction'].mean().item():.6f}, "
#                         f"sparsity_loss={loss_dict['L_sparsity'].mean().item():.6f}, "
#                         f"frac_active={frac_active.mean().item():.6f}"
#                     )
#                 step += 1

#         return data_log
    
#     # --- Resampling --- #
#     @t.no_grad()
#     def resample_simple(
#         self,
#         frac_active_in_window: Float[Tensor, "window d_sae"],
#         resample_scale: float,
#     ) -> None:
#         dead_latents_mask = (frac_active_in_window < 1e-8).all(dim=0)  # [d_sae]
#         n_dead = int(dead_latents_mask.int().sum().item())

#         replacement_values = t.randn((n_dead, self.d_model), device=self.W_enc.device)
#         replacement_values_normed = replacement_values / (
#             replacement_values.norm(dim=-1, keepdim=True) + self.weight_normalize_eps
#         )

#         # New names for weights & biases to resample
#         self.W_enc.data.transpose(-1, -2)[dead_latents_mask] = resample_scale * replacement_values_normed
#         self.W_dec.data[dead_latents_mask] = replacement_values_normed
#         self.b_enc.data[dead_latents_mask] = 0.0
#         self.log_theta.data[dead_latents_mask] = t.log(t.tensor(THETA_INIT))


# class JumpReLUSAE(SAE):
#     W_enc: Float[Tensor, "inst d_in d_sae"]
#     _W_dec: Float[Tensor, "inst d_sae d_in"] | None
#     b_enc: Float[Tensor, "inst d_sae"]
#     b_dec: Float[Tensor, "inst d_in"]
#     log_theta: Float[Tensor, "inst d_sae"]
#     def __init__(self, cfg: SAEConfig):
#         super(SAE, self).__init__()

#         self.cfg = cfg

#         self._W_dec = (
#             None
#             if self.cfg.tied_weights
#             else nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_sae, cfg.d_in))))
#         )
#         self.b_dec = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_in))

#         self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_in, cfg.d_sae))))
#         self.b_enc = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_sae))
#         self.log_theta = nn.Parameter(t.full((cfg.n_inst, cfg.d_sae), t.log(t.tensor(THETA_INIT))))

#         self.to(cfg.device)

#     @property
#     def theta(self) -> Float[Tensor, "inst d_sae"]:
#         return self.log_theta.exp()

#     def forward(
#         self, h: Float[Tensor, "batch inst d_in"]
#     ) -> tuple[
#         dict[str, Float[Tensor, "batch inst"]],
#         Float[Tensor, ""],
#         Float[Tensor, "batch inst d_sae"],
#         Float[Tensor, "batch inst d_in"],
#     ]:
#         """
#         Same as previous forward function, but allows for gated case as well (in which case we have different
#         functional form, as well as a new term "L_aux" in the loss dict).
#         """
#         h_cent = h - self.b_dec

#         acts_pre = (
#             einops.einsum(h_cent, self.W_enc, "batch inst d_in, inst d_in d_sae -> batch inst d_sae") + self.b_enc
#         )

#         acts_relu = F.relu(acts_pre)
#         acts_post = JumpReLU.apply(acts_relu, self.theta, self.cfg.ste_epsilon)

#         h_reconstructed = (
#             einops.einsum(acts_post, self.W_dec, "batch inst d_sae, inst d_sae d_in -> batch inst d_in") + self.b_dec
#         )

#         loss_dict = {
#             "L_reconstruction": (h_reconstructed - h).pow(2).mean(-1),
#             "L_sparsity": Heaviside.apply(acts_relu, self.theta, self.cfg.ste_epsilon).sum(-1),
#         }

#         loss = loss_dict["L_reconstruction"] + self.cfg.sparsity_coeff * loss_dict["L_sparsity"]

#         return loss_dict, loss, acts_post, h_reconstructed

#     @t.no_grad()
#     def resample_simple(
#         self,
#         frac_active_in_window: Float[Tensor, "window inst d_sae"],
#         resample_scale: float,
#     ) -> None:
#         dead_latents_mask = (frac_active_in_window < 1e-8).all(dim=0)  # [instances d_sae]
#         n_dead = int(dead_latents_mask.int().sum().item())

#         replacement_values = t.randn((n_dead, self.cfg.d_in), device=self.W_enc.device)
#         replacement_values_normed = replacement_values / (
#             replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
#         )

#         # New names for weights & biases to resample
#         self.W_enc.data.transpose(-1, -2)[dead_latents_mask] = resample_scale * replacement_values_normed
#         self.W_dec.data[dead_latents_mask] = replacement_values_normed
#         self.b_enc.data[dead_latents_mask] = 0.0
#         self.log_theta.data[dead_latents_mask] = t.log(t.tensor(THETA_INIT))

#     @t.no_grad()
#     def resample_advanced(
#         self, frac_active_in_window: Float[Tensor, "window inst d_sae"], resample_scale: float, batch_size: int
#     ) -> None:
#         h = self.generate_batch(batch_size)
#         l2_loss = self.forward(h)[0]["L_reconstruction"]

#         for instance in range(self.cfg.n_inst):
#             is_dead = (frac_active_in_window[:, instance] < 1e-8).all(dim=0)
#             dead_latents = t.nonzero(is_dead).squeeze(-1)
#             n_dead = dead_latents.numel()
#             if n_dead == 0:
#                 continue

#             l2_loss_instance = l2_loss[:, instance]  # [batch_size]
#             if l2_loss_instance.max() < 1e-6:
#                 continue

#             distn = Categorical(probs=l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum())
#             replacement_indices = distn.sample((n_dead,))  # type: ignore

#             replacement_values = (h - self.b_dec)[replacement_indices, instance]  # [n_dead d_in]
#             replacement_values_normalized = replacement_values / (
#                 replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
#             )

#             W_enc_norm_alive_mean = (
#                 self.W_enc[instance, :, ~is_dead].norm(dim=0).mean().item() if (~is_dead).any() else 1.0
#             )

#             # New names for weights & biases to resample
#             self.b_enc.data[instance, dead_latents] = 0.0
#             self.log_theta.data[instance, dead_latents] = t.log(t.tensor(THETA_INIT))
#             self.W_dec.data[instance, dead_latents, :] = replacement_values_normalized
#             self.W_enc.data[instance, :, dead_latents] = (
#                 replacement_values_normalized.T * W_enc_norm_alive_mean * resample_scale
#             )


#!/usr/bin/env python
"""
Combined implementation of a JumpReLU autoencoder and trainer,
designed to optimize on activations of shape (batch x seq_len x d_model).

This file defines:
  - The abstract Dictionary class.
  - JumpReluAutoEncoder: an autoencoder with jump-ReLUs.
  - Custom autograd Functions (RectangleFunction, JumpReLUFunction, StepFunction)
    that implement the jump and thresholding operations with straight-through gradients.
  - SAETrainer and JumpReluTrainer, which include learning rate and sparsity scheduling.
  - Utility functions for constraining/normalizing decoder weights.
  
To test the code, a main block creates random activations and runs a few training steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from abc import ABC, abstractmethod
import einops
from collections import namedtuple
from typing import Optional, Callable
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

# --- Custom Autograd Functions ---

class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Returns 1 when -0.5 < x < 0.5, else 0.
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Zero out the gradient outside (-0.5, 0.5)
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        # Save inputs and a tensor version of bandwidth for the backward pass.
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth, device=x.device, dtype=x.dtype))
        # Compute jump ReLU: multiply x by a binary mask based on the threshold.
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        # Gradient with respect to x: pass through where x > threshold.
        x_grad = (x > threshold).float() * grad_output
        # For threshold, use a rectangle kernel (STE).
        threshold_grad = (-(threshold / bandwidth)
                          * RectangleFunction.apply((x - threshold) / bandwidth)
                          * grad_output)
        return x_grad, threshold_grad, None  # no gradient for bandwidth

class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth, device=x.device, dtype=x.dtype))
        # Binary step: 1 if x > threshold, else 0.
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        # No gradient flows through x.
        x_grad = torch.zeros_like(x)
        # STE for the threshold using the rectangle kernel.
        threshold_grad = (-(1.0 / bandwidth)
                          * RectangleFunction.apply((x - threshold) / bandwidth)
                          * grad_output)
        return x_grad, threshold_grad, None

# --- Dictionary and Autoencoder Classes ---

class Dictionary(ABC, nn.Module):
    """
    An abstract dictionary consisting of a collection of vectors,
    an encoder, and a decoder.
    """
    dict_size: int      # number of features in the dictionary
    activation_dim: int # dimension of the activation vectors

    @abstractmethod
    def encode(self, x, output_pre_jump: bool = False):
        """
        Encode input x into the latent space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a latent representation f into input space.
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary.
        """
        pass

class JumpReluAutoEncoder(Dictionary):
    """
    An autoencoder with jump-ReLUs.
    The forward pass accepts activations of shape (batch, seq_len, activation_dim)
    and returns reconstructions of the same shape.
    """
    def __init__(self, activation_dim, dict_size, device="cpu"):
        super().__init__()
        self.activation_dim = activation_dim  # e.g. d_model
        self.dict_size = dict_size            # e.g. d_sae
        self.device = device

        # Encoder: maps from activation_dim to dict_size.
        self.W_enc = nn.Parameter(torch.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(torch.zeros(dict_size, device=device))

        # Decoder: maps from dict_size back to activation_dim.
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(dict_size, activation_dim, device=device))
        )
        self.b_dec = nn.Parameter(torch.zeros(activation_dim, device=device))

        # A threshold for gating activations.
        self.threshold = nn.Parameter(torch.ones(dict_size, device=device) * 0.001)

        # Optionally subtract the decoder bias from the input.
        self.apply_b_dec_to_input = False

        # Normalize decoder weights and initialize encoder weights accordingly.
        with torch.no_grad():
            self.W_dec.data = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)
            self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x, output_pre_jump: bool = False):
        """
        Encode input x into the latent representation.
          x: Tensor of shape (batch, seq_len, activation_dim)
        Returns:
          f: Tensor of shape (batch, seq_len, dict_size)
          (Optionally also returns pre_jump: the pre-activation values.)
        """
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc  # shape: (B, S, dict_size)
        # Apply a thresholded ReLU (here using a simple elementwise multiplication;
        # alternatively one could use JumpReLUFunction.apply).
        f = F.relu(pre_jump * (pre_jump > self.threshold))
        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        """
        Decode latent representation back to input space.
          f: Tensor of shape (batch, seq_len, dict_size)
        Returns:
          Reconstructed x of shape (batch, seq_len, activation_dim)
        """
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features: bool = False):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        self.b_enc.data *= scale
        self.threshold.data *= scale

    @classmethod
    def from_pretrained(cls, path: Optional[str] = None, load_from_sae_lens: bool = False,
                        dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None, **kwargs) -> "JumpReluAutoEncoder":
        """
        Load a pretrained autoencoder from a file.
        """
        if not load_from_sae_lens:
            state_dict = torch.load(path, map_location=device)
            activation_dim, dict_size = state_dict["W_enc"].shape[0], state_dict["W_enc"].shape[1]
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size, device=device)
            autoencoder.load_state_dict(state_dict)
            autoencoder = autoencoder.to(dtype=dtype, device=device)
        else:
            raise NotImplementedError("Loading from sae_lens is not implemented in this file.")
        return autoencoder

# --- SAE Trainer and Supporting Utilities ---

class SAETrainer:
    """
    A generic base class for SAE training algorithms.
    """
    def __init__(self, seed=None):
        self.seed = seed
        self.logging_parameters = []

    def update(self, step, activations):
        """Perform one training update (to be implemented in subclasses)."""
        pass

    def get_logging_parameters(self):
        stats = {}
        for param in self.logging_parameters:
            if hasattr(self, param):
                stats[param] = getattr(self, param)
            else:
                print(f"Warning: {param} not found in {self}")
        return stats

    @property
    def config(self):
        return {"wandb_name": "trainer"}

class ConstrainedAdam(torch.optim.Adam):
    """
    A variant of Adam that projects updates on some parameters to keep their columns at unit norm.
    """
    def __init__(self, params, constrained_params, lr: float, betas: tuple[float, float] = (0.9, 0.999)):
        super().__init__(params, lr=lr, betas=betas)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / (p.norm(dim=0, keepdim=True) + 1e-8)
                if p.grad is not None:
                    p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                p.data /= (p.data.norm(dim=0, keepdim=True) + 1e-8)

@torch.no_grad()
def set_decoder_norm_to_unit_norm(W_dec: torch.Tensor, activation_dim: int, d_sae: int) -> torch.Tensor:
    """
    Normalize each decoder weight vector (row) to have unit norm.
    """
    eps = torch.finfo(W_dec.dtype).eps
    norm = torch.norm(W_dec, dim=1, keepdim=True)
    W_dec.div_(norm + eps)
    return W_dec

@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(W_dec: torch.Tensor,
                                                    W_dec_grad: torch.Tensor,
                                                    activation_dim: int,
                                                    d_sae: int) -> torch.Tensor:
    """
    Remove from W_dec_grad the components that are parallel to the columns of W_dec.
    """
    normed_W_dec = W_dec / (torch.norm(W_dec, dim=1, keepdim=True) + 1e-6)
    parallel_component = (W_dec_grad * normed_W_dec).sum(dim=1, keepdim=True)
    W_dec_grad = W_dec_grad - parallel_component * normed_W_dec
    return W_dec_grad

def get_lr_schedule(total_steps: int,
                    warmup_steps: int,
                    decay_start: Optional[int] = None,
                    resample_steps: Optional[int] = None,
                    sparsity_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """
    Returns a learning rate schedule function with a linear warmup and optional decay.
    """
    if decay_start is not None:
        assert resample_steps is None, "decay_start and resample_steps are mutually exclusive."
        assert 0 <= decay_start < total_steps, "decay_start must be in [0, total_steps)."
        assert decay_start > warmup_steps, "decay_start must be greater than warmup_steps."
        if sparsity_warmup_steps is not None:
            assert decay_start > sparsity_warmup_steps, "decay_start must be greater than sparsity_warmup_steps."

    assert 0 <= warmup_steps < total_steps, "warmup_steps must be in [0, total_steps)."

    if resample_steps is None:
        def lr_schedule(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            if decay_start is not None and step >= decay_start:
                return (total_steps - step) / (total_steps - decay_start)
            return 1.0
    else:
        assert 0 < resample_steps < total_steps, "resample_steps must be > 0 and < total_steps."
        def lr_schedule(step: int) -> float:
            return min((step % resample_steps) / warmup_steps, 1.0)
    return lr_schedule

def get_sparsity_warmup_fn(total_steps: int, sparsity_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """
    Returns a function that scales the sparsity penalty from 0 to 1 over sparsity_warmup_steps.
    """
    if sparsity_warmup_steps is not None:
        assert 0 <= sparsity_warmup_steps < total_steps, "sparsity_warmup_steps must be in [0, total_steps)."
    def scale_fn(step: int) -> float:
        if not sparsity_warmup_steps:
            return 1.0
        else:
            return min(step / sparsity_warmup_steps, 1.0)
    return scale_fn

class JumpReluTrainer(nn.Module, SAETrainer):
    """
    Trainer for the JumpReluAutoEncoder.
    This trainer optimizes on inputs of shape (batch, seq_len, activation_dim).
    """
    def __init__(self,
                 steps: int,
                 activation_dim: int,
                 dict_size: int,
                 lr: float = 7e-5,
                 bandwidth: float = 0.001,
                 sparsity_penalty: float = 1.0,
                 warmup_steps: int = 1000,
                 sparsity_warmup_steps: Optional[int] = 2000,
                 decay_start: Optional[int] = None,
                 target_l0: float = 20.0,
                 device: str = "cpu",
                 wandb_name: str = "JumpRelu",
                 seed: Optional[int] = None):
        super().__init__()
        self.steps = steps
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.lr = lr
        self.bandwidth = bandwidth
        self.sparsity_penalty = sparsity_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.decay_start = decay_start
        self.target_l0 = target_l0
        self.device = device
        self.wandb_name = wandb_name
        self.seed = seed

        # Initialize the autoencoder.
        self.ae = JumpReluAutoEncoder(activation_dim, dict_size, device=device).to(device)

        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8)
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None,
                                  sparsity_warmup_steps=sparsity_warmup_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

        # For tracking “dead” dictionary features.
        self.dead_feature_threshold = 10_000_000
        self.num_tokens_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
        self.dead_features = -1
        self.logging_parameters = ["dead_features"]

    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """
        Compute the reconstruction and sparsity losses.
          x: Tensor of shape (batch, seq_len, activation_dim)
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        # Ensure correct type.
        x = x.to(self.ae.W_enc.dtype)

        # Compute pre-activation outputs.
        pre_jump = x @ self.ae.W_enc + self.ae.b_enc  # shape: (B, S, dict_size)
        # Apply jump ReLU using the custom autograd function.
        f = JumpReLUFunction.apply(pre_jump, self.ae.threshold, self.bandwidth)  # (B, S, dict_size)

        # Update firing statistics over both batch and sequence dimensions.
        active_indices = f.sum(dim=(0, 1)) > 0  # shape: (dict_size,)
        num_tokens = x.size(0) * x.size(1)
        self.num_tokens_since_fired += num_tokens
        self.num_tokens_since_fired[active_indices] = 0
        self.dead_features = (self.num_tokens_since_fired > self.dead_feature_threshold).sum().item()

        # Reconstruct the input.
        recon = self.ae.decode(f)  # shape: (B, S, activation_dim)
        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()  # Mean squared error.

        # Compute l0 penalty using the step function.
        l0 = StepFunction.apply(f, self.ae.threshold, self.bandwidth).sum(dim=-1).mean()
        sparsity_loss = (((l0 / self.target_l0) - 1).pow(2))

        total_loss = recon_loss + self.sparsity_penalty * sparsity_loss * sparsity_scale
        return total_loss, recon_loss, sparsity_loss

    def update(self, step: int, x: torch.Tensor) -> float:
        """
        Perform one optimization step.
          x: Tensor of shape (batch, seq_len, activation_dim)
        """
        x = x.to(self.device)
        loss_val, recon_loss, sparsity_loss = self.loss(x, step)
        loss_val.backward()

        # Adjust gradients for the decoder weights.
        if self.ae.W_dec.grad is not None:
            self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.W_dec.data, self.ae.W_dec.grad.data, self.activation_dim, self.dict_size
            )
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # Renormalize the decoder weights.
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(self.ae.W_dec.data, self.activation_dim, self.dict_size)
        return loss_val.item(), recon_loss.item(), sparsity_loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "JumpReluTrainer",
            "dict_class": "JumpReluAutoEncoder",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.activation_dim,
            "dict_size": self.dict_size,
            "device": self.device,
            "bandwidth": self.bandwidth,
            "sparsity_penalty": self.sparsity_penalty,
            "sparsity_warmup_steps": self.sparsity_warmup_steps,
            "target_l0": self.target_l0,
            "wandb_name": self.wandb_name,
        }

# --- Example Usage ---

if __name__ == "__main__":
    # Automatically select between CUDA and CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the activations.
    # (Assumes a file 'activations.pt' containing a tensor of shape [10000, 128, 2304])
    try:
        # Load your data from Hugging Face
        repo_id = "charlieoneill/gemma-medicine-sae"  # Replace with your repo

        # Download the activation tensor and dataset
        api = HfApi()
        activation_file = hf_hub_download(repo_id=repo_id, filename="10000_128.pt")

        # Load the tensors
        activations = torch.load(activation_file)
    except Exception as e:
        print(f"Error loading activations: {e}")
    
    
    # Example settings: optimize on random activations of shape (batch, seq_len, d_model)
    batch_size = 8
    total_steps = 100_000

    d_model = activations.shape[-1]  # e.g. 2304
    d_sae = 16384  # Example latent dimension; adjust as needed.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = JumpReluTrainer(steps=total_steps, activation_dim=d_model, dict_size=d_sae, device=device)

    print("Starting training...")
    
    # Shuffle activations along batch dimension
    shuffled_indices = torch.randperm(activations.shape[0])
    activations = activations[shuffled_indices]
    activations = activations.to(device)

    # Training loop
    for step in range(total_steps):
        # Get batch indices with wraparound
        start_idx = (step * batch_size) % (activations.shape[0] - batch_size)
        batch = activations[start_idx:start_idx + batch_size]

        loss, recon_loss, sparsity_loss = trainer.update(step=step, x=batch)
        
        if step % 1 == 0:
            print(f"Step {step:6d} | Rec Loss: {recon_loss:.4f} | Sparsity Loss: {sparsity_loss:.4f} | Total Loss: {loss:.4f}")

    print("Training completed.")
