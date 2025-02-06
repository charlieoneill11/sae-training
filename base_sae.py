import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import einops
import numpy as np
import torch as t
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

### our imports ###
from utils import linear_lr, constant_lr, cosine_decay_lr


@dataclass
class SAEConfig:
    n_inst: int
    d_in: int
    d_sae: int
    sparsity_coeff: float = 0.2
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    architecture: Literal["standard", "gated", "jumprelu"] = "standard"
    ste_epsilon: float = 0.01

class SAE(nn.Module):
    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]

    def __init__(self, cfg: SAEConfig, device: str) -> None:
      super(SAE, self).__init__()

      self.cfg = cfg

      self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_in, cfg.d_sae))))
      self._W_dec = (
          None
          if self.cfg.tied_weights
          else nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_sae, cfg.d_in))))
      )
      self.b_enc = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_sae))
      self.b_dec = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_in))

      self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "inst d_sae d_in"]:
        """
        Returns decoder weights, normalized over the autoencoder input dimension.
        """
        return self.W_dec / (self.W_dec.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps)

    def forward(
        self, h: Float[Tensor, "batch inst d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, "batch inst"],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Forward pass on the autoencoder.

        Args:
            h: hidden layer activations of model

        Returns:
            loss_dict:       dict of different loss terms, each dict value having shape (batch_size, n_inst)
            loss:            total loss (i.e. sum over terms of loss dict), same shape as loss_dict values
            acts_post:       autoencoder latent activations, after applying ReLU
            h_reconstructed: reconstructed autoencoder input
        """
        pre_acts = einops.einsum(
            self.W_enc, h-self.b_dec, "inst d_in d_sae, batch inst d_in -> batch inst d_sae"
        ) + self.b_enc
        z = F.relu(pre_acts)
        h_reconstructed = einops.einsum(
          self.W_dec_normalized, z, "inst d_sae d_in, batch inst d_sae -> batch inst d_in"
        ) + self.b_dec

        # Calculate losses
        L_reconstruction = ((h - h_reconstructed) ** 2).mean(-1)
        L_sparsity = z.abs().sum(-1)
        loss = L_reconstruction + self.cfg.sparsity_coeff * L_sparsity
        loss_dict = {"L_reconstruction": L_reconstruction, "L_sparsity": L_sparsity}

        return loss_dict, loss, z, h_reconstructed
        

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        resample_method: Literal["simple", "advanced", None] = None,
        resample_freq: int = 2500,
        resample_window: int = 500,
        resample_scale: float = 0.5,
        hidden_sample_size: int = 256,
    ) -> list[dict[str, Any]]:
        """
        Optimizes the autoencoder using the given hyperparameters.

        Args:
            model:              we reconstruct features from model's hidden activations
            batch_size:         size of batches we pass through model & train autoencoder on
            steps:              number of optimization steps
            log_freq:           number of optimization steps between logging
            lr:                 learning rate
            lr_scale:           learning rate scaling function
            resample_method:    method for resampling dead latents
            resample_freq:      number of optimization steps between resampling dead latents
            resample_window:    number of steps needed for us to classify a neuron as dead
            resample_scale:     scale factor for resampled neurons
            hidden_sample_size: size of hidden value sample we add to the logs (for eventual visualization)

        Returns:
            data_log:           dictionary containing data we'll use for visualization
        """
        assert resample_window <= resample_freq

        raise NotImplementedError()

        # optimizer = t.optim.Adam(list(self.parameters()), lr=lr)  # betas=(0.0, 0.999)
        # frac_active_list = []
        # progress_bar = tqdm(range(steps))

        # # Create lists of dicts to store data we'll eventually be plotting
        # data_log = []

        # for step in progress_bar:
        #     # Resample dead latents
        #     if (resample_method is not None) and ((step + 1) % resample_freq == 0):
        #         frac_active_in_window = t.stack(frac_active_list[-resample_window:], dim=0)
        #         if resample_method == "simple":
        #             self.resample_simple(frac_active_in_window, resample_scale)
        #         elif resample_method == "advanced":
        #             self.resample_advanced(frac_active_in_window, resample_scale, batch_size)

        #     # Update learning rate
        #     step_lr = lr * lr_scale(step, steps)
        #     for group in optimizer.param_groups:
        #         group["lr"] = step_lr

        #     # Get a batch of hidden activations from the model
        #     with t.inference_mode():
        #         h = self.generate_batch(batch_size)

        #     # Optimize
        #     loss_dict, loss, acts, _ = self.forward(h)
        #     loss.mean(0).sum().backward()
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     # Normalize decoder weights by modifying them directly (if not using tied weights)
        #     if not self.cfg.tied_weights:
        #         self.W_dec.data = self.W_dec_normalized.data

        #     # Calculate the mean sparsities over batch dim for each feature
        #     frac_active = (acts.abs() > 1e-8).float().mean(0)
        #     frac_active_list.append(frac_active)

        #     # Display progress bar, and log a bunch of values for creating plots / animations
        #     if step % log_freq == 0 or (step + 1 == steps):
        #         progress_bar.set_postfix(
        #             lr=step_lr,
        #             loss=loss.mean(0).sum().item(),
        #             frac_active=frac_active.mean().item(),
        #             **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
        #         )
        #         with t.inference_mode():
        #             loss_dict, loss, acts, h_r = self.forward(h := self.generate_batch(hidden_sample_size))
        #         data_log.append(
        #             {
        #                 "steps": step,
        #                 "frac_active": (acts.abs() > 1e-8).float().mean(0).detach().cpu(),
        #                 "loss": loss.detach().cpu(),
        #                 "h": h.detach().cpu(),
        #                 "h_r": h_r.detach().cpu(),
        #                 **{name: param.detach().cpu() for name, param in self.named_parameters()},
        #                 **{name: loss_term.detach().cpu() for name, loss_term in loss_dict.items()},
        #             }
        #         )

        # return data_log

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        """
        Resamples dead latents, by modifying the model's weights and biases inplace.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (d_in,), and normalize these vectors
            - Set new values of W_dec and W_enc to be these normalized vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        # Get a tensor of dead latents
        dead_latents_mask = (frac_active_in_window < 1e-8).all(dim=0)  # [instances d_sae]
        n_dead = int(dead_latents_mask.int().sum().item())

        # Get our random replacement values of shape [n_dead d_in], and scale them
        replacement_values = t.randn((n_dead, self.cfg.d_in), device=self.W_enc.device)
        replacement_values_normed = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )

        # Change the corresponding values in W_enc, W_dec, and b_enc
        self.W_enc.data.transpose(-1, -2)[dead_latents_mask] = resample_scale * replacement_values_normed
        self.W_dec.data[dead_latents_mask] = replacement_values_normed
        self.b_enc.data[dead_latents_mask] = 0.0

    @t.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        batch_size: int,
    ) -> None:
        """
        Resamples latents that have been dead for 'dead_feature_window' steps, according to `frac_active`.

        Resampling method is:
            - Compute the L2 reconstruction loss produced from the hidden state vectors `h`
            - Randomly choose values of `h` with probability proportional to their reconstruction loss
            - Set new values of W_dec and W_enc to be these (centered and normalized) vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron

        Returns colors and titles (useful for creating the animation: resampled neurons appear in red).
        """
        h = self.generate_batch(batch_size)
        l2_loss = self.forward(h)[0]["L_reconstruction"]

        for instance in range(self.cfg.n_inst):
            # Find the dead latents in this instance. If all latents are alive, continue
            is_dead = (frac_active_in_window[:, instance] < 1e-8).all(dim=0)
            dead_latents = t.nonzero(is_dead).squeeze(-1)
            n_dead = dead_latents.numel()
            if n_dead == 0:
                continue  # If we have no dead features, then we don't need to resample

            # Compute L2 loss for each element in the batch
            l2_loss_instance = l2_loss[:, instance]  # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue  # If we have zero reconstruction loss, we don't need to resample

            # Draw `d_sae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
            distn = Categorical(probs=l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum())
            replacement_indices = distn.sample((n_dead,))  # type: ignore

            # Index into the batch of hidden activations to get our replacement values
            replacement_values = (h - self.b_dec)[replacement_indices, instance]  # [n_dead d_in]
            replacement_values_normalized = replacement_values / (
                replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
            )

            # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
            W_enc_norm_alive_mean = self.W_enc[instance, :, ~is_dead].norm(dim=0).mean().item() if (~is_dead).any() else 1.0

            # Lastly, set the new weights & biases (W_dec is normalized, W_enc needs specific scaling, b_enc is zero)
            self.W_dec.data[instance, dead_latents, :] = replacement_values_normalized
            self.W_enc.data[instance, :, dead_latents] = (
                replacement_values_normalized.T * W_enc_norm_alive_mean * resample_scale
            )
            self.b_enc.data[instance, dead_latents] = 0.0



