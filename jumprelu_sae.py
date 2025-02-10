import torch as t
from torch import Tensor
from typing import Any
from torch.distributions.categorical import Categorical
from jaxtyping import Float
import torch.nn as nn
import torch.nn.functional as F
import einops

### our imports ###
from base_sae import SAE, SAEConfig


def rectangle(x: Tensor, width: float = 1.0) -> Tensor:
    """
    Returns the rectangle function value, i.e. K(x) = 1[|x| < width/2], as a float.
    """
    return (x.abs() < width / 2).float()


class Heaviside(t.autograd.Function):
    """
    Implementation of the Heaviside step function, using straight through estimators for the derivative.

        forward:
            H(z,θ,ε) = 1[z > θ]

        backward:
            dH/dz := None
            dH/dθ := -1/ε * K(z/ε)

            where K is the rectangle kernel function with width 1, centered at 0: K(u) = 1[|u| < 1/2]
    """

    @staticmethod
    def forward(ctx: Any, z: t.Tensor, theta: t.Tensor, eps: float) -> t.Tensor:
        # Save any necessary information for backward pass
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        # Compute the output
        return (z > theta).float()

    @staticmethod
    def backward(ctx: Any, grad_output: t.Tensor) -> t.Tensor:
        # Retrieve saved tensors & values
        (z, theta) = ctx.saved_tensors
        eps = ctx.eps
        # Compute gradient of the loss with respect to z (no STE) and theta (using STE)
        grad_z = 0.0 * grad_output
        grad_theta = -(1.0 / eps) * rectangle((z - theta) / eps) * grad_output
        grad_theta_agg = grad_theta.sum(dim=0)  # note, sum over batch dim isn't strictly necessary

        # # ! DEBUGGING
        # # nz = rectangle((z - theta) / eps) > 0
        # print(f"HEAVISIDE\nNumber of nonzero grads? {(grad_theta.abs() > 1e-6).float().mean():.3f}")
        # print(f"Average positive (of non-zero grads): {grad_theta[grad_theta.abs() > 1e-6].mean():.3f}")

        return grad_z, grad_theta_agg, None
    

class JumpReLU(t.autograd.Function):
    """
    Implementation of the JumpReLU function, using straight through estimators for the derivative.

        forward:
            J(z,θ,ε) = z * 1[z > θ]

        backward:
            dJ/dθ := -θ/ε * K(z/ε)
            dJ/dz := 1[z > θ]

            where K is the rectangle kernel function with width 1, centered at 0: K(u) = 1[|u| < 1/2]
    """

    @staticmethod
    def forward(ctx: Any, z: t.Tensor, theta: t.Tensor, eps: float) -> t.Tensor:
        # Save relevant stuff for backwards pass
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        # Compute the output
        return z * (z > theta).float()


    @staticmethod
    def backward(ctx: Any, grad_output: t.Tensor) -> t.Tensor:
        # Retrieve saved tensors & values
        (z, theta) = ctx.saved_tensors
        eps = ctx.eps
        # Compute gradient of the loss with respect to z (no STE) and theta (using STE)
        grad_z = (z > theta).float() * grad_output
        grad_theta = -(theta / eps) * rectangle((z - theta) / eps) * grad_output
        grad_theta_agg = grad_theta.sum(dim=0)  # note, sum over batch dim isn't strictly necessary
        return grad_z, grad_theta_agg, None
    
THETA_INIT = 0.1


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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Optional
from tqdm import tqdm

class JumpReLUSAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_sae: int,
        sparsity_coeff: float = 1.0,
        threshold_init: float = 0.1,
        ste_epsilon: float = 1e-2  # for compatibility if needed later
    ):
        """
        Args:
            d_model: Dimension of input (and reconstruction).
            d_sae: Dimension of the latent space.
            sparsity_coeff: Weighting for the sparsity penalty.
            threshold_init: Initial threshold value.
            ste_epsilon: STE epsilon (not used in this simple version).
        """
        super().__init__()

        # Encoder: shape [d_model, d_sae]
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        nn.init.kaiming_uniform_(self.W_enc, a=nn.init.calculate_gain('relu'))

        # Decoder: shape [d_sae, d_model]
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        nn.init.kaiming_uniform_(self.W_dec, a=nn.init.calculate_gain('relu'))

        # Threshold for gating activations (one per latent unit)
        self.threshold = nn.Parameter(torch.full((d_sae,), threshold_init))

        # Biases for encoder and decoder
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self.d_model = d_model
        self.d_sae = d_sae
        self.sparsity_coeff = sparsity_coeff
        self.ste_epsilon = ste_epsilon

    def encode(self, input_acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a linear transform, ReLU, and gating based on the threshold.
        
        Args:
            input_acts: Tensor of shape (..., d_model)
        Returns:
            acts: Gated latent activations (shape: (..., d_sae))
            pre_acts: Pre-activation values.
        """
        pre_acts = input_acts @ self.W_enc + self.b_enc
        relu_acts = F.relu(pre_acts)
        # The threshold is broadcast along the batch dimensions.
        mask = (pre_acts > self.threshold).float()
        acts = relu_acts * mask
        return acts, pre_acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the input from latent activations.
        
        Args:
            acts: Tensor of shape (..., d_sae)
        Returns:
            Reconstructed input of shape (..., d_model)
        """
        return acts @ self.W_dec + self.b_dec

    def forward(self, input_acts: torch.Tensor) -> torch.Tensor:
        """
        Full pass: encode then decode.
        
        Args:
            input_acts: Tensor of shape (..., d_model)
        Returns:
            Reconstruction of the input.
        """
        acts, _ = self.encode(input_acts)
        recon = self.decode(acts)
        return recon

    # --- Loss Functions ---
    def compute_reconstruction_loss(self, input_acts: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """
        Computes mean-squared error per example.
        """
        return ((recon - input_acts) ** 2).mean(dim=-1)

    def compute_sparsity_loss(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """
        Computes the sparsity loss by measuring the fraction of activations exceeding the threshold.
        For a 3D tensor (batch, seq_len, d_sae) we average over batch and sequence; for 2D input,
        we average only over the batch.
        """
        mask = (pre_acts > self.threshold).float()
        if pre_acts.dim() == 3:
            return mask.mean(dim=(0, 1))  # one value per latent dimension
        else:
            return mask.mean(dim=0)

    def compute_loss(self, input_acts: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the total loss: reconstruction + sparsity penalty.
        
        Args:
            input_acts: Input tensor of shape (batch, d_model) or (batch, seq_len, d_model)
        Returns:
            total_loss: Per-example loss (if input is batched) plus the sparsity penalty.
            loss_dict: Dictionary with individual loss components.
        """
        acts, pre_acts = self.encode(input_acts)
        recon = self.decode(acts)
        L_reconstruction = self.compute_reconstruction_loss(input_acts, recon)
        L_sparsity = self.compute_sparsity_loss(pre_acts)
        # Sum over latent dimensions (if L_sparsity is a vector) and add to reconstruction loss.
        total_loss = L_reconstruction + self.sparsity_coeff * L_sparsity.sum()
        loss_dict = {
            "L_reconstruction": L_reconstruction,
            "L_sparsity": L_sparsity,
            "total_loss": total_loss,
        }
        return total_loss, loss_dict

    # --- Optimization Routine ---
    def optimize_on_activations(
        self,
        activations: torch.Tensor,
        batch_size: int = 1024,
        steps: int = 10000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = lambda step, total: 1.0,
        resample_method: Optional[str] = None,  # "simple", "advanced", or None
        resample_freq: int = 2500,
        resample_window: int = 500,
        resample_scale: float = 0.5,
        hidden_sample_size: int = 256,
    ) -> list[dict[str, Any]]:
        """
        Trains the SAE on a tensor of activations by first flattening the sequence dimension into
        the batch dimension. Before every epoch the flattened examples are shuffled.
        
        Args:
            activations: Tensor of shape (N, seq_len, d_model)
            batch_size: Mini-batch size (each example is one token of dimension d_model).
            steps: Total training steps.
            log_freq: Frequency (in steps) to log progress.
            lr: Base learning rate.
            lr_scale: Function to scale the learning rate.
            resample_method: Optional resampling method (if implemented).
            resample_freq: Frequency (in steps) to perform resampling.
            resample_window: Number of steps to consider when resampling.
            resample_scale: Scale factor used during resampling.
            hidden_sample_size: Number of examples used for logging.
            
        Returns:
            data_log: A list of dictionaries containing logged training information.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        data_log = []
        frac_active_list = []

        # Flatten the activations so that each token is treated as an independent example.
        flat_acts = activations.view(-1, activations.shape[-1])  # shape: (N * seq_len, d_model)
        total_samples = flat_acts.size(0)
        step = 0

        while step < steps:
            # Shuffle indices over the flattened examples before each epoch.
            perm = torch.randperm(total_samples)
            for start_idx in range(0, total_samples, batch_size):
                if step >= steps:
                    break
                batch_indices = perm[start_idx : start_idx + batch_size]
                batch = flat_acts[batch_indices]  # shape: (batch_size, d_model)

                # Update the learning rate.
                step_lr = lr * lr_scale(step, steps)
                for group in optimizer.param_groups:
                    group["lr"] = step_lr

                optimizer.zero_grad()
                total_loss, loss_dict = self.compute_loss(batch)
                total_loss.mean().backward()
                optimizer.step()

                # Optionally perform resampling of dead latents if implemented.
                if (resample_method is not None) and ((step + 1) % resample_freq == 0):
                    if len(frac_active_list) >= resample_window:
                        frac_active_in_window = torch.stack(frac_active_list[-resample_window:], dim=0)
                        if resample_method == "simple" and hasattr(self, "resample_simple"):
                            self.resample_simple(frac_active_in_window, resample_scale)
                        elif resample_method == "advanced" and hasattr(self, "resample_advanced"):
                            self.resample_advanced(frac_active_in_window, resample_scale, batch_size)

                # Compute the fraction of active latent units.
                acts, _ = self.encode(batch)
                frac_active = (acts.abs() > 1e-8).float().mean(dim=0)  # shape: (d_sae,)
                frac_active_list.append(frac_active.detach().cpu())

                # Logging.
                if step % log_freq == 0 or (step + 1) == steps:
                    with torch.no_grad():
                        # For logging, take the first hidden_sample_size examples.
                        h_sample = flat_acts[:hidden_sample_size]
                        acts_sample, _ = self.encode(h_sample)
                        recon_sample = self.decode(acts_sample)
                        _, loss_dict_sample = self.compute_loss(h_sample)
                    log_dict = {
                        "step": step,
                        "lr": step_lr,
                        "loss": total_loss.mean().item(),
                        "frac_active_mean": frac_active.mean().item(),
                        "loss_reconstruction": loss_dict["L_reconstruction"].mean().item(),
                        "loss_sparsity": loss_dict["L_sparsity"].sum().item(),
                        "hidden_sample": h_sample.detach().cpu(),
                        "hidden_reconstructed_sample": recon_sample.detach().cpu(),
                        **{name: param.detach().cpu() for name, param in self.named_parameters()},
                    }
                    data_log.append(log_dict)
                    tqdm.write(
                        f"Step {step}: lr={step_lr:.6f}, loss={total_loss.mean().item():.6f}, "
                        f"frac_active={frac_active.mean().item():.6f}"
                    )
                step += 1

        return data_log