# Copyright 2024 Genmo and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch, sys
from transformers import T5EncoderModel, T5TokenizerFast

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import Mochi1LoraLoaderMixin
from diffusers.models import AutoencoderKLMochi, MochiTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.mochi.pipeline_output import MochiPipelineOutput



if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import MochiPipeline
        >>> from diffusers.utils import export_to_video

        >>> pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()
        >>> pipe.enable_vae_tiling()
        >>> prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
        >>> frames = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).frames[0]
        >>> export_to_video(frames, "mochi.mp4")
        ```
"""



def optimized_scale(noise_pred_text,noise_pred_uncond, batch_size):
    
    positive_flat = noise_pred_text.view(batch_size, -1)  
    negative_flat = noise_pred_uncond.view(batch_size, -1)  

                    
                    
               
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    alpha = dot_product / squared_norm
    
    alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
    
    alpha = alpha.to(positive_flat.dtype)    
 
    return alpha





def svd_low_rank_reconstruction(tensor: torch.Tensor, k: int) -> torch.Tensor:
 
 
    
    # Extract 2D matrix M from tensor: M = tensor[0, :, :] (shape [H, W])
    M = tensor.squeeze(0)  # Shape [H, W]
    H, W = M.shape
    
    # Validate k
    max_rank = min(H, W)
    if k < 1 or k > max_rank:
        raise ValueError(f"k must be between 1 and {max_rank}, got {k}")
    
    # Move to CPU for SVD (more stable for large matrices, avoids GPU memory spikes)
    M_cpu = M.detach().cpu().to(torch.float32)
    
    # Compute SVD: M = U * Sigma * V^T
    U, S, Vh = torch.svd(M_cpu, some=True)
    # U: [H, H], S: [min(H, W)], Vh: [W, W] (Vh = V^T)
    
    # Truncate to top k singular values and vectors
    U_k = U[:, :k]          # [H, k]
    S_k = S[:k]             # [k]
    Vh_k = Vh[:k, :]        # [k, W]
    
    # Reconstruct low-rank matrix: M_k = U_k * diag(S_k) * Vh_k
    M_k = torch.matmul(U_k, torch.diag(S_k) @ Vh_k)  # [H, W]
    
    # Move back to original device (e.g., CUDA) and restore [1, H, W] shape
    M_k = M_k.to(tensor.device).to(tensor.dtype)
    T_k = M_k.unsqueeze(0)  # Shape [1, H, W]
    
    return T_k





def butterworth_low_pass_filter(latents: torch.Tensor, d_s: float, order: int = 2, fft_axis: tuple = (-2, -1)) -> torch.Tensor:
    """
    Generate a Butterworth low-pass filter for latents with shape [1, h, w].

    Args:
        latents (torch.Tensor): Input latents tensor, shape [1, h, w].
        d_s (float): Normalized cutoff frequency (0 to 1), relative to Nyquist frequency.
        order (int): Butterworth filter order (higher = sharper cutoff).
        fft_axis (tuple): Axes for FFT, default (-2, -1) for height, width.

    Returns:
        torch.Tensor: Low-pass filter mask, shape [1, h, w].
    """
    # Get latents shape and FFT dimensions
    shape = latents.shape
    h, w = shape[-2], shape[-1]  # Height and width from [1, h, w]
    device = latents.device
    
    # Create frequency grid for height and width
    freq_h = torch.fft.fftfreq(h, d=1.0, device=device)
    freq_w = torch.fft.fftfreq(w, d=1.0, device=device)
    freq_h = torch.fft.fftshift(freq_h)
    freq_w = torch.fft.fftshift(freq_w)
    
    # Meshgrid for 2D frequencies
    grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
    freq_dist = torch.sqrt(grid_h**2 + grid_w**2)  # Euclidean distance from origin
    
    # Normalize frequencies (Nyquist = 1.0)
    max_freq = freq_dist.max()
    normalized_freq = freq_dist / max_freq
    
    # Butterworth LPF: 1 / (1 + (f/f_c)^(2*order))
    cutoff = d_s  # Normalized cutoff frequency
    lpf = 1.0 / (1.0 + (normalized_freq / cutoff)**(2 * order))
    
    # Reshape LPF to [1, h, w]
    lpf = lpf.view(1, h, w).to(latents.dtype)
    
    return lpf

def frequency_filter_latents(
    latents: torch.Tensor,
    original_latents: torch.Tensor,
    d_s: float,
    order: int = 2,
    fft_axis: tuple = (-2, -1)
) -> torch.Tensor:
    """
    Apply low-pass and high-pass filtering to mix latents and original_latents in frequency domain.

    Args:
        latents (torch.Tensor): Current noisy latents, shape [1, h, w].
        original_latents (torch.Tensor): Original latents for high-frequency components, shape [1, h, w].
        d_s (float): Normalized cutoff frequency for Butterworth filter (0 to 1).
        order (int): Butterworth filter order.
        fft_axis (tuple): Axes for FFT, default (-2, -1) for height, width.

    Returns:
        torch.Tensor: Filtered latents, shape [1, h, w].
    """
    # Ensure original_latents is same dtype and device
    #original_latents = original_latents.to(dtype=torch.float32, device=latents.device)
    
    # Compute Butterworth LPF
    lpf = butterworth_low_pass_filter(latents, d_s=d_s, order=order, fft_axis=fft_axis)
    
    # Derive HPF
    hpf = 1.0 - lpf
    
    # FFT on latents and original_latents
    latents_freq = torch.fft.fftn(latents, dim=fft_axis)
    latents_freq = torch.fft.fftshift(latents_freq, dim=fft_axis)
    original_latents_freq = torch.fft.fftn(original_latents, dim=fft_axis)
    original_latents_freq = torch.fft.fftshift(original_latents_freq, dim=fft_axis)
    
    # Frequency mixing
    new_freq = latents_freq * lpf + original_latents_freq * hpf
    
    # IFFT to return to spatial domain
    new_freq = torch.fft.ifftshift(new_freq, dim=fft_axis)
    filtered_latents = torch.fft.ifftn(new_freq, dim=fft_axis).real
    
    return filtered_latents



def softpick(x, dim=-1, eps=1e-8):
    # softpick function: relu(exp(x)-1) / sum(abs(exp(x)-1))
    # numerically stable version
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return r_x_e_1 / (torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps)



def optimized_scale(noise_pred_text,noise_pred_uncond, batch_size):
    
    positive_flat = noise_pred_text.view(batch_size, -1)  
    negative_flat = noise_pred_uncond.view(batch_size, -1)  

                    
                    
               
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    alpha = dot_product / squared_norm
    
    alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
    
    alpha = alpha.to(positive_flat.dtype)    
 
    return alpha





def cpow(xyz, pw = 0.95):
   
    a1 = torch.sign(xyz)
    m1 = torch.max(torch.abs(xyz))
    
    ix =  torch.abs(xyz)/m1
   

    return  (ix**pw)*m1*a1 




def project(v0: torch.Tensor, v1: torch.Tensor,):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def c_normalized_guidance( pred_cond: torch.Tensor, pred_uncond: torch.Tensor, diff: torch.Tensor, guidance_scale: float, eta: float = 1.0, norm_thr: float = 0.0):
 
    if norm_thr > 0:
       ones = torch.ones_like(diff)
       diff_norm = diff.norm(p=2, dim=[-1, -2, ], keepdim=True)
       scale_factor = torch.minimum(ones, norm_thr / diff_norm)
       diff = diff * scale_factor
    
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    
    return pred_guided



def project4(v0: torch.Tensor, v1: torch.Tensor,):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)



def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    y = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize
    return kernel

def apply_gaussian_smoothing(tensor, H, W, kernel_size=5, sigma=1.0):
    """
    Apply 2D Gaussian smoothing to a tensor with shape [batch, channels, flattened_spatial].
    Args:
        tensor: Input tensor of shape [batch, channels, H*W].
        H, W: Height and width of the spatial grid.
        kernel_size: Size of the Gaussian kernel (e.g., 5 for 5x5).
        sigma: Standard deviation of the Gaussian kernel.
    Returns:
        Smoothed tensor of same shape.
    """
    batch, channels, spatial = tensor.shape
    assert spatial == H * W, f"Expected spatial dim {H*W}, got {spatial}"

    # Reshape to [batch, channels, H, W]
    tensor = tensor.view(batch, channels, H, W)

    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma).to(tensor.dtype).to(tensor.device)
    # Reshape kernel to [out_channels, in_channels // groups, kernel_size, kernel_size]
    # Here, out_channels = channels, in_channels // groups = 1
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(channels, 1, kernel_size, kernel_size).to(tensor.device)

    # Apply convolution to each channel
    smoothed = F.conv2d(
        tensor,
        kernel,
        padding=kernel_size // 2,
        groups=channels
    )

    # Reshape back
    smoothed = smoothed.view(batch, channels, spatial).to(tensor.dtype).to(tensor.device)
    return smoothed



def orthogonalize_noise(cond_noise: torch.Tensor, uncond_noise: torch.Tensor) -> torch.Tensor:
    """Orthogonalize unconditional noise w.r.t. conditional noise for shape [1, H, W]."""
    # Ensure input shapes are [1, H, W]
 

    # Flatten tensors to [1, H*W] for dot product
    cond_flat = cond_noise.reshape(cond_noise.size(0), -1)  # [1, H*W]
    uncond_flat = uncond_noise.reshape(uncond_noise.size(0), -1)

    # Compute dot products
    dot_vu = torch.sum(uncond_flat * cond_flat, dim=-1, keepdim=True)  # v · u
    dot_uu = torch.sum(cond_flat * cond_flat, dim=-1, keepdim=True)  # u · u

    # Avoid division by zero
    dot_uu = torch.clamp(dot_uu, min=1e-10)

    # Compute projection: proj_u(v) = (v · u / u · u) * u
    proj = (dot_vu / dot_uu) * cond_flat

    # Orthogonal component: v_perp = v - proj_u(v)
    uncond_perp_flat = uncond_flat - proj

    # Reshape back to [1, H, W]
    uncond_perp = uncond_perp_flat.reshape(uncond_noise.shape)

    # Normalize to match original magnitude (optional, for stability)
    orig_norm = torch.norm(uncond_noise, dim=(1, 2), keepdim=True)
    perp_norm = torch.norm(uncond_perp, dim=(1, 2), keepdim=True)
    perp_norm = torch.clamp(perp_norm, min=1e-10)
    uncond_perp = uncond_perp * (orig_norm / perp_norm)

    return uncond_perp.to(cond_noise.dtype).to(cond_noise.device)





def orthome(tenush, par=0.6):

    q0 = orthogonalize_noise (tenush[1].unsqueeze(0), tenush[0].unsqueeze(0))[0]
    w1 = par
    w2 = 1-w1
    tenush[0] =  w1*q0 + w2*tenush[0] 
    return tenush
    
    
def svdme(tenush, j3, par=0.6):

    q0 = svd_low_rank_reconstruction (tenush[0].unsqueeze(0), j3)[0]
    w1 = par
    w2 = 1-w1
    tenush[0] =  w1*q0 + w2*tenush[0] 
    return tenush    



def shuffle_tokens( x):
    """
    Randomly shuffle the order of input tokens.

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, num_tokens, channels) (b, n, c)

    Returns:
        torch.Tensor: Shuffled tensor with the same shape (b, n, c)
    """
    b, n, c = x.shape
    # Generate a random permutation of indices for the token dimension
    permutation = torch.randperm(n, device=x.device)
    # Shuffle tokens across the token dimension using the same permutation for all batches
    x_shuffled = x[:, permutation]
    return x_shuffled


def compute_orthogonal_parallel_components(tensor, reference_vector):
    """
    Decompose a tensor into orthogonal and parallel components relative to a reference vector.
    
    Args:
        tensor: Input tensor of shape [B, N, C]
        reference_vector: Reference vector of shape [C] or [B, C] or [B, 1, C]
        
    Returns:
        parallel_component: Tensor of shape [B, N, C]
        orthogonal_component: Tensor of shape [B, N, C]
    """
    # Ensure reference_vector has the right shape
    if reference_vector.dim() == 1:
        # Shape [C] -> [B, 1, C]
        reference_vector = reference_vector.view(1, 1, -1).expand(tensor.size(0), 1, -1)
    elif reference_vector.dim() == 2 and reference_vector.size(1) == tensor.size(2):
        # Shape [B, C] -> [B, 1, C]
        reference_vector = reference_vector.unsqueeze(1)
    
    # Normalize reference vector
    ref_norm = F.normalize(reference_vector, p=2, dim=-1)
    
    # Compute parallel component: proj_ref(tensor) = (tensor · ref_norm) * ref_norm
    dot_product = torch.sum(tensor * ref_norm, dim=-1, keepdim=True)  # [B, N, 1]
    parallel_component = dot_product * ref_norm  # [B, N, C]
    
    # Compute orthogonal component: tensor - parallel_component
    orthogonal_component = tensor - parallel_component
    
    return parallel_component, orthogonal_component


import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

def forward_with_stg(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with STG (Spatio-Temporal Guidance) that explicitly computes Q, K, V
    for the attention mechanism in the Mochi video model, following MochiAttnProcessor2_0.

    Args:
        hidden_states: Input tensor of shape (batch_size, seq_len, dim).
        encoder_hidden_states: Encoder hidden states for cross-attention, shape (batch_size, text_seq_length, dim).
        temb: Timestep embeddings.
        encoder_attention_mask: Attention mask for encoder hidden states.
        image_rotary_emb: Optional rotary embeddings (cos, sin) for positional encoding.

    Returns:
        Tuple of (hidden_states, encoder_hidden_states).
    """
    def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to the input tensor, as defined in MochiAttnProcessor2_0.

        Args:
            x: Input tensor of shape (batch_size, heads, seq_len, head_dim).
            freqs_cos: Cosine tensor for rotary embeddings.
            freqs_sin: Sine tensor for rotary embeddings.

        Returns:
            Tensor with rotary embeddings applied, same shape as input.
        """
        x_even = x[..., 0::2].float()
        x_odd = x[..., 1::2].float()

        cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
        sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

        return torch.stack([cos, sin], dim=-1).flatten(-2)

    num_prompt = hidden_states.size(0) // 3
    hidden_states_ptb = hidden_states[2*num_prompt:]
    encoder_hidden_states_ptb = encoder_hidden_states[2*num_prompt:]
    
    norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

    if not self.context_pre_only:
        norm_encoder_hidden_states, enc_gate_msa, enc_scale_mlp, enc_gate_mlp = self.norm1_context(
            encoder_hidden_states, temb
        )
    else:
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)

    # Explicit attention with Q, K, V computation
    sequence_length = norm_hidden_states.size(1)
    encoder_sequence_length = norm_encoder_hidden_states.size(1)
    batch_size = norm_hidden_states.size(0)

    # Compute Q, K, V for video (hidden_states)
    query = self.attn1.to_q(norm_hidden_states)  # (batch_size, seq_len, dim)
    key = self.attn1.to_k(norm_hidden_states)    # (batch_size, seq_len, dim)
    value = self.attn1.to_v(norm_hidden_states)  # (batch_size, seq_len, dim)

    query = query - torch.mean(query, dim=2, keepdim=True)[0] 
    td = query.shape[2]
    H = 64
    W = td//H
    
 
 
    query = query - torch.mean(query, dim=2, keepdim=True)[0] 

        
        
    op, oo = compute_orthogonal_parallel_components(query, key)
 
    oos = apply_gaussian_smoothing(oo, H, W, kernel_size=5, sigma=1.0)
    
 
            
    query2 =  oos
 
 
    vd = 0
    #query[vd] = query2[vd] 
 
        
    cm = 0
    
    gm = 0.67
    mg = 1- gm
    query[cm] = gm * query2[cm] + mg * query[cm]
    #query[cm] = gm * squery[cm] + mg * query[cm]

    


    query = query.unflatten(2, (self.attn1.heads, -1))  # (batch_size, seq_len, heads, head_dim)
    key = key.unflatten(2, (self.attn1.heads, -1))      # (batch_size, seq_len, heads, head_dim)
    value = value.unflatten(2, (self.attn1.heads, -1))  # (batch_size, seq_len, heads, head_dim)

    if self.attn1.norm_q is not None:
        query = self.attn1.norm_q(query)
    if self.attn1.norm_k is not None:
        key = self.attn1.norm_k(key)

    # Compute Q, K, V for text (encoder_hidden_states)
    encoder_query = self.attn1.add_q_proj(norm_encoder_hidden_states)  # (batch_size, text_seq_length, dim)
    encoder_key = self.attn1.add_k_proj(norm_encoder_hidden_states)    # (batch_size, text_seq_length, dim)
    encoder_value = self.attn1.add_v_proj(norm_encoder_hidden_states)  # (batch_size, text_seq_length, dim)

    encoder_query = encoder_query.unflatten(2, (self.attn1.heads, -1))  # (batch_size, text_seq_length, heads, head_dim)
    encoder_key = encoder_key.unflatten(2, (self.attn1.heads, -1))      # (batch_size, text_seq_length, heads, head_dim)
    encoder_value = encoder_value.unflatten(2, (self.attn1.heads, -1))  # (batch_size, text_seq_length, heads, head_dim)

    if self.attn1.norm_added_q is not None:
        encoder_query = self.attn1.norm_added_q(encoder_query)
    if self.attn1.norm_added_k is not None:
        encoder_key = self.attn1.norm_added_k(encoder_key)

    # Apply rotary embeddings
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, *image_rotary_emb)
        key = apply_rotary_emb(key, *image_rotary_emb)

    # Transpose for attention
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)  # (batch_size, heads, seq_len, head_dim)
    encoder_query, encoder_key, encoder_value = (
        encoder_query.transpose(1, 2),
        encoder_key.transpose(1, 2),
        encoder_value.transpose(1, 2)  # (batch_size, heads, text_seq_length, head_dim)
    )

    # Process attention per batch element with mask
    total_length = sequence_length + encoder_sequence_length
    attn_outputs = []
    for idx in range(batch_size):
        mask = encoder_attention_mask[idx][None, :] if encoder_attention_mask is not None else None
        if mask is not None:
            valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        else:
            valid_prompt_token_indices = torch.arange(encoder_sequence_length, device=encoder_key.device)

        valid_encoder_query = encoder_query[idx:idx+1, :, valid_prompt_token_indices, :]
        valid_encoder_key = encoder_key[idx:idx+1, :, valid_prompt_token_indices, :]
        valid_encoder_value = encoder_value[idx:idx+1, :, valid_prompt_token_indices, :]

        valid_query = torch.cat([query[idx:idx+1], valid_encoder_query], dim=2)  # (1, heads, seq_len + valid_text_len, head_dim)
        valid_key = torch.cat([key[idx:idx+1], valid_encoder_key], dim=2)
        valid_value = torch.cat([value[idx:idx+1], valid_encoder_value], dim=2)

        attn_output = F.scaled_dot_product_attention(
            valid_query, valid_key, valid_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        valid_sequence_length = attn_output.size(2)
        attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
        attn_outputs.append(attn_output)

    hidden_states_attn = torch.cat(attn_outputs, dim=0)  # (batch_size, heads, total_length, head_dim)
    hidden_states_attn = hidden_states_attn.transpose(1, 2).flatten(2, 3)  # (batch_size, total_length, dim)

    # Split back into video and text components
    attn_hidden_states, context_attn_hidden_states = hidden_states_attn.split_with_sizes(
        (sequence_length, encoder_sequence_length), dim=1
    )

    # Project and apply dropout
    attn_hidden_states = self.attn1.to_out[0](attn_hidden_states)  # Linear projection
    attn_hidden_states = self.attn1.to_out[1](attn_hidden_states)  # Dropout

    if hasattr(self.attn1, "to_add_out"):
        context_attn_hidden_states = self.attn1.to_add_out(context_attn_hidden_states)

    hidden_states = hidden_states + self.norm2(attn_hidden_states, torch.tanh(gate_msa).unsqueeze(1))
    norm_hidden_states = self.norm3(hidden_states, (1 + scale_mlp.unsqueeze(1).to(torch.float32)))
    ff_output = self.ff(norm_hidden_states)
    hidden_states = hidden_states + self.norm4(ff_output, torch.tanh(gate_mlp).unsqueeze(1))
    
    if not self.context_pre_only:
        encoder_hidden_states = encoder_hidden_states + self.norm2_context(
            context_attn_hidden_states, torch.tanh(enc_gate_msa).unsqueeze(1)
        )
        norm_encoder_hidden_states = self.norm3_context(
            encoder_hidden_states, (1 + enc_scale_mlp.unsqueeze(1).to(torch.float32))
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + self.norm4_context(
            context_ff_output, torch.tanh(enc_gate_mlp).unsqueeze(1)
        )

    #hidden_states[2*num_prompt:] = hidden_states_ptb
    #encoder_hidden_states[2*num_prompt:] = encoder_hidden_states_ptb

    return hidden_states, encoder_hidden_states

# from: https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise Value_orgError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom value")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise Value_orgError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise Value_orgError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class MochiOSEGPipeline(DiffusionPipeline, Mochi1LoraLoaderMixin):
    r"""
    The mochi pipeline for text-to-video generation.

    Reference: https://github.com/genmoai/models

    Args:
        transformer ([`MochiTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLMochi`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLMochi,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: MochiTransformer3DModel,
        force_zeros_for_empty_prompt: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        # TODO: determine these scaling factors from model parameters
        self.vae_spatial_scale_factor = 8
        self.vae_temporal_scale_factor = 6
        self.patch_size = 2

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 256
        )
        self.default_height = 480
        self.default_width = 848
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        # The original Mochi implementation zeros out empty negative prompts
        # but this can lead to overflow when placing the entire pipeline under the autocast context
        # adding this here so that we can enable zeroing prompts if necessary
        if self.config.force_zeros_for_empty_prompt and (prompt == "" or prompt[-1] == ""):
            text_input_ids = torch.zeros_like(text_input_ids, device=device)
            prompt_attention_mask = torch.zeros_like(prompt_attention_mask, dtype=torch.bool, device=device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # Adapted from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise Value_orgError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise Value_orgError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise Value_orgError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise Value_orgError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise Value_orgError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise Value_orgError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise Value_orgError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise Value_orgError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise Value_orgError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise Value_orgError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = height // self.vae_spatial_scale_factor
        width = width // self.vae_spatial_scale_factor
        num_frames = (num_frames - 1) // self.vae_temporal_scale_factor + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise Value_orgError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents.to(dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0
    
    @property
    def do_spatio_temporal_guidance(self):
        return self._stg_scale > 0.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 19,
        num_inference_steps: int = 64,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        stg_applied_layers_idx: Optional[List[int]] = [34],
        stg_scale: Optional[float] = 0.0,
        do_rescaling: Optional[bool] = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to `self.default_height`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to `self.default_width`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `19`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `4.5`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.mochi.MochiPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `256`):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.mochi.MochiPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.mochi.MochiPipelineOutput`] is returned, otherwise a `tuple`
                is returned where the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.default_height
        width = width or self.default_width

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._stg_scale = stg_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        
        if self.do_spatio_temporal_guidance:
            for i in stg_applied_layers_idx:
                self.transformer.transformer_blocks[i].forward = types.MethodType(forward_with_stg, self.transformer.transformer_blocks[i])

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask, prompt_attention_mask], dim=0)

        # 5. Prepare timestep
        # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
        threshold_noise = 0.025
        sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
        sigmas = np.array(sigmas)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Note: Mochi uses reversed timesteps. To ensure compatibility with methods like FasterCache, we need
                # to make sure we're using the correct non-reversed timestep value.
                self._current_timestep = 1000 - t
                if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
                    latent_model_input = torch.cat([latents] * 3)
                else:
                    latent_model_input = latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                # Mochi CFG + Sampling runs in FP32
                noise_pred = noise_pred.to(torch.float32)

                if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
                    noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
                    
                                        
                    vp = 0.98
                    al1 = vp*optimized_scale(noise_pred_text, noise_pred_uncond, batch_size)
                    
                    al2 = vp*optimized_scale(noise_pred_text, noise_pred_perturb, batch_size)
                    
                    noise_pred = noise_pred_text  + (self.guidance_scale-1.0) * (noise_pred_text - noise_pred_uncond * al1) + .62* (noise_pred_text - noise_pred_perturb * al2)
                    
                    
                    if (i <=0):
                       noise_pred = noise_pred_text*0.1    
                    
                    
                    #noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) + self._stg_scale * (noise_pred_text - noise_pred_perturb)
                        
                if do_rescaling:
                    rescaling_scale = 0.7
                    factor = noise_pred_text.std() / noise_pred.std()
                    factor = rescaling_scale * factor + (1 - rescaling_scale)
                    noise_pred = noise_pred * factor

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents.to(torch.float32), return_dict=False)[0]
                latents = latents.to(latents_dtype)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            video = latents
        else:
            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return MochiPipelineOutput(frames=video)
