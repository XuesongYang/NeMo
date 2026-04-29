# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F


class MoERouter(torch.nn.Module):
    """
    Router for Mixture of Experts that selects which experts to use for each token.
    Supports multiple routing strategies including top-k and Sinkhorn routing.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter_noise: float = 0.0,
        routing_strategy: str = "top_k",  # "top_k" or "sinkhorn"
    ):
        """
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts
            top_k (int): Number of experts to select per token
            router_jitter_noise (float): Add noise to router logits for exploration during training
            routing_strategy (str): Strategy for routing ("top_k" or "sinkhorn")
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router_jitter_noise = router_jitter_noise
        self.routing_strategy = routing_strategy
        assert routing_strategy in ["top_k", "sinkhorn"], "Invalid routing strategy"

        # Router is a simple linear layer that outputs logits for each expert
        self.router = torch.nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing decisions for each token.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
            x_mask (torch.Tensor): Mask tensor of shape (B, T) where 1=valid token, 0=padding

        Returns:
            Tuple containing:
                - expert_weights (torch.Tensor): Normalized weights for selected experts of shape (B, T, top_k).
                    For padded positions, weights are set to 0.
                - expert_indices (torch.Tensor): Indices of selected experts of shape (B, T, top_k).
                    For padded positions, indices are set to -1 (sentinel value).
                - router_logits (torch.Tensor): Raw router logits of shape (B, T, num_experts).
                    Padded positions are masked to zero.
                - router_probs (torch.Tensor): Router probabilities after softmax of shape (B, T, num_experts).
                    Padded positions are masked to zero.
        """
        # Compute router logits: (B, T, num_experts)
        router_logits = self.router(x * x_mask.unsqueeze(-1))

        # Add jitter noise during training for exploration
        if self.training and self.router_jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.router_jitter_noise
            router_logits = router_logits + noise

        # Mask router logits to ensure padded positions remain zero
        router_logits = router_logits * x_mask.unsqueeze(-1)

        # Compute routing probabilities for each token.
        # Padded positions with logits of [0, 0, ..., 0] will produce a uniform softmax ([1/n, ..., 1/n]);
        # this is acceptable, since we require valid probabilities for top-k selection and normalization.
        # Sinkhorn routing is used only during training for balancing, while at inference simple softmax is used for efficiency.
        if self.routing_strategy == "sinkhorn" and self.training:
            router_probs = self._sinkhorn_routing(router_logits, x_mask)
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        # expert_weights: (B, T, top_k), expert_indices: (B, T, top_k)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize weights to sum to 1
        # For padded positions: uniform probs -> 1/top_k
        # For valid positions: normal routing weights
        # Avoid division by zero when all weights are zero.
        weight_sums = expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights / torch.where(weight_sums > 0, weight_sums, torch.ones_like(weight_sums))

        # Mask expert_weights and expert_indices for padded positions
        # Set expert_indices to -1 for padding so they don't match any valid expert (0 to num_experts-1)
        # This prevents padded tokens from being processed through experts
        expert_weights = expert_weights * x_mask.unsqueeze(-1)  # Zero out weights for padding
        expert_indices = expert_indices.masked_fill(~x_mask.unsqueeze(-1).bool(), -1)  # Set to -1 for padding

        # Mask router_probs for return
        router_probs = router_probs * x_mask.unsqueeze(-1)

        return expert_weights, expert_indices, router_logits, router_probs

    @staticmethod
    def _sinkhorn_routing(
        logits: torch.Tensor, x_mask: torch.Tensor, num_iters: int = 100, e_tol: float = 1e-3
    ) -> torch.Tensor:
        """
        Padding-aware Sinkhorn routing with convergence checking.

        This implementation:
        1. Extracts only valid (non-padded) tokens before Sinkhorn
        2. Applies Sinkhorn-Knopp algorithm with convergence criterion
        3. Re-pads the output to original shape

        The algorithm computes a doubly stochastic matrix by iteratively
        normalizing rows and columns using diagonal scaling factors.

        Args:
            logits (torch.Tensor): Router logits of shape (B, T, num_experts)
            x_mask (torch.Tensor): Mask of shape (B, T) where 1=valid token, 0=padding
            num_iters (int): Maximum number of Sinkhorn iterations (default: 100)
            e_tol (float): Convergence tolerance for scaling factors (default: 1e-3)

        Returns:
            torch.Tensor: Routing probabilities of shape (B, T, num_experts)
                Valid tokens: doubly stochastic probabilities
                Padded tokens: zeros
        """
        B, T, E = logits.shape

        # Extract valid tokens (exclude padding)
        valid_mask = x_mask.view(-1).bool()  # (B*T,)
        valid_logits = logits.view(B * T, E)[valid_mask]  # (N, E) where N = number of valid tokens

        if valid_logits.numel() == 0:
            # All tokens are padding, return zeros
            return torch.zeros_like(logits)

        # Numerical stability: subtract max per row to prevent exp overflow.
        # This is similar to the log-sum-exp trick used in softmax.
        # For Sinkhorn, subtracting a constant per row doesn't change the final
        # doubly-stochastic result since both row and column normalizations will
        # absorb the scaling factor.
        valid_logits_stable = valid_logits - valid_logits.max(dim=-1, keepdim=True).values

        # Apply exp to get cost matrix (must be positive for Sinkhorn)
        K = torch.exp(valid_logits_stable)  # (N, E)

        # Initialize diagonal scaling factors
        d1 = torch.ones(K.size(0), device=K.device, dtype=K.dtype)  # Row scaling (N,)
        d2 = torch.ones(K.size(1), device=K.device, dtype=K.dtype)  # Column scaling (E,)

        # Sinkhorn-Knopp iterations with convergence check
        for _ in range(num_iters):
            d1_old = d1.clone()

            # Update row scaling: d1[i] = 1 / sum_j(K[i,j] * d2[j])
            d1 = 1.0 / (torch.matmul(K, d2) + 1e-9)

            # Update column scaling: d2[j] = 1 / sum_i(K[i,j] * d1[i])
            d2 = 1.0 / (torch.matmul(K.t(), d1) + 1e-9)

            # Clamp scaling factors to prevent numerical instability from accumulating
            d1 = torch.clamp(d1, min=1e-9, max=1e9)
            d2 = torch.clamp(d2, min=1e-9, max=1e9)

            # Check convergence based on change in scaling factors
            err = torch.mean(torch.abs(d1_old - d1))
            if err < e_tol:
                break

        # Compute scaled matrix using broadcasting (avoids materializing NxN diagonal matrices):
        # P = diag(d1) @ K @ diag(d2)  =>  P[i, j] = d1[i] * K[i, j] * d2[j]
        P = (d1[:, None] * K) * d2[None, :]  # (N, E)

        # Final row normalization to ensure each row sums to 1 (valid probability distribution)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-9)  # (N, E)

        # Re-pad to original shape
        result = torch.zeros(B * T, E, device=logits.device, dtype=logits.dtype)
        result[valid_mask] = P
        result = result.view(B, T, E)

        return result


class PositionwiseConvFFMoE(torch.nn.Module):
    """
    Mixture of Experts version of ``PositionwiseConvFF``.

    Expert weights are stored as pre-stacked ``nn.Parameter`` tensors of shape
    ``(num_experts, out_features, in_features)`` rather than a ``ModuleList``
    of individual ``Conv1d`` layers.  This follows the same design used by
    `Megatron-LM GroupedMLP <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/experts.py>`_
    and `TorchTitan GroupedExperts <https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/common/moe/moe.py>`_,
    and eliminates the per-forward ``torch.stack`` copy that otherwise doubles
    expert-weight memory in the autograd graph (~3.5 GB for a 12-layer decoder
    with 16 experts in FP32).

    Backward compatibility: checkpoints saved with the old ``ModuleList``
    layout (keys like ``experts.0.proj.conv.weight``) are automatically
    converted in ``_load_from_state_dict``.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        p_dropout: float,
        num_experts: int = 8,
        top_k_experts: int = 2,
        kernel_size: int = 1,
        bias: bool = False,
        is_causal: bool = True,
        non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
        router_jitter_noise: float = 0.0,
        routing_strategy: str = "top_k",
    ):
        """
        Args:
            d_model (int): Input and output dimension
            d_ffn (int): Hidden dimension of FFN (usually 4 * d_model, or d_model for param-matched MoE)
            p_dropout (float): Dropout probability
            num_experts (int): Number of expert networks
            top_k_experts (int): Number of experts to use per token
            kernel_size (int): Convolution kernel size. Must be 1 for MoE so that each expert
                is a standard pointwise linear FFN (Conv1d with kernel_size=1 is equivalent to
                nn.Linear applied independently at each position).
            bias (bool): Whether to use bias in convolution layers
            is_causal (bool): Whether to use causal convolution (accepted for API compat, unused when kernel_size=1)
            non_linearity (Callable): Activation function
            router_jitter_noise (float): Noise for router exploration
            routing_strategy (str): Routing strategy ("top_k" or "sinkhorn")
        """
        if kernel_size != 1:
            raise ValueError(
                f"`PositionwiseConvFFMoE` requires kernel_size=1, got {kernel_size}. "
                f"Each MoE expert must be a pointwise linear FFN (Conv1d with kernel_size=1 == nn.Linear). "
                f"kernel_size > 1 is not supported because (1) standard MoE experts are linear layers, "
                f"and (2) MoE dispatch gathers tokens from arbitrary (batch, time) positions, so "
                f"Conv1d with kernel_size > 1 would mix non-adjacent tokens."
            )
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.non_linearity = non_linearity
        self._has_bias = bias

        # Router for expert selection
        self.router = MoERouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k_experts,
            router_jitter_noise=router_jitter_noise,
            routing_strategy=routing_strategy,
        )

        # Pre-stacked expert weight parameters.
        # Storing weights as (num_experts, ...) tensors lets torch.bmm use the
        # leaf parameter directly; autograd stores a reference (not a copy),
        # saving ~288 MB per decoder layer compared to torch.stack on the fly.
        self.w1 = torch.nn.Parameter(torch.empty(num_experts, d_ffn, d_model))
        self.w2 = torch.nn.Parameter(torch.empty(num_experts, d_model, d_ffn))
        if bias:
            self.b1 = torch.nn.Parameter(torch.empty(num_experts, d_ffn))
            self.b2 = torch.nn.Parameter(torch.empty(num_experts, d_model))
        else:
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)

        # Initialize each expert slice in the same order as the legacy
        # per-expert Conv1d modules so that a given random seed produces
        # identical weights (proj weight, [proj bias], o_net weight, [o_net bias]).
        for i in range(num_experts):
            torch.nn.init.kaiming_uniform_(self.w1.data[i], a=math.sqrt(5))
            if bias:
                fan_in_proj = d_model  # kernel_size=1 → fan_in = in_channels
                bound = 1.0 / math.sqrt(fan_in_proj)
                torch.nn.init.uniform_(self.b1.data[i], -bound, bound)
            torch.nn.init.kaiming_uniform_(self.w2.data[i], a=math.sqrt(5))
            if bias:
                fan_in_onet = d_ffn
                bound = 1.0 / math.sqrt(fan_in_onet)
                torch.nn.init.uniform_(self.b2.data[i], -bound, bound)

        self.dropout = torch.nn.Dropout(p_dropout)

    # ------------------------------------------------------------------
    # Backward-compatible checkpoint loading
    # ------------------------------------------------------------------
    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Convert legacy per-expert Conv1d keys to pre-stacked format.

        Old checkpoints store expert weights as individual Conv1d parameters::

            {prefix}experts.{i}.proj.conv.weight   (d_ffn, d_model, 1)
            {prefix}experts.{i}.proj.conv.bias      (d_ffn,)
            {prefix}experts.{i}.o_net.conv.weight   (d_model, d_ffn, 1)
            {prefix}experts.{i}.o_net.conv.bias     (d_model,)

        These are fused into ``w1``, ``w2``, ``b1``, ``b2`` before the
        parent ``_load_from_state_dict`` runs.
        """
        legacy_key = f'{prefix}experts.0.proj.conv.weight'
        if legacy_key in state_dict:
            w1_list: List[torch.Tensor] = []
            w2_list: List[torch.Tensor] = []
            b1_list: List[torch.Tensor] = []
            b2_list: List[torch.Tensor] = []
            for i in range(self.num_experts):
                w1_list.append(state_dict.pop(f'{prefix}experts.{i}.proj.conv.weight').squeeze(-1))
                w2_list.append(state_dict.pop(f'{prefix}experts.{i}.o_net.conv.weight').squeeze(-1))
                b1_key = f'{prefix}experts.{i}.proj.conv.bias'
                b2_key = f'{prefix}experts.{i}.o_net.conv.bias'
                if b1_key in state_dict:
                    b1_list.append(state_dict.pop(b1_key))
                if b2_key in state_dict:
                    b2_list.append(state_dict.pop(b2_key))
            state_dict[f'{prefix}w1'] = torch.stack(w1_list)
            state_dict[f'{prefix}w2'] = torch.stack(w2_list)
            if b1_list:
                state_dict[f'{prefix}b1'] = torch.stack(b1_list)
            if b2_list:
                state_dict[f'{prefix}b2'] = torch.stack(b2_list)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Mixture of Experts feedforward layer.

        For each valid token (x_mask=1), routes to top_k experts based on router predictions.
        Padded tokens (x_mask=0) are assigned expert_indices=-1 and are not processed through any expert,
        ensuring they remain zero in the output.

        All experts are processed in parallel via ``torch.bmm`` using pre-stacked
        weight parameters.  Two paths are used depending on context:

        * **Training** (``torch.is_grad_enabled()``): all ``num_experts`` are
          included in the bmm so that the leaf parameter is passed directly —
          autograd stores a reference, not a copy (~288 MB saved per layer).
        * **Inference** (``torch.no_grad()``): only active experts (those that
          received at least one token) are included, avoiding wasted FLOPs on
          idle experts.  The temporary indexed copy is freed immediately since
          no autograd graph retains it.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
            x_mask (torch.Tensor): Mask tensor of shape (B, T) where 1=valid token, 0=padding

        Returns:
            Tuple containing:
                - output (torch.Tensor): Output tensor of shape (B, T, C).
                    Valid tokens contain weighted combination of top_k expert outputs.
                    Padded positions remain zero (never processed by experts).
                - router_logits (torch.Tensor): Raw router logits for auxiliary loss of shape (B, T, num_experts).
                    Padded positions are masked to zero.
                - router_probs (torch.Tensor): Router probabilities for auxiliary loss of shape (B, T, num_experts).
                    Padded positions are masked to zero.
                - expert_indices (torch.Tensor): Selected expert indices of shape (B, T, top_k).
                    For padded positions, indices are -1. For computing expert selection statistics.
        """
        # Get expert routing from router
        expert_weights, expert_indices, router_logits, router_probs = self.router(x, x_mask)
        # expert_weights: (B, T, top_k)
        # expert_indices: (B, T, top_k)
        # router_logits: (B, T, num_experts)
        # router_probs: (B, T, num_experts)

        # Batched dispatch: flatten all (token, expert-slot) assignments once,
        # sort by expert to get contiguous slices, pad to equal length,
        # then process all experts in parallel via batched matmul.
        B, T, C = x.shape
        top_k = expert_indices.shape[-1]

        # Flatten token dimension: (B*T, C)
        x_flat = x.view(-1, C)
        num_tokens = x_flat.size(0)  # B * T

        # Flatten routing assignments to 1-D vectors:
        #   assign_expert: (num_tokens * top_k,)  — which expert each assignment targets
        #   assign_weight: (num_tokens * top_k, 1) — routing weight for each assignment
        assign_expert = expert_indices.reshape(-1)
        assign_weight = expert_weights.reshape(-1, 1)

        # Map each assignment back to its source token index (0 .. num_tokens-1).
        # token_indices: (num_tokens * top_k,)
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)

        # Filter out padding assignments (expert_indices == -1 for padded positions).
        # This is required because torch.bincount does not accept negative values,
        # and padded tokens should not be processed by any expert.
        valid_assign_mask = assign_expert != -1
        assign_expert = assign_expert[valid_assign_mask]
        assign_weight = assign_weight[valid_assign_mask]
        token_indices = token_indices[valid_assign_mask]

        # Initialize flat output buffer.
        output_flat = torch.zeros_like(x_flat)

        if assign_expert.numel() > 0:
            # Sort assignments by expert so each expert's tokens form a contiguous slice.
            sorted_expert, sort_idx = torch.sort(assign_expert)
            sorted_token_indices = token_indices[sort_idx]
            sorted_weights = assign_weight[sort_idx]

            # Compute per-expert assignment counts and slice boundaries.
            counts = torch.bincount(sorted_expert, minlength=self.num_experts)

            # During inference (no grad), index into pre-stacked weights to
            # skip idle experts — the temporary copy is freed immediately
            # since no autograd graph retains it.  During training, use all
            # experts so that torch.bmm receives the leaf parameter directly
            # and autograd stores only a reference, not a copy.
            if torch.is_grad_enabled():
                active_experts = None
                batch_experts = self.num_experts
                batch_counts = counts
                expert_idx = sorted_expert  # global ids, 0 .. num_experts-1
                w1, w2 = self.w1, self.w2
                b1, b2 = self.b1, self.b2
            else:
                active_experts = torch.where(counts > 0)[0]
                batch_experts = active_experts.size(0)
                batch_counts = counts[active_experts]
                # Remap global expert ids → dense local ids (0 .. batch_experts-1)
                expert_to_local = torch.empty(self.num_experts, dtype=torch.long, device=x.device)
                expert_to_local[active_experts] = torch.arange(batch_experts, device=x.device)
                expert_idx = expert_to_local[sorted_expert]
                w1 = self.w1[active_experts]
                w2 = self.w2[active_experts]
                b1 = self.b1[active_experts] if self.b1 is not None else None
                b2 = self.b2[active_experts] if self.b2 is not None else None

            max_count = batch_counts.max().item()

            # Compute 0-based position of each assignment within its expert group.
            starts = torch.zeros(batch_experts, dtype=torch.long, device=x.device)
            starts[1:] = batch_counts[:-1].cumsum(0)
            within_idx = torch.arange(len(sorted_token_indices), device=x.device) - starts[expert_idx]

            # Scatter tokens into a padded (batch_experts, max_count, C) batch.
            padded_input = torch.zeros(batch_experts, max_count, C, device=x.device, dtype=x.dtype)
            padded_input[expert_idx, within_idx] = x_flat[sorted_token_indices]

            # (batch_experts, d_ffn, d_model) @ (batch_experts, d_model, max_count)
            hidden = torch.bmm(w1, padded_input.transpose(1, 2))

            if b1 is not None:
                hidden = hidden + b1.unsqueeze(-1)

            hidden = self.non_linearity(hidden)

            # (batch_experts, d_model, d_ffn) @ (batch_experts, d_ffn, max_count)
            output_batched = torch.bmm(w2, hidden)

            if b2 is not None:
                output_batched = output_batched + b2.unsqueeze(-1)

            output_batched = self.dropout(output_batched)

            # (batch_experts, d_model, max_count) -> (batch_experts, max_count, d_model)
            output_batched = output_batched.transpose(1, 2)

            # Gather valid outputs, apply routing weights, and accumulate back
            # to the source token positions.
            expert_outputs = output_batched[expert_idx, within_idx]  # (total_assignments, d_model)
            weighted_outputs = expert_outputs * sorted_weights
            output_flat.index_add_(0, sorted_token_indices, weighted_outputs)

        # Reshape back to (B, T, C)
        output = output_flat.view(B, T, C)

        return output, router_logits, router_probs, expert_indices
