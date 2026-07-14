"""Ranking/classification/regression losses.

Every loss is a ``torch.nn.Module`` with the uniform signature
``forward(scores, target, group=None)``:

* ``scores`` — model logits, shape ``[B]`` (or ``[B, K]`` for multiclass /
  CORAL-layer heads),
* ``target`` — per-row label; ordinal losses use the raw level, binary losses
  treat any ``target > 0`` as positive,
* ``group`` — per-row query id for pairwise/listwise losses. Batches are
  group-contiguous, so these losses always see whole groups.

Families: pointwise, pairwise, listwise, ordinal, and convex combinations.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange


class Grouped1D(NamedTuple):
    scores: torch.Tensor
    target: torch.Tensor
    row_group: torch.Tensor
    lengths: torch.Tensor
    order: torch.Tensor


class PaddedGroups(NamedTuple):
    scores: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    lengths: torch.Tensor


class PairwiseTensors(NamedTuple):
    scores: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    lengths: torch.Tensor
    diff_scores: torch.Tensor
    diff_target: torch.Tensor
    pos_pairs: torch.Tensor


class NdcgComponents(NamedTuple):
    """NDCG building blocks used by LambdaRank-family losses."""

    ranks: torch.Tensor  # [G, M] predicted ranks 1..M (zeros for padding)
    gains: torch.Tensor  # [G, M] ``2 ** target - 1``
    discounts: torch.Tensor  # [G, M] ``1/log2(rank+1)``, masked by valid & rank<=K
    ideal_ndcg: torch.Tensor  # [G] per-group ideal DCG normalizer


class Loss(ABC, torch.nn.Module):
    """Base loss strategy."""

    requires_group = False

    @abstractmethod
    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


def _zero_loss_if_empty(scores: torch.Tensor) -> torch.Tensor | None:
    if scores.size(0) == 0:
        return scores.sum() * 0
    return None


def _validate_batch_lengths(
    scores: torch.Tensor,
    target: torch.Tensor,
    group: torch.Tensor | None,
) -> None:
    if scores.size(0) != target.size(0):
        raise ValueError(
            "scores and target have inconsistent batch lengths: "
            f"{scores.size(0)} != {target.size(0)}",
        )
    if group is not None and group.size(0) != scores.size(0):
        raise ValueError(
            "scores and group have inconsistent batch lengths: "
            f"{scores.size(0)} != {group.size(0)}",
        )


def _grouped_1d(
    scores: torch.Tensor,
    target: torch.Tensor,
    group: torch.Tensor | None,
) -> Grouped1D:
    """Sort rows by group and return grouped 1D tensors.

    ``torch.segment_reduce`` works on contiguous segments, so all vectorized
    ranking losses start from this representation.
    """
    _validate_batch_lengths(scores, target, group)
    if group is None:
        group = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)

    # Densify groups to [0..G-1] and get per-group counts in a single pass.
    # On empty input, all three return tensors are empty and the rest of the
    # pipeline degenerates correctly without a special case.
    _, inverse, lengths = torch.unique(
        group,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    order = torch.argsort(inverse, stable=True)
    return Grouped1D(
        scores=scores[order],
        target=target[order].to(dtype=scores.dtype),
        row_group=inverse[order],
        lengths=lengths,
        order=order,
    )


def _pad_by_group(
    scores: torch.Tensor,
    target: torch.Tensor,
    group: torch.Tensor | None,
) -> PaddedGroups:
    """Pad group-contiguous vectors to ``[G, max_group_size]``.

    Padding values are zeros and must be masked by callers via ``valid``.
    """
    grouped = _grouped_1d(scores, target, group)
    n_groups = grouped.lengths.numel()
    max_len = int(grouped.lengths.max().item()) if n_groups else 0

    padded_s = scores.new_zeros((n_groups, max_len))
    padded_y = grouped.target.new_zeros((n_groups, max_len))
    if max_len:
        # offsets[i] = start row of row i's group within the flat grouped tensor
        offsets = torch.repeat_interleave(
            torch.cumsum(grouped.lengths, 0) - grouped.lengths,
            grouped.lengths,
        )
        pos = torch.arange(grouped.scores.numel(), device=scores.device) - offsets
        padded_s[grouped.row_group, pos] = grouped.scores
        padded_y[grouped.row_group, pos] = grouped.target

    cols = torch.arange(max_len, device=scores.device).unsqueeze(0)  # [1, M]
    valid = cols < grouped.lengths.unsqueeze(1)  # [G, M]
    return PaddedGroups(
        scores=padded_s,
        target=padded_y,
        valid=valid,
        lengths=grouped.lengths,
    )


def _pairwise_tensors(
    scores: torch.Tensor,
    target: torch.Tensor,
    group: torch.Tensor | None,
) -> PairwiseTensors:
    padded_out = _pad_by_group(scores, target, group)
    diff_s = rearrange(padded_out.scores, "g i -> g i 1") - rearrange(
        padded_out.scores,
        "g j -> g 1 j",
    )
    diff_y = rearrange(padded_out.target, "g i -> g i 1") - rearrange(
        padded_out.target,
        "g j -> g 1 j",
    )
    valid_pair = rearrange(padded_out.valid, "g i -> g i 1") & rearrange(
        padded_out.valid,
        "g j -> g 1 j",
    )
    pos_pairs = valid_pair & (diff_y > 0)
    return PairwiseTensors(
        scores=padded_out.scores,
        target=padded_out.target,
        valid=padded_out.valid,
        lengths=padded_out.lengths,
        diff_scores=diff_s,
        diff_target=diff_y,
        pos_pairs=pos_pairs,
    )


def _inv_log_discount(positions: torch.Tensor, k: int) -> torch.Tensor:
    """``1/log2(positions + 1)`` zeroed where ``positions > k``."""
    return (positions <= k).to(positions.dtype) / torch.log2(positions + 1.0)


def _ndcg_components(
    scores: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    truncation_level: int | None,
) -> NdcgComponents:
    """Vectorized NDCG building blocks used by LambdaRank-family losses."""
    n_groups, max_len = scores.shape
    device, dtype = scores.device, scores.dtype
    K = truncation_level if truncation_level is not None else max_len

    # Predicted ranks: 1..M scattered by descending score.
    scores_for_sort = scores.detach().masked_fill(~valid, -torch.inf)
    sorted_idx = torch.argsort(scores_for_sort, dim=1, descending=True)
    positions = torch.arange(1, max_len + 1, device=device, dtype=dtype)
    ranks = torch.zeros_like(scores)
    ranks.scatter_(1, sorted_idx, positions.expand(n_groups, -1))

    gains = 2.0 ** target.float() - 1.0
    discounts = _inv_log_discount(ranks, K) * valid.to(dtype)

    # Ideal NDCG: sort gains descending; invalid rows hold 0 gain and contribute 0.
    ideal_gains = torch.sort(
        gains.masked_fill(~valid, 0.0),
        dim=1,
        descending=True,
    ).values
    ideal_discounts = _inv_log_discount(positions.unsqueeze(0), K)  # [1, M]
    ideal_ndcg = (ideal_gains * ideal_discounts).sum(dim=1)

    return NdcgComponents(
        ranks=ranks,
        gains=gains,
        discounts=discounts,
        ideal_ndcg=ideal_ndcg,
    )


def _masked_sinkhorn(
    log_alpha: torch.Tensor,
    valid: torch.Tensor,
    n_iters: int,
) -> torch.Tensor:
    """Sinkhorn normalization over only valid rows/columns of padded matrices."""
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    log_alpha = log_alpha.masked_fill(~pair_valid, -torch.inf)
    for _ in range(n_iters):
        # ``torch.where`` guards rows/cols that are entirely invalid: their
        # logsumexp is -inf and (-inf) - (-inf) would propagate NaN.
        row_norm = torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = torch.where(pair_valid, log_alpha - row_norm, log_alpha)
        col_norm = torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = torch.where(pair_valid, log_alpha - col_norm, log_alpha)
    # Clamp keeps exp() within float32 representable range.
    return log_alpha.clamp(min=-50.0).exp().masked_fill(~pair_valid, 0.0)


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


class BCELoss(Loss):
    """Binary cross-entropy on logits. Targets > 0 are treated as positives."""

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        return F.binary_cross_entropy_with_logits(scores, (target > 0).float())


class FocalLoss(Loss):
    """Binary focal loss on logits (Lin et al., 2017).

        p_t = sigmoid(scores) for positives, 1 - sigmoid(scores) otherwise
        L   = -alpha_t * (1 - p_t)**gamma * log(p_t)

    ``gamma`` focuses on hard examples; ``alpha`` balances classes
    (set ``alpha < 0`` to disable). ``gamma=0`` with ``alpha < 0`` recovers BCE.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self._gamma = gamma
        self._alpha = alpha

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        y = (target > 0).to(scores.dtype)
        ce = F.binary_cross_entropy_with_logits(scores, y, reduction="none")
        p = torch.sigmoid(scores)
        p_t = p * y + (1.0 - p) * (1.0 - y)
        loss = ce * (1.0 - p_t) ** self._gamma
        if self._alpha >= 0:
            alpha_t = self._alpha * y + (1.0 - self._alpha) * (1.0 - y)
            loss = loss * alpha_t
        return loss.mean()


class LabelSmoothingBCELoss(Loss):
    """Binary cross-entropy with label smoothing.

    Hard targets ``{0, 1}`` become soft targets ``{eps, 1 - eps}``, capping
    overconfidence and adding robustness to label noise. ``smoothing=0``
    recovers plain BCE. ``smoothing`` must be in ``[0, 0.5)``.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        # eps must stay in [0, 0.5): at eps=0.5 both targets collapse to 0.5
        # (no class signal); eps>=0.5 inverts the positive/negative ordering.
        if not 0.0 <= smoothing < 0.5:
            raise ValueError(f"smoothing must be in [0, 0.5), got {smoothing}")
        self._smoothing = smoothing

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        y = (target > 0).to(scores.dtype)
        y_smooth = y * (1.0 - self._smoothing) + (1.0 - y) * self._smoothing
        return F.binary_cross_entropy_with_logits(scores, y_smooth)


class CrossEntropyLoss(Loss):
    """Multiclass cross-entropy; expects scores of shape [B, K]."""

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        return F.cross_entropy(scores, target.long())


class MSELoss(Loss):
    """Mean squared error for regression."""

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        return F.mse_loss(scores, target.float())


class MAELoss(Loss):
    """Mean absolute error for regression."""

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        return F.l1_loss(scores, target.float())


class QuantileRegressionLoss(Loss):
    """Pinball loss for one or more quantiles.

    For a single quantile, ``scores`` is shape ``[B]``. For multiple quantiles,
    ``scores`` is shape ``[B, Q]`` and ``quantile`` is a sequence of length ``Q``.
    """

    def __init__(self, quantile: float | Sequence[float] = 0.5) -> None:
        super().__init__()
        quantiles = torch.as_tensor(
            [quantile] if isinstance(quantile, float) else quantile,
            dtype=torch.float32,
        )
        if quantiles.ndim != 1 or quantiles.numel() == 0:
            raise ValueError("quantile must be a float or a non-empty sequence")
        if torch.any((quantiles <= 0) | (quantiles >= 1)):
            raise ValueError("quantile values must be strictly between 0 and 1")
        self.register_buffer("_quantiles", quantiles)

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        target_f = target.float()
        quantiles = self._quantiles
        if quantiles.numel() == 1:
            error = target_f.to(scores.dtype) - scores
            q = quantiles.squeeze(0)
        else:
            if scores.ndim != 2 or scores.size(1) != quantiles.numel():
                raise ValueError(
                    "scores must have shape [B, Q] when quantile is a sequence",
                )
            error = target_f.to(scores.dtype).unsqueeze(1) - scores
            q = quantiles.view(1, -1)
        return torch.maximum(q * error, (q - 1) * error).mean()


class SoftOrdinalBCELoss(Loss):
    """BCE against a soft ordinal target ``target / (num_classes - 1)``."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self._num_classes = num_classes

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        soft_target = target.float() / float(self._num_classes - 1)
        return F.binary_cross_entropy_with_logits(scores, soft_target)


class DistillBCELoss(Loss):
    """Distillation BCE: (1-beta)*BCE(hard) + beta*BCE(sigmoid(teacher/T)).

    Falls back to plain BCE when ``teacher_scores`` is not given.
    """

    def __init__(self, temperature: float = 2.0, beta: float = 0.5) -> None:
        super().__init__()
        self._temperature = temperature
        self._beta = beta
        self._fallback = BCELoss()

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
        teacher_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (zero_loss := _zero_loss_if_empty(scores)) is not None:
            return zero_loss
        if teacher_scores is None:
            return self._fallback(scores, target, group)
        binary_target = (target > 0).float()
        soft_label = torch.sigmoid(teacher_scores / self._temperature)
        hard_loss = F.binary_cross_entropy_with_logits(scores, binary_target)
        soft_loss = F.binary_cross_entropy_with_logits(
            scores / self._temperature,
            soft_label,
        )
        return (1 - self._beta) * hard_loss + self._beta * (self._temperature**2) * soft_loss


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------


class PairwiseMarginLoss(Loss):
    """Vectorized pairwise margin loss within groups."""

    requires_group = True

    def __init__(self, margin: float = 1.0, scale_by_diff: bool = True) -> None:
        super().__init__()
        self._margin = margin
        self._scale_by_diff = scale_by_diff

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pairwise_out = _pairwise_tensors(scores, target, group)
        if not pairwise_out.pos_pairs.any():
            return scores.sum() * 0
        margin: float | torch.Tensor = self._margin
        if self._scale_by_diff:
            margin = margin * pairwise_out.diff_target[pairwise_out.pos_pairs]
        return torch.clamp(
            margin - pairwise_out.diff_scores[pairwise_out.pos_pairs],
            min=0,
        ).mean()


class BPRLoss(Loss):
    """Vectorized BPR with uniform or all-pairs sampling within groups."""

    requires_group = True

    def __init__(self, sampling: str = "uniform") -> None:
        super().__init__()
        if sampling not in {"uniform", "all_pairs"}:
            raise ValueError(
                f"sampling must be 'uniform' or 'all_pairs', got {sampling!r}",
            )
        self._sampling = sampling

    @property
    def sampling(self) -> str:
        return self._sampling

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pairwise_out = _pairwise_tensors(scores, target, group)
        if not pairwise_out.pos_pairs.any():
            return scores.sum() * 0
        if self._sampling == "all_pairs":
            return F.softplus(
                -pairwise_out.diff_scores[pairwise_out.pos_pairs],
            ).mean()

        cand_mask = pairwise_out.pos_pairs.float()
        has_cand = cand_mask.sum(dim=2) > 0
        if not has_cand.any():
            return scores.sum() * 0
        cand_rows = cand_mask[has_cand]
        j_idx = torch.multinomial(cand_rows, num_samples=1).squeeze(1)
        s_i = pairwise_out.scores[has_cand]
        s_j = pairwise_out.scores[has_cand.nonzero(as_tuple=True)[0], j_idx]
        return F.softplus(-(s_i - s_j)).mean()


class LambdaRankLoss(Loss):
    """Vectorized LambdaRank: pairwise logistic loss weighted by |delta NDCG|."""

    requires_group = True

    def __init__(self, truncation_level: int | None = None) -> None:
        super().__init__()
        self._truncation_level = truncation_level

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pairwise_out = _pairwise_tensors(scores, target, group)
        if not pairwise_out.pos_pairs.any():
            return scores.sum() * 0
        ndcg = _ndcg_components(
            pairwise_out.scores,
            pairwise_out.target,
            pairwise_out.valid,
            self._truncation_level,
        )
        valid_group = (
            (pairwise_out.lengths >= 2)
            & (pairwise_out.target.sum(dim=1) > 0)
            & (ndcg.ideal_ndcg > 0)
        )
        valid_pos = pairwise_out.pos_pairs & rearrange(valid_group, "g -> g 1 1")
        if not valid_pos.any():
            return scores.sum() * 0
        gain_diff = rearrange(ndcg.gains, "g i -> g i 1") - rearrange(
            ndcg.gains,
            "g j -> g 1 j",
        )
        discount_diff = rearrange(ndcg.discounts, "g i -> g i 1") - rearrange(
            ndcg.discounts,
            "g j -> g 1 j",
        )
        delta_ndcg = torch.abs(gain_diff * discount_diff) / rearrange(
            ndcg.ideal_ndcg.clamp_min(torch.finfo(scores.dtype).tiny),
            "g -> g 1 1",
        )
        return (delta_ndcg[valid_pos] * F.softplus(-pairwise_out.diff_scores[valid_pos])).mean()


class FocalLambdaRankLoss(Loss):
    """Vectorized focal LambdaRank."""

    requires_group = True

    def __init__(
        self,
        truncation_level: int | None = 30,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self._truncation_level = truncation_level
        self._gamma = gamma

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pairwise_out = _pairwise_tensors(scores, target, group)
        if not pairwise_out.pos_pairs.any():
            return scores.sum() * 0
        ndcg = _ndcg_components(
            pairwise_out.scores,
            pairwise_out.target,
            pairwise_out.valid,
            self._truncation_level,
        )
        valid_group = (
            (pairwise_out.lengths >= 2)
            & (pairwise_out.target.sum(dim=1) > 0)
            & (ndcg.ideal_ndcg > 0)
        )
        valid_pos = pairwise_out.pos_pairs & rearrange(valid_group, "g -> g 1 1")
        if not valid_pos.any():
            return scores.sum() * 0
        gain_diff = rearrange(ndcg.gains, "g i -> g i 1") - rearrange(
            ndcg.gains,
            "g j -> g 1 j",
        )
        discount_diff = rearrange(ndcg.discounts, "g i -> g i 1") - rearrange(
            ndcg.discounts,
            "g j -> g 1 j",
        )
        delta_ndcg = torch.abs(gain_diff * discount_diff) / rearrange(
            ndcg.ideal_ndcg.clamp_min(torch.finfo(scores.dtype).tiny),
            "g -> g 1 1",
        )
        p_ij = torch.sigmoid(pairwise_out.diff_scores[valid_pos])
        focal = (1.0 - p_ij) ** self._gamma
        return (
            delta_ndcg[valid_pos] * focal * F.softplus(-pairwise_out.diff_scores[valid_pos])
        ).mean()


class LambdaNDCG2PPLoss(Loss):
    """Vectorized LambdaLoss ndcgLoss2++."""

    requires_group = True

    def __init__(
        self,
        sigma: float = 1.0,
        mu: float = 10.0,
        truncation_level: int | None = None,
    ) -> None:
        super().__init__()
        self._sigma = sigma
        self._mu = mu
        self._truncation_level = truncation_level

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pairwise_out = _pairwise_tensors(scores, target, group)
        if not pairwise_out.pos_pairs.any():
            return scores.sum() * 0
        ndcg = _ndcg_components(
            pairwise_out.scores,
            pairwise_out.target,
            pairwise_out.valid,
            self._truncation_level,
        )
        valid_group = (
            (pairwise_out.lengths >= 2)
            & (pairwise_out.target.sum(dim=1) > 0)
            & (ndcg.ideal_ndcg > 0)
        )
        valid_pos = pairwise_out.pos_pairs & rearrange(valid_group, "g -> g 1 1")
        if not valid_pos.any():
            return scores.sum() * 0
        gain_diff = rearrange(ndcg.gains, "g i -> g i 1") - rearrange(
            ndcg.gains,
            "g j -> g 1 j",
        )
        disc_diff = rearrange(ndcg.discounts, "g i -> g i 1") - rearrange(
            ndcg.discounts,
            "g j -> g 1 j",
        )
        rank_gap = (
            torch.abs(
                rearrange(ndcg.ranks, "g i -> g i 1") - rearrange(ndcg.ranks, "g j -> g 1 j"),
            )
            + 1.0
        )
        weights = torch.abs(disc_diff * gain_diff) + self._mu * torch.abs(
            1.0 / torch.log2(rank_gap + 1.0),
        )
        weights = weights / rearrange(
            ndcg.ideal_ndcg.clamp_min(torch.finfo(scores.dtype).tiny),
            "g -> g 1 1",
        )
        return (
            weights[valid_pos] * F.softplus(-self._sigma * pairwise_out.diff_scores[valid_pos])
        ).mean()


# ---------------------------------------------------------------------------
# Listwise
# ---------------------------------------------------------------------------


class ListwiseSoftmaxLoss(Loss):
    """Vectorized ListNet-style softmax loss within groups."""

    requires_group = True

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grouped_out = _grouped_1d(scores, target, group)
        if grouped_out.scores.numel() == 0:
            return scores.sum() * 0
        y_sum = torch.segment_reduce(
            grouped_out.target,
            "sum",
            lengths=grouped_out.lengths,
        )
        valid_group = (grouped_out.lengths >= 2) & (y_sum > 0)
        if not valid_group.any():
            return scores.sum() * 0
        max_s = torch.segment_reduce(
            grouped_out.scores,
            "max",
            lengths=grouped_out.lengths,
        )
        shifted = grouped_out.scores - max_s[grouped_out.row_group]
        exp_s = shifted.exp()
        log_denom = torch.segment_reduce(
            exp_s,
            "sum",
            lengths=grouped_out.lengths,
        ).log()
        log_softmax = shifted - log_denom[grouped_out.row_group]
        per_row = (
            -(
                grouped_out.target
                / y_sum.clamp_min(torch.finfo(scores.dtype).tiny)[grouped_out.row_group]
            )
            * log_softmax
        )
        group_loss = torch.segment_reduce(
            per_row,
            "sum",
            lengths=grouped_out.lengths,
        )
        return group_loss[valid_group].mean()


class ApproxNDCGLoss(Loss):
    """Vectorized differentiable ApproxNDCG."""

    requires_group = True

    def __init__(self, alpha: float = 10.0) -> None:
        super().__init__()
        self._alpha = alpha

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padded_out = _pad_by_group(scores, target, group)
        if padded_out.scores.numel() == 0:
            return scores.sum() * 0
        valid_group = (padded_out.lengths >= 2) & (padded_out.target.sum(dim=1) > 0)
        if not valid_group.any():
            return scores.sum() * 0
        pairwise_diff = rearrange(padded_out.scores, "g j -> g 1 j") - rearrange(
            padded_out.scores,
            "g i -> g i 1",
        )
        contrib = torch.sigmoid(self._alpha * pairwise_diff).masked_fill(
            ~rearrange(padded_out.valid, "g j -> g 1 j"),
            0.0,
        )
        approx_rank = 1.0 + contrib.sum(dim=2)
        gains = 2.0**padded_out.target - 1.0
        discounts = 1.0 / torch.log2(approx_rank + 1.0)
        ndcg = (gains * discounts * padded_out.valid).sum(dim=1)
        ideal_gains = torch.sort(
            gains.masked_fill(~padded_out.valid, -torch.inf),
            dim=1,
            descending=True,
        ).values
        ideal_gains = torch.where(
            torch.isfinite(ideal_gains),
            ideal_gains,
            torch.zeros_like(ideal_gains),
        )
        ideal_ranks = rearrange(
            torch.arange(
                1,
                padded_out.scores.size(1) + 1,
                device=padded_out.scores.device,
                dtype=padded_out.scores.dtype,
            ),
            "m -> 1 m",
        )
        ideal_ndcg = (ideal_gains / torch.log2(ideal_ranks + 1.0)).sum(dim=1)
        valid_group = valid_group & (ideal_ndcg > 0)
        if not valid_group.any():
            return scores.sum() * 0
        return -(ndcg[valid_group] / ideal_ndcg[valid_group]).mean()


class ListMLELoss(Loss):
    """Vectorized ListMLE with random tie-breaking via tiny random noise."""

    requires_group = True

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padded_out = _pad_by_group(scores, target, group)
        valid_group = (padded_out.lengths >= 2) & (padded_out.target.sum(dim=1) > 0)
        if not valid_group.any():
            return scores.sum() * 0
        # Equivalent distribution to random shuffle before sorting ties when
        # labels are integer-like.
        key = padded_out.target + torch.rand_like(padded_out.target) * 1e-6
        sorted_idx = torch.argsort(
            key.masked_fill(~padded_out.valid, -torch.inf),
            dim=1,
            descending=True,
        )
        s_sorted = torch.gather(padded_out.scores, 1, sorted_idx)
        valid_sorted = torch.gather(padded_out.valid, 1, sorted_idx)
        row_max = s_sorted.masked_fill(~valid_sorted, -torch.inf).max(dim=1, keepdim=True).values
        s_centered = s_sorted - row_max
        exp_s = s_centered.exp().masked_fill(~valid_sorted, 0.0)
        cumsums = torch.cumsum(exp_s.flip(1), dim=1).flip(1)
        per_pos = -(s_centered - torch.log(cumsums + 1e-10)).masked_fill(
            ~valid_sorted,
            0.0,
        )
        return per_pos.sum(dim=1)[valid_group].mean()


class NeuralNDCGLoss(Loss):
    """Vectorized NeuralNDCG with masked Sinkhorn over padded groups."""

    requires_group = True

    def __init__(
        self,
        temperature: float = 5.0,
        n_iters: int = 10,
        stochastic: bool = False,
    ) -> None:
        super().__init__()
        self._temperature = temperature
        self._n_iters = n_iters
        self._stochastic = stochastic

    @staticmethod
    def _sinkhorn(log_alpha: torch.Tensor, n_iters: int) -> torch.Tensor:
        return _masked_sinkhorn(
            log_alpha,
            torch.ones(log_alpha.shape[:2], dtype=torch.bool, device=log_alpha.device),
            n_iters,
        )

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padded_out = _pad_by_group(scores, target, group)
        valid_group = (padded_out.lengths >= 2) & (padded_out.target.sum(dim=1) > 0)
        if not valid_group.any():
            return scores.sum() * 0
        padded_out = PaddedGroups(
            scores=padded_out.scores[valid_group],
            target=padded_out.target[valid_group],
            valid=padded_out.valid[valid_group],
            lengths=padded_out.lengths[valid_group],
        )
        valid_f = padded_out.valid.to(padded_out.scores.dtype)
        group_sum = valid_f.sum(dim=1, keepdim=True)
        mean = (padded_out.scores * valid_f).sum(
            dim=1,
            keepdim=True,
        ) / group_sum.clamp_min(1.0)
        var = (((padded_out.scores - mean) * valid_f) ** 2).sum(
            dim=1,
            keepdim=True,
        ) / group_sum.clamp_min(1.0)
        s_norm = (padded_out.scores - mean) / var.sqrt().clamp(min=1e-6)

        pair_valid = rearrange(padded_out.valid, "g i -> g i 1") & rearrange(
            padded_out.valid,
            "g j -> g 1 j",
        )
        diff = torch.abs(
            rearrange(s_norm, "g i -> g i 1") - rearrange(s_norm, "g j -> g 1 j"),
        ).masked_fill(~pair_valid, 0.0)
        abs_diff_sum = diff.sum(dim=2, keepdim=True)
        max_len = padded_out.scores.size(1)
        idx = rearrange(
            torch.arange(
                1,
                max_len + 1,
                device=padded_out.scores.device,
                dtype=padded_out.scores.dtype,
            ),
            "m -> 1 m",
        )
        coeff = (
            rearrange(padded_out.lengths.to(padded_out.scores.dtype), "g -> g 1 1")
            + 1.0
            - 2.0 * rearrange(idx, "g m -> g 1 m")
        )
        c = coeff * rearrange(s_norm, "g i -> g i 1") - 2.0 * abs_diff_sum
        if self._stochastic and self.training:
            g = -torch.log(-torch.log(torch.rand_like(c) + 1e-20) + 1e-20)
            c = c + g
        p_hat = _masked_sinkhorn(c / self._temperature, padded_out.valid, self._n_iters)
        gains = (2.0**padded_out.target - 1.0).masked_fill(~padded_out.valid, 0.0)
        ideal_gains = torch.sort(
            gains.masked_fill(~padded_out.valid, -torch.inf),
            dim=1,
            descending=True,
        ).values
        ideal_gains = torch.where(
            torch.isfinite(ideal_gains),
            ideal_gains,
            torch.zeros_like(ideal_gains),
        )
        discounts = 1.0 / torch.log2(idx + 1.0)
        ideal_ndcg = (ideal_gains * discounts).sum(dim=1)
        valid_ideal = ideal_ndcg > 0
        if not valid_ideal.any():
            return scores.sum() * 0
        sorted_gains = torch.einsum("g i j, g i -> g j", p_hat, gains)
        ndcg = (sorted_gains * discounts).sum(dim=1)
        return -(ndcg[valid_ideal] / ideal_ndcg[valid_ideal]).mean()


# ---------------------------------------------------------------------------
# Ordinal
# ---------------------------------------------------------------------------


class CORALOrdinalLoss(Loss):
    """CORAL ordinal loss with K-1 cumulative binary heads.

        P(Y >= k) = sigmoid(z - b_k)

    Monotone thresholds ``b_1 <= ... <= b_{K-1}`` are enforced via a cumulative
    sum of softplus increments. The model emits a scalar logit; biases live in
    the loss. ``preinit_bias`` starts thresholds equally spaced.
    """

    def __init__(self, num_classes: int = 4, preinit_bias: bool = True) -> None:
        super().__init__()
        self._num_classes = num_classes
        if preinit_bias:
            init = torch.full(
                (num_classes - 1,),
                math.log(math.expm1(1.0 / (num_classes - 1))),
            )
        else:
            init = torch.zeros(num_classes - 1)
        self._bias_deltas = torch.nn.Parameter(init)

    def _thresholds(self) -> torch.Tensor:
        deltas = F.softplus(self._bias_deltas)
        thresh = torch.cumsum(deltas, dim=0)
        return thresh - thresh.mean()

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = scores.device
        levels = torch.arange(1, self._num_classes, device=device).float()
        y = (rearrange(target, "b -> b 1") >= rearrange(levels, "k -> 1 k")).float()
        thresh = self._thresholds()
        z = rearrange(scores, "b -> b 1") - rearrange(thresh, "k -> 1 k")
        return F.binary_cross_entropy_with_logits(z, y)


class CORNLoss(Loss):
    """CORN conditional ordinal regression (Shi et al., 2023).

    Unlike CORAL, no monotonicity constraint on biases; rank consistency comes
    from the conditional factorization ``P(y >= k) = prod_j sigmoid(z - b_j)``.
    Head ``k`` trains only on the subset ``{i : target_i >= k-1}``.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._biases = torch.nn.Parameter(torch.zeros(num_classes - 1))

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses_weighted = []
        total_n = 0

        for k in range(1, self._num_classes):
            mask = target >= (k - 1)
            n_k = int(mask.sum().item())
            if n_k < 2:
                continue
            z_k = scores[mask] - self._biases[k - 1]
            y_k = (target[mask] >= k).float()
            losses_weighted.append(F.binary_cross_entropy_with_logits(z_k, y_k) * n_k)
            total_n += n_k

        if not losses_weighted:
            return scores.sum() * 0
        return torch.stack(losses_weighted).sum() / total_n


class BinomialOrdinalLoss(Loss):
    """Ordinal loss via a binomial distribution (Beckham & Pal, 2017).

    With ``p = sigmoid(z)``, ``P(y = k) = C(K-1, k) * p^k * (1-p)^(K-1-k)`` and
    ``L = -log P(y = target)``. The distribution is always unimodal.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self._num_classes = num_classes
        N = num_classes - 1
        k = torch.arange(num_classes, dtype=torch.float)
        log_coeffs = (
            torch.lgamma(torch.tensor(float(N + 1)))
            - torch.lgamma(k + 1)
            - torch.lgamma(torch.tensor(float(N + 1)) - k)
        )
        self.register_buffer("_log_coeffs", log_coeffs)
        self.register_buffer("_k", k)
        self._N = float(N)

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_p = F.logsigmoid(scores)  # [B]
        log_1mp = F.logsigmoid(-scores)  # [B]
        log_probs = (
            rearrange(self._log_coeffs, "k -> 1 k")
            + rearrange(self._k, "k -> 1 k") * rearrange(log_p, "b -> b 1")
            + (self._N - rearrange(self._k, "k -> 1 k")) * rearrange(log_1mp, "b -> b 1")
        )  # [B, K]
        nll = -log_probs.gather(1, rearrange(target.long(), "b -> b 1")).squeeze(1)
        return nll.mean()


class SoftLabelCORALoss(Loss):
    """CORAL with label smoothing ``{0, 1} -> {eps, 1-eps}``.

    ``smoothing=0`` recovers standard CORAL.
    """

    def __init__(
        self,
        num_classes: int = 4,
        smoothing: float = 0.05,
        preinit_bias: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._smoothing = smoothing
        self._reduction = reduction
        if preinit_bias:
            init = torch.full(
                (num_classes - 1,),
                math.log(math.expm1(1.0 / (num_classes - 1))),
            )
        else:
            init = torch.zeros(num_classes - 1)
        self._bias_deltas = torch.nn.Parameter(init)

    def _thresholds(self) -> torch.Tensor:
        deltas = F.softplus(self._bias_deltas)
        thresh = torch.cumsum(deltas, dim=0)
        return thresh - thresh.mean()

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = scores.device
        levels = torch.arange(1, self._num_classes, device=device).float()
        y = (rearrange(target, "b -> b 1") >= rearrange(levels, "k -> 1 k")).float()
        y_smooth = y * (1.0 - self._smoothing) + (1.0 - y) * self._smoothing
        z = rearrange(scores, "b -> b 1") - rearrange(self._thresholds(), "k -> 1 k")
        return F.binary_cross_entropy_with_logits(z, y_smooth, reduction=self._reduction)


class WeightedCORALoss(Loss):
    """CORAL weighting each sample by its target level (level 0 -> weight 1)."""

    def __init__(self, num_classes: int = 4, preinit_bias: bool = True) -> None:
        super().__init__()
        self._num_classes = num_classes
        if preinit_bias:
            init = torch.full(
                (num_classes - 1,),
                math.log(math.expm1(1.0 / (num_classes - 1))),
            )
        else:
            init = torch.zeros(num_classes - 1)
        self._bias_deltas = torch.nn.Parameter(init)

    def _thresholds(self) -> torch.Tensor:
        deltas = F.softplus(self._bias_deltas)
        thresh = torch.cumsum(deltas, dim=0)
        return thresh - thresh.mean()

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = scores.device
        target = target.float()
        levels = torch.arange(1, self._num_classes, device=device).float()
        y = (rearrange(target, "b -> b 1") >= rearrange(levels, "k -> 1 k")).float()
        thresh = self._thresholds()
        z = rearrange(scores, "b -> b 1") - rearrange(thresh, "k -> 1 k")
        weights = torch.where(target == 0, torch.ones_like(target), target)  # [N]
        per_sample = F.binary_cross_entropy_with_logits(
            z,
            y,
            reduction="none",
        )  # [N, K-1]
        return (per_sample * rearrange(weights, "b -> b 1")).sum() / (
            weights.sum() * (self._num_classes - 1)
        )


class CORALRawBiasLoss(Loss):
    """CORAL with plain (non-monotone) bias parameters.

    Unlike ``CORALOrdinalLoss``, biases are free parameters without the
    cumsum(softplus) monotonicity guarantee. ``preinit_bias`` uses ascending init.
    """

    def __init__(self, num_classes: int = 4, preinit_bias: bool = False) -> None:
        super().__init__()
        self._num_classes = num_classes
        if preinit_bias:
            init = (
                torch.arange(1, num_classes).float() / (num_classes - 1)
                - (torch.arange(1, num_classes).float() / (num_classes - 1)).mean()
            )
        else:
            init = torch.zeros(num_classes - 1)
        self._biases = torch.nn.Parameter(init)

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = scores.device
        levels = torch.arange(1, self._num_classes, device=device).float()
        y = (rearrange(target, "b -> b 1") >= rearrange(levels, "k -> 1 k")).float()
        z = rearrange(scores, "b -> b 1") - rearrange(self._biases, "k -> 1 k")
        return F.binary_cross_entropy_with_logits(z, y)


class SoftLabelCORALRawBiasLoss(Loss):
    """CORALRawBias with label smoothing ``{0, 1} -> {eps, 1-eps}``."""

    def __init__(
        self,
        num_classes: int = 4,
        smoothing: float = 0.1,
        preinit_bias: bool = False,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._smoothing = smoothing
        if preinit_bias:
            init = (
                torch.arange(1, num_classes).float() / (num_classes - 1)
                - (torch.arange(1, num_classes).float() / (num_classes - 1)).mean()
            )
        else:
            init = torch.zeros(num_classes - 1)
        self._biases = torch.nn.Parameter(init)

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = scores.device
        levels = torch.arange(1, self._num_classes, device=device).float()
        y = (rearrange(target, "b -> b 1") >= rearrange(levels, "k -> 1 k")).float()
        y_smooth = y * (1.0 - self._smoothing) + (1.0 - y) * self._smoothing
        z = rearrange(scores, "b -> b 1") - rearrange(self._biases, "k -> 1 k")
        return F.binary_cross_entropy_with_logits(z, y_smooth)


class CORALLayerLoss(Loss):
    """CORAL where the K-1 heads live in the model (CoralLayer) rather than the loss.

    Expects scores of shape [B, K-1] from a model with a CORAL output head.
    Sum the K-1 outputs at inference to recover a scalar ranking score.
    """

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scores.dim() != 2:
            raise ValueError(
                f"CORALLayerLoss expects scores.shape=[B, K-1], got {tuple(scores.shape)}. "
                "Configure the model to emit K-1 outputs.",
            )
        device = scores.device
        num_classes = scores.size(1) + 1
        levels = torch.arange(1, num_classes, device=device).float()
        y = (rearrange(target, "b -> b 1") >= rearrange(levels, "k -> 1 k")).float()
        return F.binary_cross_entropy_with_logits(scores, y)


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


class CombinedLoss(Loss):
    """Convex combination of two named losses: ``(1-alpha)*L1 + alpha*L2``.

    ``loss1_kwargs`` / ``loss2_kwargs`` are forwarded to the sub-loss constructors.
    """

    def __init__(
        self,
        loss1: str = "bce",
        loss2: str = "approx_ndcg",
        alpha: float = 0.5,
        loss1_kwargs: dict | None = None,
        loss2_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self._loss1 = LOSSES[loss1](**(loss1_kwargs or {}))
        self._loss2 = LOSSES[loss2](**(loss2_kwargs or {}))
        self._alpha = alpha
        self.requires_group = self._loss1.requires_group or self._loss2.requires_group

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        l1 = self._loss1(scores, target, group)
        l2 = self._loss2(scores, target, group)
        return (1 - self._alpha) * l1 + self._alpha * l2


class BCEV2LambdaRankCombined(Loss):
    """``(1-alpha) * BCE + alpha * LambdaRank@30``."""

    requires_group = True

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        self._bce = BCELoss()
        self._ltr = LambdaRankLoss(truncation_level=30)
        self._alpha = alpha

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (1 - self._alpha) * self._bce(
            scores,
            target,
            group,
        ) + self._alpha * self._ltr(scores, target, group)


class WBCE2CoralCombinedLoss(Loss):
    """``(1-alpha) * BCE + alpha * CORAL``. CORAL biases reach the optimizer."""

    def __init__(self, alpha: float = 0.3, num_classes: int = 4) -> None:
        super().__init__()
        self._bce = BCELoss()
        self._coral = CORALOrdinalLoss(num_classes=num_classes)
        self._alpha = alpha

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (1 - self._alpha) * self._bce(
            scores,
            target,
            group,
        ) + self._alpha * self._coral(scores, target, group)


class WBCECORNLoss(Loss):
    """``(1-alpha) * BCE + alpha * CORN``."""

    def __init__(self, alpha: float = 0.3, num_classes: int = 4) -> None:
        super().__init__()
        self._alpha = alpha
        self._bce = BCELoss()
        self._corn = CORNLoss(num_classes=num_classes)

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (1 - self._alpha) * self._bce(
            scores,
            target,
            group,
        ) + self._alpha * self._corn(scores, target, group)


class CoralLambdaRankLoss(Loss):
    """``(1-alpha) * CORAL + alpha * LambdaRank@k``."""

    requires_group = True

    def __init__(
        self,
        alpha: float = 0.3,
        num_classes: int = 4,
        truncation_level: int = 30,
    ) -> None:
        super().__init__()
        self._alpha = alpha
        self._coral = CORALOrdinalLoss(num_classes=num_classes)
        self._lambdarank = LambdaRankLoss(truncation_level=truncation_level)

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (1 - self._alpha) * self._coral(
            scores,
            target,
            group,
        ) + self._alpha * self._lambdarank(scores, target, group)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

LOSSES: dict[str, type[Loss]] = {
    # pointwise
    "bce": BCELoss,
    "focal": FocalLoss,
    "ls_bce": LabelSmoothingBCELoss,
    "cross_entropy": CrossEntropyLoss,
    "mse": MSELoss,
    "mae": MAELoss,
    "quantile": QuantileRegressionLoss,
    "soft_ordinal_bce": SoftOrdinalBCELoss,
    "distill_bce": DistillBCELoss,
    # pairwise
    "pairwise": PairwiseMarginLoss,
    "bpr": BPRLoss,
    "lambdarank": LambdaRankLoss,
    "focal_lambdarank": FocalLambdaRankLoss,
    "lambda_ndcg2pp": LambdaNDCG2PPLoss,
    # listwise
    "listwise": ListwiseSoftmaxLoss,
    "approx_ndcg": ApproxNDCGLoss,
    "list_mle": ListMLELoss,
    "neural_ndcg": NeuralNDCGLoss,
    # ordinal
    "coral": CORALOrdinalLoss,
    "corn": CORNLoss,
    "binomial_ordinal": BinomialOrdinalLoss,
    "soft_label_coral": SoftLabelCORALoss,
    "weighted_coral": WeightedCORALoss,
    "coral_raw_bias": CORALRawBiasLoss,
    "soft_label_coral_raw_bias": SoftLabelCORALRawBiasLoss,
    "coral_layer": CORALLayerLoss,
    # combined
    "combined": CombinedLoss,
    "coral_lambdarank": CoralLambdaRankLoss,
}


def make_loss(loss: str | Loss, **kwargs: Any) -> Loss:
    """Create a loss from a ``LOSSES`` name (plus kwargs) or return a ready loss."""
    if isinstance(loss, Loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}")
    return LOSSES[loss](**kwargs)
