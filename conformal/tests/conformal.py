from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def aps_scores_all(probs: torch.Tensor, randomized: bool = True) -> torch.Tensor:
    """APS nonconformity score for *every* class.

    s(x, y) = cumulative sorted probability mass down to and including y.
    Lower score => more conforming => more likely to be in the prediction set.

    Args:
        probs: (N, C) softmax probabilities.
        randomized: subtract U * p_y for exact (rather than conservative)
            coverage. Turn this OFF at training time if you want determinism;
            turn it ON when you report coverage numbers in the paper.

    Returns:
        (N, C) scores aligned to the original class ordering.
    """
    sorted_probs, sort_idx = probs.sort(dim=1, descending=True)
    cumsum = sorted_probs.cumsum(dim=1)

    if randomized:
        u = torch.rand_like(sorted_probs)
        cumsum = cumsum - u * sorted_probs

    # scatter back to original class order
    scores = torch.empty_like(cumsum)
    scores.scatter_(1, sort_idx, cumsum)
    return scores


def raps_scores_all(
    probs: torch.Tensor,
    k_reg: int = 5,
    lam_reg: float = 0.01,
    randomized: bool = True,
) -> torch.Tensor:
    """RAPS: APS plus a penalty on classes beyond rank k_reg.

    Shrinks set sizes considerably on many-class datasets (OfficeHome has 65
    classes, miniDomainNet 126), which is exactly where naive APS blows up.
    """
    sorted_probs, sort_idx = probs.sort(dim=1, descending=True)
    cumsum = sorted_probs.cumsum(dim=1)

    if randomized:
        u = torch.rand_like(sorted_probs)
        cumsum = cumsum - u * sorted_probs

    ranks = torch.arange(1, probs.size(1) + 1, device=probs.device).float()
    penalty = lam_reg * torch.clamp(ranks - k_reg, min=0.0)
    cumsum = cumsum + penalty.unsqueeze(0)

    scores = torch.empty_like(cumsum)
    scores.scatter_(1, sort_idx, cumsum)
    return scores


def thr_scores_all(probs: torch.Tensor) -> torch.Tensor:
    """Simple threshold score: s(x, y) = 1 - p(y | x)."""
    return 1.0 - probs


SCORE_FNS = {
    "aps": aps_scores_all,
    "raps": raps_scores_all,
    "thr": thr_scores_all,
}


def true_label_scores(all_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pick out s(x_i, y_i) for the observed label."""
    return all_scores.gather(1, labels.view(-1, 1)).squeeze(1)


def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of calibration scores.

    Uses ceil((n+1)(1-alpha))/n, the standard split-conformal correction that
    gives you the >= 1 - alpha marginal coverage guarantee.
    """
    n = scores.numel()
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(min(level, 1.0))
    return torch.quantile(scores.float(), level).item()


def weighted_conformal_quantile(
    scores: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
) -> float:
    """Weighted quantile for covariate shift (Tibshirani et al., 2019).

    Normalises weights over calibration points *plus* a point mass at +inf for
    the test point, then takes the smallest score whose cumulative normalised
    weight reaches 1 - alpha. This is the piece that adapts calibration to the
    shift between source domains -- the axis on which this method differs from
    a plain CCSSL transplant.

    Args:
        scores: (n,) calibration scores.
        weights: (n,) non-negative likelihood-ratio weights w(x_i).
        alpha: miscoverage level.
    """
    if scores.numel() == 0:
        return float("inf")

    scores = scores.float()
    weights = weights.float().clamp(min=1e-8)

    order = scores.argsort()
    s_sorted = scores[order]
    w_sorted = weights[order]

    # test point carries weight 1 in the self-normalised formulation
    total = w_sorted.sum() + 1.0
    cum = w_sorted.cumsum(0) / total

    idx = torch.searchsorted(cum, torch.tensor(1.0 - alpha, device=cum.device))
    if idx >= s_sorted.numel():
        return float("inf")
    return s_sorted[idx].item()


class DomainWeightEstimator(nn.Module):
    """Estimates w(x) = p_pool(x) / p_domain(x) via a domain discriminator.

    Trained as a K-way domain classifier on (frozen or detached) backbone
    features. For a calibration point from domain k:

        w(x) = (1/K) / p(domain = k | x)

    Clipped to [1/clip, clip] -- unclipped ratios are the single most common
    source of instability in weighted conformal prediction with small
    calibration sets, and a reviewer will ask about it.
    """

    def __init__(self, feat_dim: int, num_domains: int, hidden: int = 256,
                 clip: float = 10.0):
        super().__init__()
        self.num_domains = num_domains
        self.clip = clip
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

    @torch.no_grad()
    def weights(self, feats: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        """Likelihood-ratio weight for each calibration point."""
        p = F.softmax(self.net(feats), dim=1)
        p_k = p.gather(1, domains.view(-1, 1)).squeeze(1).clamp(min=1e-6)
        w = (1.0 / self.num_domains) / p_k
        return w.clamp(1.0 / self.clip, self.clip)


class ConformalPseudoLabeler:
    """Turns model logits into set-valued pseudo-labels with controlled coverage.

    Typical loop (see trainer):
        calibrator.calibrate(cal_logits, cal_labels, cal_feats, cal_domains)
        sets = calibrator.predict_sets(unlabeled_logits)   # (B, C) bool
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score: str = "raps",
        weighted: bool = True,
        max_set_size: int | None = None,
        k_reg: int = 5,
        lam_reg: float = 0.01,
        randomized: bool = False,
    ):
        assert score in SCORE_FNS, f"unknown score {score}"
        self.alpha = alpha
        # CRITICAL: randomization must match between calibrate() and
        # predict_sets(). Randomizing only at calibration lowers the scores
        # there, shrinks qhat, and silently undercovers. Default False =
        # deterministic and slightly conservative, which is what you want
        # inside a training loop. Set True only to report exact coverage.
        self.randomized = randomized
        self.score = score
        self.weighted = weighted
        self.max_set_size = max_set_size
        self.k_reg = k_reg
        self.lam_reg = lam_reg
        self.qhat: float = float("inf")

    def _scores(self, probs: torch.Tensor, randomized: bool) -> torch.Tensor:
        if self.score == "raps":
            return raps_scores_all(probs, self.k_reg, self.lam_reg, randomized)
        if self.score == "aps":
            return aps_scores_all(probs, randomized)
        return thr_scores_all(probs)

    @torch.no_grad()
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> float:
        """Recompute qhat from a held-out labelled calibration split."""
        probs = F.softmax(logits, dim=1)
        all_s = self._scores(probs, randomized=self.randomized)
        s = true_label_scores(all_s, labels)

        if self.weighted and weights is not None:
            self.qhat = weighted_conformal_quantile(s, weights, self.alpha)
        else:
            self.qhat = conformal_quantile(s, self.alpha)
        return self.qhat

    @torch.no_grad()
    def predict_sets(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, C) boolean membership mask. Always contains the argmax."""
        probs = F.softmax(logits, dim=1)
        all_s = self._scores(probs, randomized=self.randomized)
        sets = all_s <= self.qhat

        # guarantee non-empty: fall back to top-1
        empty = ~sets.any(dim=1)
        if empty.any():
            top1 = probs.argmax(dim=1)
            sets[empty, top1[empty]] = True

        if self.max_set_size is not None:
            sets = _truncate_sets(probs, sets, self.max_set_size)
        return sets


def _truncate_sets(probs: torch.Tensor, sets: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest-probability members of each set.

    Breaks the formal guarantee (it can only shrink sets), so report results
    with and without it. In practice it stops OfficeHome/miniDomainNet sets
    from degenerating to near-uniform early in training.
    """
    masked = probs.masked_fill(~sets, -1.0)
    topk = masked.topk(min(k, probs.size(1)), dim=1).indices
    out = torch.zeros_like(sets)
    out.scatter_(1, topk, True)
    return out & sets


def partial_label_loss(
    logits: torch.Tensor,
    label_sets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """-log sum_{y in C(x)} p(y | x).

    The set-valued generalisation of FixMatch's cross-entropy consistency term.
    When |C(x)| == 1 this reduces *exactly* to standard hard pseudo-labelling,
    which is a useful sanity check and a good sentence for the paper.
    """
    log_probs = F.log_softmax(logits, dim=1)
    neg_inf = torch.finfo(log_probs.dtype).min
    masked = log_probs.masked_fill(~label_sets, neg_inf)
    loss = -torch.logsumexp(masked, dim=1)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def set_size_penalty(label_sets: torch.Tensor, target_size: float = 1.0,
                     weight: float = 0.0) -> torch.Tensor:
    """Optional pressure toward singleton sets as training progresses.

    Note this is a diagnostic-style penalty on the *produced* sets; it does not
    backprop through set construction (which is non-differentiable). Use the
    alpha schedule below as the primary mechanism instead.
    """
    if weight == 0.0:
        return torch.zeros((), device=label_sets.device)
    sizes = label_sets.float().sum(dim=1)
    return weight * F.relu(sizes - target_size).mean()


def alpha_schedule(epoch: int, max_epoch: int, a_start: float = 0.30,
                   a_end: float = 0.05, mode: str = "cosine") -> float:
    """Anneal miscoverage from loose (large sets, high utilisation) to tight.

    Early training: the model is bad, so demand high coverage -> large sets ->
    weak but non-vacuous supervision on *all* unlabelled data.
    Late training: the model is good, so tighten -> sets collapse to singletons
    -> recovers FixMatch-quality supervision.

    This schedule *is* the quantity-quality trade-off made explicit, and it is
    the sentence that answers "why not just use a threshold?".
    """
    if max_epoch <= 1:
        return a_end
    t = epoch / (max_epoch - 1)
    if mode == "linear":
        return a_start + t * (a_end - a_start)
    return a_end + 0.5 * (a_start - a_end) * (1 + np.cos(np.pi * t))



@torch.no_grad()
def coverage_and_size(label_sets: torch.Tensor, y_true: torch.Tensor) -> dict:
    """Empirical coverage and average set size.

    Coverage on held-out *source* data is your validity check.
    Coverage on the *target* domain is the honest degradation plot -- report it,
    do not hide it. Reviewers respect the concession and it is the empirical
    content of the non-exchangeability discussion.
    """
    covered = label_sets.gather(1, y_true.view(-1, 1)).squeeze(1).float()
    sizes = label_sets.float().sum(dim=1)
    return {
        "coverage": covered.mean().item(),
        "avg_set_size": sizes.mean().item(),
        "singleton_frac": (sizes == 1).float().mean().item(),
        "utilisation": 1.0,  # every sample is used -- contrast with mask rate
    }


@torch.no_grad()
def effective_data_utilisation(label_sets: torch.Tensor, y_true: torch.Tensor) -> float:
    """SemAlign's EDU, generalised to sets.

    For hard pseudo-labels EDU = fraction of the dataset retained AND correct.
    For sets the natural analogue is the fraction whose set contains the true
    label, weighted by informativeness 1/|C|. A singleton correct set scores 1;
    a 10-class set containing the truth scores 0.1. This makes EDU comparable
    across set and threshold methods, which you need for the headline table.
    """
    covered = label_sets.gather(1, y_true.view(-1, 1)).squeeze(1).float()
    sizes = label_sets.float().sum(dim=1).clamp(min=1.0)
    return (covered / sizes).mean().item()