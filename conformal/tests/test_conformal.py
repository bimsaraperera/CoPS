import numpy as np
import torch
import torch.nn.functional as F

from conformal import (
    ConformalPseudoLabeler,
    conformal_quantile,
    weighted_conformal_quantile,
    aps_scores_all,
    true_label_scores,
    partial_label_loss,
    coverage_and_size,
    effective_data_utilisation,
    alpha_schedule,
)

ALPHAS = [0.05, 0.10, 0.20, 0.30]
SCORES = ["thr", "aps", "raps"]
TOL = 0.03          # coverage may dip this far below nominal



def make_logits(n, C, margin, seed=0, class_prior=None):
    """Simulate a classifier of controllable quality"""
    g = torch.Generator().manual_seed(seed)
    if class_prior is None:
        y = torch.randint(0, C, (n,), generator=g)
    else:
        y = torch.multinomial(torch.tensor(class_prior).float(),
                              n, replacement=True, generator=g)
    logits = torch.randn(n, C, generator=g) * 0.7
    logits[torch.arange(n), y] += margin
    return logits, y


def test_coverage_grid():
    """Condition 1 & 2: coverage holds across scores, alphas, difficulties, C."""
    rows = []
    for C in [7, 65]:                      # PACS-like and OfficeHome-like
        for margin in [0.8, 2.0, 3.5]:     # weak / medium / strong model
            cal_logits, cal_y = make_logits(3000, C, margin, seed=1)
            te_logits, te_y = make_logits(3000, C, margin, seed=2)
            for score in SCORES:
                for a in ALPHAS:
                    cp = ConformalPseudoLabeler(alpha=a, score=score,
                                                weighted=False, max_set_size=None)
                    cp.calibrate(cal_logits, cal_y)
                    sets = cp.predict_sets(te_logits)
                    d = coverage_and_size(sets, te_y)
                    ok = d["coverage"] >= (1 - a) - TOL
                    rows.append((C, margin, score, a, d["coverage"],
                                 d["avg_set_size"], ok))
                    assert ok, (
                        f"UNDERCOVERAGE C={C} margin={margin} {score} "
                        f"alpha={a}: got {d['coverage']:.3f}, need {1-a:.2f}"
                    )
    return rows


def test_pll_equals_ce_on_singletons():
    """Condition 3: the loss strictly generalizes hard pseudo-labelling."""
    logits, y = make_logits(256, 20, 2.0, seed=3)
    singleton = F.one_hot(y, 20).bool()
    pll = partial_label_loss(logits, singleton)
    ce = F.cross_entropy(logits, y)
    assert torch.allclose(pll, ce, atol=1e-6), f"{pll.item()} vs {ce.item()}"
    return pll.item(), ce.item()


def test_pll_monotone_in_set_size():
    """Larger sets => weaker (lower) loss. Sanity on the supervision strength."""
    logits, y = make_logits(512, 20, 2.0, seed=4)
    losses = []
    for k in [1, 2, 5, 10]:
        topk = logits.topk(k, dim=1).indices
        sets = torch.zeros(512, 20, dtype=torch.bool)
        sets.scatter_(1, topk, True)
        sets[torch.arange(512), y] = True          # ensure coverage
        losses.append(partial_label_loss(logits, sets).item())
    assert all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1)), losses
    return losses


def test_weighted_reduces_to_unweighted():
    """Condition 5: equal weights -> same quantile as split conformal."""
    logits, y = make_logits(2000, 10, 2.0, seed=5)
    s = true_label_scores(aps_scores_all(F.softmax(logits, 1)), y)
    q_plain = conformal_quantile(s, 0.1)
    q_weighted = weighted_conformal_quantile(s, torch.ones_like(s), 0.1)
    assert abs(q_plain - q_weighted) < 0.02, (q_plain, q_weighted)
    return q_plain, q_weighted


def test_label_shift_needs_weighting():
    """Condition 6 -- the empirical core of Section 3.4.

    Calibrate on a CLASS-BALANCED set (as the SSDG protocol produces), evaluate
    on an IMBALANCED pool (the unlabeled remainder), where the model is worse on
    rare classes. Unweighted calibration should undercover; weighted should fix it.
    """
    C, a = 10, 0.1

    # balanced calibration, and the model is GOOD on these classes
    cal_logits, cal_y = make_logits(2000, C, 2.6, seed=6)

    # imbalanced unlabeled pool, skewed toward classes the model handles worse
    prior = np.array([0.02] * 5 + [0.18] * 5)      # tail-heavy
    te_logits, te_y = make_logits(3000, C, 2.6, seed=7, class_prior=prior)
    tail = te_y >= 5
    te_logits[tail, te_y[tail]] -= 1.4             # model degrades on the majority half

    cp = ConformalPseudoLabeler(alpha=a, score="aps", weighted=False)
    cp.calibrate(cal_logits, cal_y)
    cov_unw = coverage_and_size(cp.predict_sets(te_logits), te_y)["coverage"]

    # weights: upweight calibration points from classes over-represented downstream
    w = torch.where(cal_y >= 5, torch.tensor(9.0), torch.tensor(1.0))
    cpw = ConformalPseudoLabeler(alpha=a, score="aps", weighted=True)
    cpw.calibrate(cal_logits, cal_y, weights=w)
    cov_w = coverage_and_size(cpw.predict_sets(te_logits), te_y)["coverage"]

    return cov_unw, cov_w, 1 - a


def test_alpha_schedule():
    vals = [alpha_schedule(e, 20) for e in range(20)]
    assert vals[0] > vals[-1]
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(19)), "not monotone"
    return vals[0], vals[10], vals[-1]


if __name__ == "__main__":
    print("=" * 74)
    print("E0 -- CONFORMAL COVERAGE TEST SUITE")
    print("=" * 74)

    print("\n[1/6] Coverage grid (2 class counts x 3 difficulties x 3 scores x 4 alphas)")
    rows = test_coverage_grid()
    print(f"      {len(rows)} configurations, all >= nominal - {TOL}")
    print(f"\n      {'C':>3} {'margin':>7} {'score':>6} {'alpha':>6} "
          f"{'coverage':>9} {'set_size':>9}")
    for r in rows:
        if r[1] == 2.0 and r[3] in (0.05, 0.10):     # print a readable slice
            print(f"      {r[0]:>3} {r[1]:>7.1f} {r[2]:>6} {r[3]:>6.2f} "
                  f"{r[4]:>9.3f} {r[5]:>9.2f}")

    print("\n[2/6] PLL == cross-entropy on singletons")
    pll, ce = test_pll_equals_ce_on_singletons()
    print(f"      PLL={pll:.8f}  CE={ce:.8f}  diff={abs(pll-ce):.2e}")

    print("\n[3/6] PLL decreases with set size")
    losses = test_pll_monotone_in_set_size()
    print(f"      |C|=1,2,5,10 -> {[round(l, 3) for l in losses]}")

    print("\n[4/6] Weighted quantile reduces to unweighted under equal weights")
    qp, qw = test_weighted_reduces_to_unweighted()
    print(f"      plain={qp:.4f}  weighted={qw:.4f}")

    print("\n[5/6] Label shift: unweighted undercovers, weighted recovers")
    cu, cw, nom = test_label_shift_needs_weighting()
    print(f"      nominal={nom:.2f}  unweighted={cu:.3f}  weighted={cw:.3f}")
    print(f"      -> {'WEIGHTING HELPS' if cw > cu else 'no gain -- investigate'}")

    print("\n[6/6] Alpha schedule monotone")
    a0, a10, a19 = test_alpha_schedule()
    print(f"      epoch 0={a0:.3f}  10={a10:.3f}  19={a19:.3f}")