import numpy as np
from scipy.special import softmax

from sse.evaluate import metrics as M


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(0)
    n = 200_000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.uniform(0, 1, n) < conf
    probs = np.stack([conf, 1 - conf], axis=1)
    y = np.where(correct, 0, 1)
    assert M.ece(probs, y) < 0.005


def test_ece_of_an_overconfident_predictor():
    probs = np.tile([[0.99, 0.01]], (100, 1))
    y = np.array([0] * 70 + [1] * 30)
    assert np.isclose(M.ece(probs, y), 0.29)


def test_temperature_recovers_a_known_scaling():
    rng = np.random.default_rng(1)
    true_logits = rng.normal(size=(20_000, 5)) * 2
    y = np.array([rng.choice(5, p=p) for p in softmax(true_logits, axis=1)])
    t = M.fit_temperature(true_logits * 3.0, y)
    assert abs(t - 3.0) < 0.15


def test_auroc_matches_definition_and_handles_ties():
    assert M.auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert M.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert M.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=500), rng.normal(1, size=600)
    brute = np.mean(b[None, :] > a[:, None]) + 0.5 * np.mean(b[None, :] == a[:, None])
    assert np.isclose(M.auroc(a, b), brute)


def test_fpr_at_95_tpr():
    known = np.arange(100, dtype=float)
    novel = np.arange(50, 150, dtype=float)
    assert np.isclose(M.fpr_at_tpr(known, novel), 0.45, atol=0.02)


def test_risk_coverage_abstains_on_least_confident_first():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    y = np.array([0, 0, 1, 1])  # the two least confident are wrong
    rc = M.risk_coverage(probs, y, points=4)
    assert rc["curve"][1] == {"coverage": 0.5, "risk": 0.0}
    assert rc["curve"][3]["risk"] == 0.5


def test_mahalanobis_and_knn_rank_far_points_higher():
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.normal(0, 1, (300, 4)), rng.normal(5, 1, (300, 4))])
    y = np.array([0] * 300 + [1] * 300)
    near = np.concatenate([rng.normal(0, 1, (50, 4)) + 3, rng.normal(5, 1, (50, 4))])
    far = rng.normal(0, 1, (100, 4)) + np.array([-20, 20, -20, 20])  # a direction no class lies in
    for scorer in (M.Mahalanobis(z, y, 2), M.KNN(z, 5)):
        assert M.auroc(scorer.score(near), scorer.score(far)) > 0.95


def test_accuracy_by_snr_bins_cover_every_window():
    logits = np.eye(3)[[0, 1, 2, 0]]
    y = np.array([0, 1, 0, 0])
    snr = np.array([-4.0, 5.0, 19.0, 20.0])
    rows = M.accuracy_by_snr(logits, y, snr)
    assert sum(r["n"] for r in rows) == 4
    assert rows[-1]["accuracy"] == 0.5
