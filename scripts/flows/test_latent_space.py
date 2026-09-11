import numpy as np

from latent_space import ranking_quality


def _q(components, volumes, n_top=2):
    return ranking_quality(np.asarray(components, dtype=float),
                           np.asarray(volumes, dtype=float), n_top=n_top)


def test_dimensions_topped_by_the_same_targets_are_not_distinct():
    same = [[9.0, 9.0, 0.1, 0.1], [9.0, 9.0, 0.1, 0.1]]
    assert _q(same, [1, 2, 3, 4])['unweighted/distinctness'] == 0.0


def test_dimensions_topped_by_disjoint_targets_are_fully_distinct():
    apart = [[9.0, 9.0, 0.1, 0.1], [0.1, 0.1, 9.0, 9.0]]
    assert _q(apart, [1, 2, 3, 4])['unweighted/distinctness'] == 1.0


def test_prevalence_is_the_volume_percentile_of_the_targets_that_top_a_dim():
    # loadings pick the two rarest targets, so they sit at percentile 0 and 1/3
    rare = [[9.0, 9.0, 0.1, 0.1], [9.0, 9.0, 0.1, 0.1]]
    got = _q(rare, [1, 2, 3, 4])['unweighted/prevalence']
    assert got == np.mean([0.0, 1 / 3])


def test_volume_weighting_lifts_prevalence_when_the_loadings_favour_rare_targets():
    # dimension 1's largest loading is on the rarest target; a common one is
    # close enough behind that sqrt(volume) reorders the pair
    components = [[9.0, 8.0, 0.1, 0.1], [0.1, 0.1, 9.0, 8.0]]
    volumes = [1, 1000, 1, 1000]
    q = _q(components, volumes, n_top=1)
    assert q['by_volume/prevalence'] > q['unweighted/prevalence']


def test_score_is_zero_when_either_axis_collapses():
    same = [[9.0, 9.0, 0.1, 0.1], [9.0, 9.0, 0.1, 0.1]]
    assert _q(same, [1, 2, 3, 4])['unweighted/score'] == 0.0


def test_a_single_dimension_has_nothing_to_be_distinct_from():
    assert _q([[9.0, 9.0, 0.1, 0.1]], [1, 2, 3, 4])['unweighted/distinctness'] == 1.0


def test_no_volumes_means_no_metrics():
    assert ranking_quality(np.ones((2, 4)), None) == {}
