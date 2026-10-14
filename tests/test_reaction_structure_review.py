import numpy as np
from benchmarks.offline_accuracy.paper_baseline.reaction_attention.structure_review import spectrum, structural_maps


def test_rank_detects_missing_and_repeated_channels():
    assert spectrum(np.array([[1., 1.], [0., 0.]]))['rank_fp64_threshold'] == 1
    assert spectrum(np.eye(3))['rank_fp32_threshold'] == 3
    measured = spectrum(np.array([[1.], [0.]]), np.diag([2., 3.]))
    assert measured['training_null_variance_fraction'] == .6


def test_joint_channels_can_recover_information_lost_in_one_average():
    graph = dict(reactant=np.array([[1., 1.], [1., 0.], [0., 1.]]),
                 product=np.eye(3)[:, :2], collider=np.zeros((3, 2)),
                 pooling=np.array([[.5, .5], [1., 0.], [0., 1.]]),
                 active=np.array([True, True, False]))
    result = structural_maps(graph, ['A', 'B', 'AR'])
    assert result['global_prepared_y']['rank_fp64_threshold'] == 3
    assert result['per_species'][0]['prepared_y']['rank_fp64_threshold'] == 3
    assert result['per_species'][1]['prepared_y']['rank_fp64_threshold'] == 2
    assert result['per_species'][0]['base_output']['input_dimensions'] == 2
