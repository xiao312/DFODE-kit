import numpy as np

from dfode_kit.physics.atom_conservation import completion_matrix, mass_fraction_element_matrix


def test_completion_matrix_has_identity_key_rows_and_conserves_atoms():
    # Elements x species for H2, O2, H2O.
    atom_matrix = np.array(
        [
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 1.0],
        ]
    )

    completion = completion_matrix(atom_matrix)

    assert completion.matrix.shape == (3, 1)
    assert np.allclose(atom_matrix @ completion.matrix, 0.0)
    assert len(completion.key_species_indices) == 1
    key_idx = completion.key_species_indices[0]
    assert completion.matrix[key_idx, 0] == 1.0


def test_mass_fraction_element_matrix_divides_by_molecular_weight():
    class Gas:
        n_elements = 2
        n_species = 2
        molecular_weights = np.array([2.0, 18.0])

        def n_atoms(self, species_idx, elem_idx):
            return np.array([[2.0, 0.0], [2.0, 1.0]])[species_idx, elem_idx]

    matrix = mass_fraction_element_matrix(Gas())

    assert np.allclose(matrix, [[1.0, 1.0 / 9.0], [0.0, 1.0 / 18.0]])
