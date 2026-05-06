"""Property tests for domain assignment (Properties 5, 6).

Property 5: Domain Partition Validity
Property 6: Domain Assignment Immutability

Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from code.domains.assignment import DomainAssignment
from code.domains.config import DomainConfig


# --- Custom strategies ---

def layer_configs():
    """Generate valid layer size configurations."""
    return st.tuples(
        st.integers(min_value=4, max_value=256),  # in_features
        st.integers(min_value=4, max_value=256),  # out_features
    )


def domain_sizes():
    """Generate valid domain sizes."""
    return st.integers(min_value=2, max_value=64)


# --- Property 5: Domain Partition Validity ---

@pytest.mark.property
class TestDomainPartitionValidity:
    """Property 5: Domain Partition Validity.

    **Validates: Requirements 4.1, 4.2, 4.3**

    For any layer with out_features neurons and domain_size configuration
    (in either "spatial" or "random" mode), the domain assignment shall
    produce exactly ceil(out_features / domain_size) domains where every
    neuron belongs to exactly one domain (no overlaps, no unassigned neurons).
    """

    @given(
        out_features=st.integers(min_value=4, max_value=256),
        domain_size=st.integers(min_value=2, max_value=64),
        mode=st.sampled_from(["spatial", "random"]),
    )
    @settings(max_examples=200, deadline=None)
    def test_correct_number_of_domains(self, out_features, domain_size, mode):
        """Produces exactly ceil(out_features / domain_size) domains."""
        config = DomainConfig(domain_size=domain_size, mode=mode)
        layer_sizes = [(out_features, out_features)]  # square for simplicity

        # Generate a weight matrix for spatial mode
        weight = torch.randn(out_features, out_features)
        assignment = DomainAssignment(
            layer_sizes=layer_sizes,
            config=config,
            weight_matrices=[weight],
        )

        expected_n_domains = math.ceil(out_features / domain_size)
        actual_n_domains = assignment.n_domains_per_layer[0]
        assert actual_n_domains == expected_n_domains

    @given(
        out_features=st.integers(min_value=4, max_value=256),
        domain_size=st.integers(min_value=2, max_value=64),
        mode=st.sampled_from(["spatial", "random"]),
    )
    @settings(max_examples=200, deadline=None)
    def test_no_overlaps_no_unassigned(self, out_features, domain_size, mode):
        """Every neuron belongs to exactly one domain."""
        config = DomainConfig(domain_size=domain_size, mode=mode)
        layer_sizes = [(out_features, out_features)]

        weight = torch.randn(out_features, out_features)
        assignment = DomainAssignment(
            layer_sizes=layer_sizes,
            config=config,
            weight_matrices=[weight],
        )

        domains = assignment.get_domain_indices(0)

        # Collect all assigned neurons
        all_neurons = []
        for domain in domains:
            all_neurons.extend(domain)

        # No duplicates (no overlaps)
        assert len(all_neurons) == len(set(all_neurons))

        # All neurons assigned
        assert set(all_neurons) == set(range(out_features))

    @given(
        out_features=st.integers(min_value=4, max_value=256),
        domain_size=st.integers(min_value=2, max_value=64),
        mode=st.sampled_from(["spatial", "random"]),
    )
    @settings(max_examples=200, deadline=None)
    def test_neuron_to_domain_consistent(self, out_features, domain_size, mode):
        """get_neuron_to_domain is consistent with get_domain_indices."""
        config = DomainConfig(domain_size=domain_size, mode=mode)
        layer_sizes = [(out_features, out_features)]

        weight = torch.randn(out_features, out_features)
        assignment = DomainAssignment(
            layer_sizes=layer_sizes,
            config=config,
            weight_matrices=[weight],
        )

        domains = assignment.get_domain_indices(0)
        n2d = assignment.get_neuron_to_domain(0)

        for domain_idx, indices in enumerate(domains):
            for neuron_idx in indices:
                assert n2d[neuron_idx].item() == domain_idx


# --- Property 6: Domain Assignment Immutability ---

@pytest.mark.property
class TestDomainAssignmentImmutability:
    """Property 6: Domain Assignment Immutability.

    **Validates: Requirements 4.4**

    For any DomainAssignment instance, calling get_domain_indices(layer_index)
    multiple times shall return identical results.
    """

    @given(
        out_features=st.integers(min_value=4, max_value=128),
        domain_size=st.integers(min_value=2, max_value=32),
        n_calls=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=200, deadline=None)
    def test_repeated_calls_identical(self, out_features, domain_size, n_calls):
        """Multiple calls to get_domain_indices return identical results."""
        config = DomainConfig(domain_size=domain_size, mode="spatial")
        layer_sizes = [(out_features, out_features)]

        weight = torch.randn(out_features, out_features)
        assignment = DomainAssignment(
            layer_sizes=layer_sizes,
            config=config,
            weight_matrices=[weight],
        )

        first_result = assignment.get_domain_indices(0)
        for _ in range(n_calls - 1):
            result = assignment.get_domain_indices(0)
            assert result == first_result


# --- Unit tests ---

class TestDomainAssignmentUnit:
    """Unit tests for DomainAssignment."""

    def test_128_neurons_16_domain_size(self):
        """128 neurons with domain_size=16 → 8 domains."""
        config = DomainConfig(domain_size=16, mode="spatial")
        layer_sizes = [(784, 128)]
        weight = torch.randn(128, 784)

        assignment = DomainAssignment(
            layer_sizes=layer_sizes,
            config=config,
            weight_matrices=[weight],
        )

        assert assignment.n_domains_per_layer[0] == 8
        assert assignment.total_domains == 8

        # All neurons assigned
        domains = assignment.get_domain_indices(0)
        all_neurons = [n for d in domains for n in d]
        assert len(all_neurons) == 128
        assert len(set(all_neurons)) == 128

    def test_multi_layer(self):
        """Multiple layers each get their own domain assignment."""
        config = DomainConfig(domain_size=16, mode="spatial")
        layer_sizes = [(784, 128), (128, 128), (128, 128), (128, 128), (128, 10)]
        weights = [torch.randn(out, inp) for inp, out in layer_sizes]

        assignment = DomainAssignment(
            layer_sizes=layer_sizes,
            config=config,
            weight_matrices=weights,
        )

        # 128/16 = 8 domains for hidden layers, ceil(10/16) = 1 for output
        assert assignment.n_domains_per_layer == [8, 8, 8, 8, 1]
        assert assignment.total_domains == 33

    def test_random_mode_different_from_spatial(self):
        """Random mode produces different assignment than spatial."""
        layer_sizes = [(128, 128)]
        weight = torch.randn(128, 128)

        spatial = DomainAssignment(
            layer_sizes=layer_sizes,
            config=DomainConfig(domain_size=16, mode="spatial"),
            weight_matrices=[weight],
        )
        random = DomainAssignment(
            layer_sizes=layer_sizes,
            config=DomainConfig(domain_size=16, mode="random"),
            weight_matrices=[weight],
        )

        # They should have the same number of domains
        assert spatial.n_domains_per_layer == random.n_domains_per_layer

        # But different neuron assignments (with high probability)
        spatial_domains = spatial.get_domain_indices(0)
        random_domains = random.get_domain_indices(0)
        # At least one domain should differ
        assert spatial_domains != random_domains

    def test_domain_distances_shape(self):
        """Domain distances have correct shape."""
        config = DomainConfig(domain_size=16, mode="spatial")
        layer_sizes = [(784, 128)]
        weight = torch.randn(128, 784)

        assignment = DomainAssignment(
            layer_sizes=layer_sizes,
            config=config,
            weight_matrices=[weight],
        )

        distances = assignment.get_domain_distances(0)
        assert distances.shape == (8, 8)
        # Diagonal should be zero
        assert (distances.diag() == 0).all()
        # Symmetric
        assert torch.allclose(distances, distances.T)
