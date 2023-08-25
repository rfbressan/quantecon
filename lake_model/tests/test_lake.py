"""Tests for the lake model."""

import numpy as np
import pytest

from lake_model.lake import LakeModel


@pytest.fixture
def lake():
    return LakeModel()


def test_long_term_state(lake):
    """Test the long_term_state method"""
    u_bar = (1 + lake.g - (1 - lake.d) * (1 - lake.α)) / (
        1 + lake.g - (1 - lake.d) * (1 - lake.α) + (1 - lake.d) * lake.λ
    )
    e_bar = 1 - u_bar
    assert np.allclose(lake.long_term_state(), np.asarray([u_bar, e_bar]))


def test_simulate_path_lt(lake):
    """Test the simulate_path method for wrong dimension of x0"""
    x0 = [10, 10, 10]  # wrong dimension
    with pytest.raises(ValueError):
        lake.simulate_path(x0)
