import pytest
from src.example import sample_sum


def test_sample_sum_basic():
    """Test basic functionality with a simple list."""
    assert sample_sum([1, 2, 3, 4, 5]) == 15


def test_sample_sum_single_element():
    """Test with a single element."""
    assert sample_sum([5]) == 5


def test_sample_sum_empty_list():
    """Test with an empty list."""
    assert sample_sum([]) == 0


def test_sample_sum_negative_numbers():
    """Test with negative numbers."""
    assert sample_sum([-1, -2, -3]) == -6


def test_sample_sum_mixed_numbers():
    """Test with mixed positive and negative numbers."""
    assert sample_sum([-5, 10, -3, 2]) == 4


def test_sample_sum_zeros():
    """Test with zeros."""
    assert sample_sum([0, 0, 0]) == 0


def test_sample_sum_large_numbers():
    """Test with larger numbers."""
    assert sample_sum([100, 200, 300]) == 600


def test_sample_sum_floats():
    """Test with floating point numbers."""
    assert sample_sum([1.5, 2.5, 3.0]) == 7.0

