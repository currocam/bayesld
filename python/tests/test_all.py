import pytest
import bayesld


def test_sum_as_string():
    assert bayesld.sum_as_string(1, 1) == "2"
