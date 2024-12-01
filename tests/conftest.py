import numpy as np
import pytest


@pytest.fixture(autouse=True)
def random_seed():
    np.random.seed(42)
