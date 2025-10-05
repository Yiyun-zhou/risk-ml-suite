import numpy as np
from src.utils import ks_stat, psi

def test_ks_stat_nonnegative():
    y = np.array([0,0,1,1,0,1])
    s = np.array([0.1,0.2,0.8,0.7,0.3,0.9])
    ks = ks_stat(y, s)
    assert 0 <= ks <= 1

def test_psi_symmetry():
    a = np.random.randn(1000)
    b = a.copy()
    assert abs(psi(a, b)) < 0.01
