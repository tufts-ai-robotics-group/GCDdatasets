import torch

from gcd_data.data_utils import WeightedConsistentSampler


def test_sampler():
    sampler = WeightedConsistentSampler(([1] * 100) + ([2] * 100), 200)
    samples = torch.Tensor([i for i in sampler])
    for i in range(2):
        assert torch.allclose(samples, torch.Tensor([i for i in sampler]))
