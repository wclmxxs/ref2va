import json

import pytest
import torch

from scripts.diagnose_sol_attention import diagnose_case, error_metrics, reference_preprocess, route_details
from openvdn_comfy.sol_reference import sol_reference


def test_diagnostics_keep_error_limits_and_localize_bad_head_and_tail():
    expected = torch.ones((1, 129, 4, 2))
    actual = expected.clone()
    actual[:, 128, 2] += 2
    report = error_metrics(actual, expected)
    assert not report['passed'] and report['max_abs'] == 2
    assert report['worst_index_bthd'] == [0, 128, 2, 0]
    assert report['relative_l2_by_query_block'][:2] == [0, 0]
    assert report['relative_l2_by_head'][:2] == [0, 0]
    actual.fill_(float('nan'))
    assert error_metrics(actual, expected) == {'passed': False, 'finite': False}


def test_diagnostic_reference_keeps_exact_neighbors_without_threshold_pass():
    q = torch.zeros((2, 129, 3, 128), dtype=torch.bfloat16)
    k = torch.zeros((2, 517, 3, 128), dtype=torch.bfloat16)
    kc, vc, threshold = reference_preprocess(q, k, k, 128**-.5, 1.)
    assert kc.shape == vc.shape == (2, 9, 3, 128)
    torch.testing.assert_close(threshold, torch.full((2, 3, 3), .001))
    margin, route = route_details(q, kc, threshold, 128**-.5, 65)
    assert (margin < 0).all()
    for i, selected in enumerate(({0, 1}, {0, 1, 2}, {0, 1, 2, 3})):
        assert set(route[0, i, 0].nonzero().flatten().tolist()) == selected


@pytest.mark.parametrize('mismatch', [False, True])
def test_diagnostic_captures_numeric_failure_without_hiding_exact_success(tmp_path, monkeypatch, mismatch):
    from openvdn_comfy import sol_kernel
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(sol_kernel, 'prepare_rectangular',
                        lambda q, k, v, *, scale, tau: reference_preprocess(q, k, v, scale, tau))
    class Kernel:
        def __call__(self, q, k, v, **kwargs):
            result = sol_reference(q, k, v, **kwargs)
            if mismatch and not kwargs.get('all_exact'):
                result[:, :, 1] += .1
            return result

        def snapshot(self):
            return {}

        def since(self, before):
            return {'compile_misses': 0}

    rng = torch.Generator().manual_seed(8)
    q = torch.randn((1, 3, 2, 128), generator=rng).to(torch.bfloat16)
    k, v = [torch.randn((1, 70, 2, 128), generator=rng).to(torch.bfloat16) for _ in range(2)]
    report = diagnose_case(Kernel(), q, k, v, tmp_path, 2)
    assert report['passed'] is not mismatch
    assert report['all_exact']['passed'] and report['repeat_exact']
    assert report['reference_route_differences_after_preprocess'] == 0
    for metric in report['preprocess'].values():
        assert metric['finite'] and metric['max_abs'] == 0
        assert 'passed' not in metric  # Block sums do not use output tolerances.
    assert json.loads((tmp_path/'heads-2.json').read_text()) == report
    if mismatch:
        bundle = torch.load(tmp_path/report['sample_file'], weights_only=True)
        assert bundle['parameters'] == dict(scale=128**-.5, tau=1., sink_tokens=65)
        torch.testing.assert_close(bundle['tensors']['q'], q, rtol=0, atol=0)
        assert not report['sparse_vs_cpu_reference']['passed']
        assert report['gpu_vs_cpu_reference']['passed']
    else:
        assert not list(tmp_path.glob('*.pt'))
