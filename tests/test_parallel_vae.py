import ast
from pathlib import Path
import os
import subprocess
import sys
import types

import pytest
import torch

from openvdn_comfy.parallel_vae import assemble_native, clip_plan, clip_provider, decode_parallel, pad_latents


def native_decoder():
    root = Path(__file__).resolve().parents[1]
    path = root / 'work/patch-check/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py'
    if not path.exists():
        path = root / '.deps/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py'
    if not path.exists():
        pytest.skip('Pinned Diffusers sources not installed')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AutoencoderKLMiniMaxH3')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_decode', '_blend', 'decode')]
    for n in methods:
        n.decorator_list = []
    ns = {'torch': torch, 'DecoderOutput': object}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), ns)

    class Decoder(torch.nn.Module):
        tokens_chunk_size = 5
        token_overlap = 2
        temporal_compression_ratio = 4
        spatial_compression_ratio = 2
        frame_pre_padding = 3
        frame_overlap = 5
        use_slicing = False
        config = types.SimpleNamespace(token_drop=3, clip_length=17)

        def _decode_clip(self, z):
            # Include the whole clip's context, so wrong boundaries/order or
            # missing overlap cannot accidentally pass this parity test.
            return (z[:, :3] + z.mean()).repeat_interleave(4, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)

    for name in ('_decode', '_blend', 'decode'):
        setattr(Decoder, name, ns[name])
    return Decoder()


@pytest.mark.parametrize('length', [7, 8, 12, 27, 32, 57, 72, 87, 107])
def test_sharded_clips_preserve_native_padding_blending_and_tail(length):
    vae = native_decoder()
    z = torch.arange(24 * length * 2, dtype=torch.float32).reshape(1, 24, length, 2, 1) / 1000
    expected = vae.decode(z, return_dict=False)[0]
    padding, bounds = clip_plan(vae, z.shape)
    padded = pad_latents(z, padding)
    clips = {i: vae._decode_clip(padded[:, :, a:b]) for i, (a, b) in enumerate(bounds)}
    actual, parity = assemble_native(vae, z, bounds, clips.pop, verify=True)
    assert torch.equal(actual, expected)
    assert parity == {'checked': True, 'exact': True, 'clips_checked': len(bounds)}
    assert clips == {}
    assert '_decode_clip' not in vae.__dict__
    if length == 72:
        assert len(bounds) == 14 and actual.shape[2] == 243


def test_restores_decoder_after_failure_and_rejects_drift():
    vae = native_decoder()
    z = torch.zeros(1, 24, 7, 1, 1)
    original = vae._decode_clip
    with pytest.raises(RuntimeError, match='failed'):
        with clip_provider(vae, lambda z: (_ for _ in ()).throw(RuntimeError('failed'))):
            vae.decode(z, return_dict=False)
    assert '_decode_clip' not in vae.__dict__
    vae._decode_clip = original
    with clip_provider(vae, lambda z: torch.ones(1)):
        pass
    assert vae.__dict__['_decode_clip'] is original
    _, bounds = clip_plan(vae, z.shape)
    for value in (1., float('inf'), float('nan')):
        got = original(z) + value
        _, parity = assemble_native(vae, z, bounds, lambda _: got, verify=True)
        assert not parity['exact']


def test_real_distributed_transport_with_idle_ranks_and_parity_failure(tmp_path):
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    if sys.platform == 'darwin':
        env['GLOO_SOCKET_IFNAME'] = 'lo0'
    children = []
    try:
        for rank in range(3):
            children.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                               str(tmp_path/'passed'), str(tmp_path/'store'), str(rank)],
                                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        for child in children:
            output, _ = child.communicate(timeout=45)
            assert child.returncode == 0, output[-12000:]
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    assert (tmp_path / 'passed').read_text() == 'native parity and all-rank failure propagation passed'


def distributed_worker():
    from datetime import timedelta
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(sys.argv[2]).as_uri(), rank=int(sys.argv[3]),
                            world_size=3, timeout=timedelta(seconds=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    vae = native_decoder()
    with torch.no_grad():
        for length in (7, 8, 72, 107):
            z = torch.arange(24 * length, dtype=torch.float32).reshape(1, 24, length, 1, 1) / 1000
            video, info = decode_parallel(vae, z, rank=rank, world_size=world, verify=True)
            assert info['parity']['exact']
            assert sum(r['clips'] for r in info['by_rank']) == info['temporal_clips']
            if rank == 0:
                assert torch.equal(video, vae.decode(z, return_dict=False)[0])
            else:
                assert video is None
        # Only one replica drifts. All ranks must drain communication and fail;
        # otherwise the test hangs and the process-group timeout catches it.
        if rank == 1:
            original = vae._decode_clip
            vae._decode_clip = lambda z: original(z) + 1
        try:
            decode_parallel(vae, torch.zeros(1, 24, 72, 1, 1), rank=rank, world_size=world, verify=True)
        except RuntimeError as error:
            assert 'startup parity failed' in str(error)
        else:
            raise AssertionError('Expected all-rank parity failure')
        dist.barrier()
    if rank == 0:
        Path(sys.argv[1]).write_text('native parity and all-rank failure propagation passed')
    dist.destroy_process_group()


if __name__ == '__main__':
    distributed_worker()
