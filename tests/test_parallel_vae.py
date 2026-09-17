import ast
import math
from pathlib import Path
import os
import subprocess
import sys
import types

import pytest
import torch

from openvdn_comfy.parallel_vae import assemble_native, clip_plan, clip_provider, decode_parallel, pad_latents
from openvdn_comfy.vae_tiles import ClipDecoder


def native_decoder(spatial=False):
    root = Path(__file__).resolve().parents[1]
    path = root / 'work/patch-check/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py'
    if not path.exists():
        path = root / '.deps/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py'
    if not path.exists():
        pytest.skip('Pinned Diffusers sources not installed')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AutoencoderKLMiniMaxH3')
    names = ('_decode', '_blend', 'decode')
    if spatial:
        names += ('_split_tiles', '_stitch_tiles', '_decode_clip')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in methods:
        n.decorator_list = []
    ns = {'torch': torch, 'math': math, 'DecoderOutput': object}
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

    for name in names:
        setattr(Decoder, name, ns[name])
    vae = Decoder()
    if spatial:
        vae.use_tiling = True
        vae.tile_sample_min_height = vae.tile_sample_min_width = 8
        vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 2
        vae.post_quant_conv = torch.nn.Identity()
        class Core(torch.nn.Module):
            def forward(self, z):
                return (z[:, :3] + z.mean()).repeat_interleave(4, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
        vae.decoder = Core()
    return vae


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('shape', [(3, 4), (6, 7), (8, 12)])
def test_optimized_tiles_match_native_overlap_edges_and_dtype(dtype, shape):
    vae = native_decoder(spatial=True)
    torch.manual_seed(17)
    z = torch.randn(1, 24, 7, *shape, dtype=dtype)
    original = vae._decode_clip(z)
    decode = ClipDecoder(vae)
    got = decode(z)
    assert torch.equal(got, original)
    assert torch.isfinite(got).all()
    assert decode.tile_count > 0
    assert '_blend' not in vae.__dict__ and '_stitch_tiles' not in vae.__dict__
    # Warm cached weights are reused without retaining large decoded tensors.
    weights = list(decode.weights.values())
    assert torch.equal(decode(z + .125), vae._decode_clip(z + .125))
    assert [id(w) for w in weights] == [id(w) for w in decode.weights.values()]


def test_tile_overrides_restore_when_decode_fails():
    vae = native_decoder(spatial=True)
    def broken(_):
        raise RuntimeError('tile failed')
    vae.decoder.forward = broken
    with pytest.raises(RuntimeError, match='tile failed'):
        ClipDecoder(vae)(torch.zeros(1, 24, 7, 6, 7))
    assert '_blend' not in vae.__dict__ and '_stitch_tiles' not in vae.__dict__


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
        # Exercise the real spatial splitting/stitching and temporal assembly
        # together, across real processes, against the unmodified native path.
        spatial_vae = native_decoder(spatial=True)
        z = torch.arange(24 * 12 * 6 * 7, dtype=torch.float32).reshape(1, 24, 12, 6, 7) / 1000
        video, info = decode_parallel(spatial_vae, z, rank=rank, world_size=world,
                                      verify=True, clip_decode=ClipDecoder(spatial_vae))
        assert info['parity']['exact']
        if rank == 0:
            assert torch.equal(video, spatial_vae.decode(z, return_dict=False)[0])
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
