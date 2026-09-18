from pathlib import Path
import subprocess

import pytest

from scripts.install_sol import CUTLASS_VERSION, PROTECTED, SPLIT_PACKAGES, install


def existing(cute='4.7.1'):
    return {'torch': '2.13.0+cu129', 'torchvision': '0.28.0+cu129', 'triton': '3.7.1',
            'flash-attn-4': '4.0.0b26', 'quack-kernels': '0.5.3',
            'nvidia-cutlass-dsl': cute, 'nvidia-cutlass-dsl-libs-base': cute}


def test_repair_resolves_full_stack_before_removing_split_libraries():
    state = {**existing(), **dict.fromkeys(SPLIT_PACKAGES, '4.7.1')}
    commands = []
    install(state, lambda cmd, **kwargs: commands.append(cmd), root=Path('/test repo'), python='/test/python')
    dry, remove, apply, check, imports = commands
    assert '--dry-run' in dry and '--dry-run' not in apply
    assert remove[1:3] == ['pip', 'uninstall']
    assert remove[5:] == list(SPLIT_PACKAGES)
    assert apply == dry[:-1]
    for name in PROTECTED:
        assert f'{name}=={state[name]}' in dry
    assert '--prerelease=allow' in dry
    assert '--reinstall-package' in apply
    assert check[1:3] == ['pip', 'check']
    assert 'import flash_attn.cute.interface; import quack;' in imports[2]
    assert 'SolKernel.dependencies()' in imports[2]


def test_failed_preflight_never_mutates_environment():
    commands = []
    def fail(cmd, **kwargs):
        commands.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)
    with pytest.raises(subprocess.CalledProcessError):
        install(existing(), fail)
    assert len(commands) == 1 and '--dry-run' in commands[0]


def test_healthy_environment_is_not_reinstalled_and_unknown_stack_is_not_replaced():
    commands = []
    install(existing(CUTLASS_VERSION), lambda cmd, **kw: commands.append(cmd))
    assert len(commands) == 4
    assert all('uninstall' not in cmd and '--reinstall-package' not in cmd for cmd in commands)
    state = existing()
    state['flash-attn-4'] = '4.0.0b31'
    with pytest.raises(RuntimeError, match='refusing'):
        install(state, lambda *a, **kw: pytest.fail('Must not mutate unsupported stack'))


def test_sol_and_vdn_share_exact_cutlass_pin():
    root = Path(__file__).resolve().parents[1]
    pin = f'nvidia-cutlass-dsl=={CUTLASS_VERSION}'
    assert pin in (root/'requirements-sol.txt').read_text().splitlines()
    assert pin in (root/'constraints-vdn.txt').read_text().splitlines()


def test_sol_converter_preserves_new_keywords_and_restores_native_on_error():
    import ast
    from collections import namedtuple
    from contextlib import contextmanager
    from types import SimpleNamespace
    from typing import Generic, TypeVar, get_origin
    root = Path(__file__).resolve().parents[1]
    source = root/'openvdn_comfy/_vendor/sol_attn/sm90/_compat/cute_dsl_utils.py'
    parsed = ast.parse(source.read_text())
    names = ('_converter_wrapper', 'converter_compatibility')
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    # Execute the actual converter wrapper without importing the Linux/CUDA DSL.
    def original(arg, name, typ, ctx, *, is_constexpr=False):
        return typ, is_constexpr
    T = TypeVar('T')
    class Constexpr(Generic[T]):
        pass
    converter = SimpleNamespace(_convert_single_arg=original)
    namespace = dict(contextmanager=contextmanager, get_origin=get_origin,
                     cutlass=SimpleNamespace(Constexpr=Constexpr), _converter_module=converter,
                     spec=SimpleNamespace(ConstNone=lambda name: ('none', name)))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    Point = namedtuple('Point', 'value')
    with pytest.raises(RuntimeError, match='compile failed'):
        with namespace['converter_compatibility']():
            assert converter._convert_single_arg(1, 'n', int, None, is_constexpr=True) == (int, True)
            assert converter._convert_single_arg(Point(1), 'p', tuple, None) == (Point, False)
            assert converter._convert_single_arg(None, 'c', Constexpr[int], None) == ('none', 'c')
            raise RuntimeError('compile failed')
    assert converter._convert_single_arg is original
    # Importing Sol must not globally replace the converter before the context.
    assert not any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
                   and target.attr == '_convert_single_arg' for target in node.targets) for node in parsed.body)


@pytest.mark.parametrize('cuda_version', [(12, 9), (13, 0)])
@pytest.mark.parametrize('third', [None, -2., 8.])
def test_sol_fmax_uses_pinned_nvvm_signature_independent_of_cuda(monkeypatch, cuda_version, third):
    import ast
    import sys
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    source = root/'openvdn_comfy/_vendor/sol_attn/_vendor/flash_attn/cute/utils.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'fmax')
    # Exercise the real wrapper against CuTe 4.6.0.dev0's generated NVVM API.
    # No native CUDA/MLIR libraries are available in the macOS test environment.
    calls = []
    class Float32:
        def __init__(self, value):
            self.value = value.value if isinstance(value, Float32) else value

        def ir_value(self, *, loc=None, ip=None):
            return self.value

    def nvvm_fmax(a, b, *, c=None, ftz=None, nan=None, abs=None, results=None, loc=None, ip=None):
        calls.append((a, b, c, ftz, nan, abs, results, loc, ip))
        return max(a, b) if c is None else max(a, b, c)

    monkeypatch.setitem(sys.modules, 'cutlass', SimpleNamespace(
        CUDA_VERSION=SimpleNamespace(major=cuda_version[0], minor=cuda_version[1])))
    namespace = dict(Float32=Float32, nvvm=SimpleNamespace(fmax=nvvm_fmax),
                     T=SimpleNamespace(f32=lambda: 'f32'), dsl_user_op=lambda fn: fn)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    result = namespace['fmax'](Float32(-4.), 3., third, loc='location', ip='insertion')
    assert result.value == (8. if third == 8. else 3.)
    assert calls == [(-4., 3., third, None, None, None, None, 'location', 'insertion')]
