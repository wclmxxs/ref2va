"""Request-local optimization controls; quality experiments stay opt-in."""
import math

FIELDS = ('fast_communication', 'attention_kernel', 'isolate_padding', 'streaming_output', 'cleanup_policy',
          'linear_stats_chunk_frames', 'sol_tau', 'sol_dense_steps', 'sol_dense_layers')


def validate(settings):
    for name in ('fast_communication', 'isolate_padding', 'streaming_output'):
        if type(getattr(settings, name)) is not bool:
            raise ValueError(f'{name} must be boolean')
    if type(settings.linear_stats_chunk_frames) is not int or settings.linear_stats_chunk_frames not in (8, 16, 32):
        raise ValueError('linear_stats_chunk_frames must be 8, 16 or 32')
    if settings.attention_kernel not in ('native', 'decomposed', 'sol'):
        raise ValueError('attention_kernel must be native, decomposed or sol')
    if (type(settings.sol_tau) not in (float, int) or not math.isfinite(settings.sol_tau)
            or not 0 <= settings.sol_tau <= 4):
        raise ValueError('sol_tau must be a finite number in [0, 4]')
    for name, maximum in (('sol_dense_steps', 8), ('sol_dense_layers', 50)):
        if type(getattr(settings, name)) is not int or not 0 <= getattr(settings, name) <= maximum:
            raise ValueError(f'{name} must be an integer in [0, {maximum}]')
    if settings.attention_kernel == 'sol' and (not settings.inference_kernels or settings.softmax_backend == 'ref'):
        raise ValueError('Sol requires inference_kernels=true and softmax_backend=flex or decomposed')
    if settings.attention_kernel == 'decomposed' and settings.softmax_backend == 'ref':
        raise ValueError('decomposed comparison requires a flex or decomposed resident profile')
    if settings.cleanup_policy not in ('adaptive', 'always'):
        raise ValueError('cleanup_policy must be adaptive or always')
    if settings.isolate_padding and (settings.softmax_backend != 'flex' or settings.attention_kernel not in ('native', 'sol')):
        raise ValueError('isolate_padding requires softmax_backend=flex and attention_kernel=native or sol')
