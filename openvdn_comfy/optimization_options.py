"""Request-local optimization controls; quality experiments stay opt-in."""
FIELDS = ('fast_communication', 'attention_kernel', 'isolate_padding', 'streaming_output', 'cleanup_policy', 'linear_stats_chunk_frames', 'linear_kv_keep_ratio')


def validate(settings):
    for name in ('fast_communication', 'isolate_padding', 'streaming_output'):
        if type(getattr(settings, name)) is not bool:
            raise ValueError(f'{name} must be boolean')
    if type(settings.linear_stats_chunk_frames) is not int or settings.linear_stats_chunk_frames not in (8, 16, 32):
        raise ValueError('linear_stats_chunk_frames must be 8, 16 or 32')
    from .linear_kv import validate_ratio
    validate_ratio(settings.linear_kv_keep_ratio)
    if settings.attention_kernel not in ('native', 'decomposed'):
        raise ValueError('attention_kernel must be native or decomposed')
    if settings.attention_kernel == 'decomposed' and settings.softmax_backend == 'ref':
        raise ValueError('decomposed comparison requires a flex or decomposed resident profile')
    if settings.cleanup_policy not in ('adaptive', 'always'):
        raise ValueError('cleanup_policy must be adaptive or always')
    if settings.isolate_padding and (settings.softmax_backend != 'flex' or settings.attention_kernel != 'native'):
        raise ValueError('isolate_padding requires softmax_backend=flex and attention_kernel=native')
