"""Shared raw-QKV Ulysses with overlapped full softmax/linear branches.

Inspired by SGLang's VDN transport. This adapter retains OpenVDN layouts,
reference rows, gates, separate FP8 output projections and uneven row shards.
"""
from contextlib import contextmanager, nullcontext


def dual_forward(attn, hidden_states, rotary_emb=None, attention_mask=None):
    return attn._ref2va_dual.forward(attn, hidden_states, rotary_emb, attention_mask)


class DualStream:
    def __init__(self, runtime, dit_runtime, native_forward):
        self.runtime, self.dit_runtime, self.native_forward = runtime, dit_runtime, native_forward
        self.stream, self.buffers, self.checked = None, {}, set()
        self.verification = {'checked': False}
        self.active = False
        for attn in dit_runtime.hybrids:
            attn._ref2va_dual = self
        dit_runtime.forwards['_dual_stream_attention_forward'] = dual_forward

    def stage(self, name):
        profiler = getattr(self.runtime, '_ref2va_fine', None)
        return profiler.stage(name) if self.runtime.profile_enabled and profiler else nullcontext()

    def buffer(self, role, shape, like):
        key = (role, tuple(shape), like.dtype, like.device)
        if key not in self.buffers:
            self.buffers[key] = like.new_empty(shape)
        return self.buffers[key]

    def to_heads(self, tensor, role):
        import torch.distributed as dist
        r = self.runtime
        rows, heads, width = tensor.shape
        local = heads//r.world_size
        send = self.buffer(role+'_send', (r.world_size,rows,local,width), tensor)
        with self.stage('branch_pack'):
            send.copy_(tensor.view(rows,r.world_size,local,width).permute(1,0,2,3))
        recv = self.buffer(role+'_recv', (r.sequence_length,local,width), tensor)
        work = dist.all_to_all_single(recv.view(-1), send.view(-1),
            output_split_sizes=[s*local*width for s in r.splits],
            input_split_sizes=[rows*local*width]*r.world_size,
            group=r.softmax_dispatch_group, async_op=True)
        return recv, work

    def to_rows(self, tensor, role, group):
        import torch.distributed as dist
        r = self.runtime
        _, heads, width = tensor.shape
        rows = r.splits[r.rank]
        recv = self.buffer(role+'_recv', (r.world_size,rows,heads,width), tensor)
        send = tensor.contiguous()
        work = dist.all_to_all_single(recv.view(-1), send.view(-1),
            input_split_sizes=[s*heads*width for s in r.splits],
            output_split_sizes=[rows*heads*width]*r.world_size, group=group, async_op=True)
        return recv, work, send

    @contextmanager
    def request(self, enabled):
        import torch
        if self.active:
            raise RuntimeError('Dual-stream requests cannot overlap')
        self.active = enabled
        self.buffers = {}
        if enabled and self.stream is None:
            self.stream = torch.cuda.Stream(device=self.runtime.device, priority=-1)
        try:
            yield self
        finally:
            # Request exits only after sampler synchronize (or worker failure).
            if self.stream is not None and enabled:
                torch.cuda.current_stream(self.runtime.device).wait_stream(self.stream)
            self.buffers.clear()
            self.active = False
            self.runtime._ref2va_full_rotary = None

    def report(self, requested):
        return {'rank': self.runtime.rank, 'enabled': requested, 'qkv_exchange': 'raw_shared' if requested else 'native',
                'branch_streams': 2 if requested else 1, 'verification': self.verification}

    def forward(self, attn, hidden_states, rotary_emb, attention_mask):
        import torch
        import torch.distributed as dist
        if not self.active or attention_mask is not None or hidden_states.shape[0] != 1 or attn.teacher_mode:
            raise RuntimeError('Invalid dual-stream inference context')
        # Verify the first block of each shape/layout against the native Ulysses
        # path on the actual request. Every rank participates before continuing.
        key = (tuple(hidden_states.shape), attn.layout.seq_len, attn.layout.video_start,
               attn.layout.num_frames, attn.anchor_frames,
               getattr(self.runtime, '_ref2va_attention_signature', ()))
        check = key not in self.checked
        reference = self.native_forward(attn, hidden_states, rotary_emb, attention_mask) if check else None
        result = self.compute(attn, hidden_states)
        if check:
            error = (result.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8)
            verdict = (torch.isfinite(result).all() & (error <= .01)).to(torch.int32)
            dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
            if not bool(verdict):
                raise RuntimeError(f'Dual-stream parity failed, local relative error={float(error)}')
            self.checked.add(key)
            if len(self.checked)>64:
                self.checked = {key}
            self.verification = {'checked':True, 'passed':True, 'relative_error':float(error),
                                 'all_rank_agreement':True, 'scope':'first block per request geometry'}
        return result

    def compute(self, attn, hidden_states):
        import torch
        import torch.nn.functional as F
        from src.models.ops.fp8_linear import Fp8Linear, quantize_activation
        from src.models.softmax_attention.kernels import _qk_prep, apply_softmax_gate
        from src.inference.utils.ulysses import _linear_branch_shard
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from .attention_runtime import window_attention, isolated
        from .exact_runtime import project_video_rows
        r, layout, x = self.runtime, attn.layout, hidden_states[0]
        if r.branch_parallel or r.num_heads % r.world_size or attn.linear_attention.head_dim != attn.head_dim:
            raise RuntimeError('dual_stream requires softmax_ranks=0 and divisible heads')
        full_rope = getattr(r, '_ref2va_full_rotary', None)
        if full_rope is None:
            raise RuntimeError('Missing full-sequence RoPE for dual-stream attention')
        projections = (attn.orig.to_q, attn.orig.to_k, attn.orig.to_v)
        with self.stage('qkv'):
            if all(isinstance(p, Fp8Linear) for p in projections):
                quantized = quantize_activation(x)
                raw = [p.forward_quantized(*quantized,out_dtype=x.dtype) for p in projections]
            else:
                raw = [p(x) for p in projections]
            raw = [t.unflatten(-1,(attn.num_heads,-1)) for t in raw]
        with self.stage('gates'):
            sm_gate = attn.softmax_gate(x) if attn.enable_softmax_gate else x.new_ones(len(x),attn.num_heads,1)
            beta = torch.sigmoid(attn.linear_attention.beta_proj(x)).unsqueeze(-1)
            gate = attn.linear_attention.output_gate
            hidden = gate.down(x) if gate.down is not None else x
        # One QKV payload is shared by both branches. Gate's low-rank hidden is
        # gathered separately; never send the full per-channel output gate.
        packed, exchange = self.to_heads(torch.cat([*raw,sm_gate,beta],dim=-1),'qkv')
        with self.stage('frame_mean_launch'):
            sums, sums_work = r.video_frame_mean_async(x,layout)
        with self.stage('linear_gate_hidden_gather'):
            hidden_full = r.gather_sequence(hidden)
        with self.stage('branch_relevant_wait'):
            exchange.wait()
        d, h = attn.head_dim, r.heads_per_rank
        q, k, v, sm_gate, beta = packed.split((d,d,d,1,1),dim=-1)
        first, last = r.rank*h, (r.rank+1)*h
        with self.stage('linear_gate_up'):
            b = None if gate.up.bias is None else gate.up.bias[first*d:last*d]
            linear_gate = torch.sigmoid(F.linear(hidden_full,gate.up.weight[first*d:last*d],b)).view(layout.seq_len,h,d)
        bounds = attn._bounds(layout)
        full_cover = all(lo<=0 and hi>=layout.num_frames-1 for lo,hi in bounds)
        linear_active = not full_cover and attn.linear_attention_enabled
        main = torch.cuda.current_stream(x.device)
        self.stream.wait_stream(main)  # dependencies precede softmax work
        with self.stage('softmax_compute'):
            qsm = _qk_prep(q,attn.orig.norm_q.weight,attn.orig.norm_q.eps,*full_rope)
            ksm = _qk_prep(k,attn.orig.norm_k.weight,attn.orig.norm_k.eps,*full_rope)
            if full_cover and not isolated(attn):
                sm = dispatch_attention_fn(qsm[None],ksm[None],v[None],attn_mask=None,dropout_p=0.,is_causal=False,
                    backend=getattr(type(attn.orig.processor),'_attention_backend',None)).squeeze(0)
            elif attn.softmax_impl == 'ref':
                from src.models.softmax_attention import window_softmax_reference
                sm = window_softmax_reference(qsm, ksm, v, layout, bounds, d**-.5,
                                               anchor_frames=attn.anchor_frames)
            else:
                sm = window_attention(attn,qsm,ksm,v,layout,bounds,d**-.5)
            sm = apply_softmax_gate(sm,sm_gate,inference=attn.inference_mode).view(layout.seq_len,h,d)
        with self.stage('softmax_return_launch'):
            soft_recv, soft_work, soft_send = self.to_rows(sm,'soft',r.softmax_dispatch_group)
        linear_recv = None
        with torch.cuda.stream(self.stream):
            with self.stage('frame_mean_wait'):
                sums_work.wait()
            if linear_active:
                with self.stage('linear_compute'):
                    linear = _linear_branch_shard(attn,(q,k,v),beta.squeeze(-1),linear_gate,
                        sums/layout.tokens_per_frame,first,last)
                with self.stage('linear_return_launch'):
                    linear_recv, linear_work, linear_send = self.to_rows(linear,'linear',r.linear_dispatch_group)
                    linear_work.wait()
        with self.stage('output_stream_join'):
            main.wait_stream(self.stream)
            soft_work.wait()
        with self.stage('output_unpack'):
            soft_local = soft_recv.permute(1,0,2,3).contiguous().view(len(x),attn.num_heads,d)
            if linear_recv is not None:
                linear_recv.record_stream(main)
                linear_local = linear_recv.permute(1,0,2,3).contiguous().view(len(x),attn.num_heads,d)
        with self.stage('output_projection'):
            out = attn.orig.to_out[1](attn.orig.to_out[0](soft_local.reshape(len(x),-1).type_as(x)))
            if linear_active:
                out = project_video_rows(attn,out,linear_local,r,layout,x)
        return out.unsqueeze(0)
