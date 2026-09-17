"""Reset per-request Ulysses geometry without rebuilding weights or NCCL groups."""
class GeometryCache:
    def __init__(self, reset_compiler, max_shapes=8):
        self.shapes = set()
        self.reset_compiler = reset_compiler
        self.max_shapes = max_shapes

    def prepare(self, runtime, plan, embeds, tags, conditions):
        # Official configure() guards one geometry per process. Between completed
        # requests all collectives have finished; the next forward recomputes every
        # split/offset/head assignment. Keep the communicators and rank layout.
        runtime.sequence_length = 0
        signature = (plan.generation_width, plan.generation_height, plan.sampling_frames,
                     tuple(embeds.shape), tuple(tags.tolist()),
                     tuple(conditions[0]) if conditions else (),
                     tuple(tuple(value.shape) for value in conditions[1]) if conditions else ())
        new_shape = signature not in self.shapes
        if new_shape and len(self.shapes) >= self.max_shapes:
            # Static Flex has a finite Dynamo recompile budget. Reset compiled graph
            # bookkeeping before reaching it; on-disk kernels and models are retained.
            self.reset_compiler()
            self.shapes.clear()
        self.shapes.add(signature)
        return new_shape
