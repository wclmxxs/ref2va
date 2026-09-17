"""Bound static graph specializations without discarding the working set every 8 jobs."""
import hashlib
import json


class GeometryCache:
    def __init__(self, reset_compiler, max_shapes=32):
        self.shapes = set()
        self.reset_compiler = reset_compiler
        self.max_shapes = max_shapes
        self.generation = 0
        self.pending = None
        self.last = {}

    def prepare(self, runtime, plan, embeds, tags, conditions):
        # Official configure() guards one geometry per process. Between completed
        # requests all collectives have finished; the next forward recomputes every
        # split/offset/head assignment. Keep the communicators and rank layout.
        runtime.sequence_length = 0
        signature = (getattr(runtime, "softmax_ranks", 6), plan.generation_width, plan.generation_height, plan.sampling_frames,
                     tuple(embeds.shape), tuple(tags.tolist()),
                     tuple(conditions[0]) if conditions else (),
                     tuple(tuple(value.shape) for value in conditions[1]) if conditions else ())
        fingerprint = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        new_shape = fingerprint not in self.shapes
        reset = new_shape and len(self.shapes) >= self.max_shapes
        if reset:
            # Static Flex has a finite Dynamo recompile budget. Reset compiled graph
            # bookkeeping before reaching it; on-disk kernels and models are retained.
            self.reset_compiler()
            self.shapes.clear()
            self.generation += 1
        self.pending = fingerprint
        self.last = {"geometry_id": fingerprint, "geometry_seen": not new_shape,
                     "reset": reset, "generation": self.generation,
                     "capacity": self.max_shapes, "successful_geometries": len(self.shapes),
                     "reset_reason": "geometry_capacity" if reset else None}
        return new_shape

    def commit(self):
        # A failed sample must not advertise that its geometry is warm.
        if self.pending is not None:
            self.shapes.add(self.pending)
            self.last["successful_geometries"] = len(self.shapes)
            self.pending = None
