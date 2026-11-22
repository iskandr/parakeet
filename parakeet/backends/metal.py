"""Apple GPU backend: parallel loops become Metal compute kernels.

The host program is the ordinary C module; each eligible parallel loop also
gets a Metal Shading Language kernel.  At run time the C code asks this module
(through a callback) to run the loop on the GPU; loops that cannot be expressed
in Metal - most importantly anything touching ``float64``, which Apple GPUs do
not support - transparently run on the CPU thread pool instead.
"""

from __future__ import annotations

import ctypes
import sys
import threading

from .. import ir
from ..config import config
from ..errors import BackendUnavailable, InternalError
from ..types import BufType, ScalarType
from . import runtime_c
from .cfamily import CFamilyEmitter, Unsupported, defined_vars, expr_types, used_vars
from .device import DeviceBase, DevicePlanner, KernelPlan, ctype_of, make_runtime_error, merge_regions

_MATH = {
    "sqrt": "sqrt", "exp": "exp", "exp2": "exp2", "log": "log", "log10": "log10", "log2": "log2",
    "sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "arcsinh": "asinh", "arccosh": "acosh",
    "arctanh": "atanh", "floor": "floor", "ceil": "ceil", "trunc": "trunc", "rint": "rint",
    "arctan2": "atan2", "copysign": "copysign",
    # not in the Metal standard library: implemented in the prelude below
    "log1p": "pk_log1p", "expm1": "pk_expm1", "erf": "pk_erf", "cbrt": "pk_cbrt", "hypot": "pk_hypot",
}  # fmt: skip

PRELUDE = (
    r"""
#include <metal_stdlib>
using namespace metal;

#define PK_INLINE static inline
#define INT64_C(x) (x##L)
#define UINT64_C(x) (x##UL)
#ifndef INT64_MIN
#define INT64_MIN (-9223372036854775807L - 1L)
#endif
#define fmodf fmod
#define floorf floor
#define copysignf copysign
"""
    + runtime_c.HELPERS_INT
    + runtime_c.HELPERS_FLOAT_DEF
    + runtime_c.HELPERS_F32
    + runtime_c.HELPERS_RNG
    + r"""
PK_INLINE float pk_log1p(float x) {
  float u = 1.0f + x;
  return (u == 1.0f) ? x : (log(u) * (x / (u - 1.0f)));
}
PK_INLINE float pk_expm1(float x) {
  float u = exp(x);
  if (u == 1.0f) return x;
  if (u - 1.0f == -1.0f) return -1.0f;
  return isinf(u) ? u : ((u - 1.0f) * (x / log(u)));
}
PK_INLINE float pk_erf(float x) {
  /* Abramowitz & Stegun 7.1.26: absolute error below 1.5e-7 */
  float a = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * a);
  float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a);
  return (x != x) ? x : copysign(y, x);
}
PK_INLINE float pk_cbrt(float x) { return (x == 0.0f || x != x) ? x : copysign(pow(fabs(x), 1.0f / 3.0f), x); }
PK_INLINE float pk_hypot(float x, float y) {
  float a = fabs(x), b = fabs(y);
  float hi = max(a, b), lo = min(a, b);
  if (isinf(a) || isinf(b)) return INFINITY;
  if (hi == 0.0f) return hi + lo;
  float r = lo / hi;
  return hi * sqrt(1.0f + r * r);
}
PK_INLINE void pk_fail(device atomic_int* flag, device long* info, int id, long a0, long a1, long a2, long a3) {
  int expected = 0;
  if (atomic_compare_exchange_weak_explicit(flag, &expected, id + 1, memory_order_relaxed, memory_order_relaxed)) {
    info[0] = a0; info[1] = a1; info[2] = a2; info[3] = a3;
  }
}
"""
)


class MetalEmitter(CFamilyEmitter):
    float32_suffix = ""

    def ctype(self, t: ScalarType) -> str:
        if t.name == "float64":
            raise Unsupported(
                "it uses float64, which Apple GPUs do not support (use float32 arrays and np.float32 scalars)"
            )
        return t.ctype

    def math_fn(self, op: str, t: ScalarType) -> str:
        name = _MATH.get(op)
        if name is None:
            raise Unsupported(f"the function '{op}' is not available in Metal")
        return name

    def float_prim(self, op: str, a: list[str], t: ScalarType) -> str:
        self.ctype(t)
        if op == "abs":
            return f"fabs({a[0]})"
        if op == "fmod":
            return f"fmod({a[0]}, {a[1]})"
        if op == "pow":
            return f"pow({a[0]}, {a[1]})"
        return super().float_prim(op, a, t)

    def predicate(self, op: str, x: str, t: ScalarType) -> str:
        self.ctype(t)
        return f"{op}({x})"

    def float_to_int(self, x: str, src: ScalarType, dst: ScalarType) -> str:
        self.ctype(src)
        if dst.name == "uint64":
            return f"pk_f2u64f((float){x})"
        if dst.name == "int64":
            return f"pk_f2i64f((float){x})"
        return f"(({self.ctype(dst)})pk_f2i64f((float){x}))"

    def cast(self, e: ir.Cast) -> str:
        self.ctype(e.type)
        self.ctype(e.arg.type)
        return super().cast(e)

    def const(self, c: ir.Const) -> str:
        self.ctype(c.type)
        return super().const(c)

    def rtcall(self, e: ir.RtCall) -> str:
        if not e.pure or e.name == "rand_f64":
            raise Unsupported(f"the runtime function '{e.name}' is not available on the GPU")
        return f"pk_{e.name}({', '.join(self.expr(a) for a in e.args)})"

    def check(self, s: ir.Check, out: list[str], pad: str) -> None:
        eid = self.error_id(s.err)
        args = [f"(long)({self.expr(a)})" for a in s.args[:4]]
        args += ["0L"] * (4 - len(args))
        out.append(
            f"{pad}if (!({self.expr(s.cond)})) {{ pk_fail(pk_flag, pk_info, {eid}, {', '.join(args)}); return; }}"
        )

    def release(self, v: ir.Var, out: list[str], pad: str) -> None:
        raise Unsupported("it allocates temporary arrays inside the loop")

    def alloc(self, s: ir.Alloc, out: list[str], pad: str) -> None:
        raise Unsupported("it allocates temporary arrays inside the loop")

    def assign_buf(self, s: ir.Assign, out: list[str], pad: str) -> None:
        raise Unsupported("it selects between arrays at run time inside the loop")

    def incref(self, s: ir.Incref, out: list[str], pad: str) -> None:
        raise Unsupported("it keeps references to temporary arrays inside the loop")


class MetalPlanner(DevicePlanner):
    name = "Metal"
    max_buffers = 28

    def generate(self, loop: ir.ParLoop, free: list[ir.Var]) -> KernelPlan:
        em = MetalEmitter()
        defined = defined_vars(loop.body, into_parloops=True)
        scalars = [v for v in free if not isinstance(v.type, BufType)]
        bufs = [v for v in free if isinstance(v.type, BufType)]
        if len(bufs) > self.max_buffers:
            raise Unsupported(f"it uses {len(bufs)} arrays (Metal kernels are limited to {self.max_buffers})")
        for t in expr_types(loop.body):
            em.ctype(t)
        for v in free:
            em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)
        body: list[str] = []
        em.block(loop.body, body, 1)
        # uniforms, largest first so the layout has no surprising padding
        scalars = sorted(scalars, key=lambda v: -v.type.itemsize)
        lines = [PRELUDE, "struct pk_args {", "  long total;", "  long csize;"]
        for v in scalars:
            lines.append(f"  {em.ctype(v.type)} {v.name};")
        lines.append("};")
        params = [
            "constant pk_args& pk_a [[buffer(0)]]",
            "device atomic_int* pk_flag [[buffer(1)]]",
            "device long* pk_info [[buffer(2)]]",
        ]
        for k, v in enumerate(bufs):
            params.append(f"device {em.ctype(v.type.elt)}* {v.name}_d [[buffer({k + 3})]]")
        params.append("uint3 pk_gid [[thread_position_in_grid]]")
        lines.append("kernel void pk_kernel(" + ",\n    ".join(params) + ") {")
        lines.append(f"  const long {loop.chunk.name} = (long)pk_gid.x;")
        lines.append(f"  const long {loop.start.name} = {loop.chunk.name} * pk_a.csize;")
        lines.append(f"  if ({loop.start.name} >= pk_a.total) return;")
        lines.append(f"  const long {loop.stop.name} = min({loop.start.name} + pk_a.csize, pk_a.total);")
        for v in scalars:
            lines.append(f"  const {em.ctype(v.type)} {v.name} = pk_a.{v.name};")
        for _uid, v in sorted(defined.items()):
            if v is loop.chunk or v is loop.start or v is loop.stop:
                continue
            if isinstance(v.type, BufType):
                raise Unsupported("it creates arrays inside the loop")
            init = "false" if v.type.kind == "b" else "0"
            lines.append(f"  {em.ctype(v.type)} {v.name} = {init};")
        lines += body
        lines.append("}")
        written = {s.buf.uid for s in ir.walk_stmts(loop.body) if isinstance(s, ir.Store)}
        return KernelPlan(
            loop=loop, source="\n".join(lines), scalars=scalars, bufs=bufs, errors=em.errors,
            written={v.uid for v in bufs if v.uid in written},
        )  # fmt: skip


class _Launch(threading.local):
    def __init__(self) -> None:
        self.stack: list = []


class MetalDevice(DeviceBase):
    """The system's default Metal device (a process-wide singleton)."""

    name = "metal"
    _instance: MetalDevice | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> MetalDevice:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise BackendUnavailable(
                "the Metal backend only exists on macOS; on this platform use backend='multicore' "
                "(or backend='cuda' with an NVIDIA GPU)"
            )
        try:
            import Metal
        except ImportError as e:
            raise BackendUnavailable(
                "the Metal backend needs the PyObjC Metal bindings, which are not installed. "
                'Install them with: pip install "parakeet[metal]"  (or: pip install pyobjc-framework-Metal)'
            ) from e
        self.Metal = Metal
        self.dev = Metal.MTLCreateSystemDefaultDevice()
        if self.dev is None:
            raise BackendUnavailable(
                "Metal is installed but this machine has no usable GPU device "
                "(MTLCreateSystemDefaultDevice returned nil; this happens in some virtual machines and "
                "over SSH without a GUI session). Use backend='multicore' instead."
            )
        self.queue = self.dev.newCommandQueue()
        self.device_name = str(self.dev.name())
        super().__init__()

    def planner(self) -> MetalPlanner:
        return MetalPlanner(self)

    # ---- kernels -----------------------------------------------------------------

    def compile(self, plan: KernelPlan):
        Metal = self.Metal
        opts = Metal.MTLCompileOptions.alloc().init()
        if hasattr(opts, "setMathMode_") and hasattr(Metal, "MTLMathModeSafe"):
            opts.setMathMode_(Metal.MTLMathModeSafe)
        else:
            opts.setFastMathEnabled_(False)
        lib, err = self.dev.newLibraryWithSource_options_error_(plan.source, opts, None)
        if lib is None:
            raise InternalError(f"Metal rejected a generated kernel:\n{err}\n--- kernel source ---\n{plan.source}")
        fn = lib.newFunctionWithName_("pk_kernel")
        pso, err = self.dev.newComputePipelineStateWithFunction_error_(fn, None)
        if pso is None:
            raise InternalError(f"Metal could not create a pipeline for a generated kernel: {err}")
        return pso

    def _buffer(self, addr: int, nbytes: int, shared: bool):
        """A Metal buffer for host memory: zero-copy when page aligned."""
        Metal = self.Metal
        opts = Metal.MTLResourceStorageModeShared
        length = max(nbytes, 16)
        if shared and addr % runtime_c.PAGE == 0 and nbytes % runtime_c.PAGE == 0 and nbytes > 0:
            mem = (ctypes.c_char * nbytes).from_address(addr)
            buf = self.dev.newBufferWithBytesNoCopy_length_options_deallocator_(mem, nbytes, opts, None)
            if buf is not None:
                return buf, True
        if nbytes > 0:
            mem = (ctypes.c_char * nbytes).from_address(addr)
            buf = self.dev.newBufferWithBytes_length_options_(mem, length, opts)
        else:
            buf = self.dev.newBufferWithLength_options_(length, opts)
        if buf is None:
            raise MemoryError(f"Metal could not allocate a {length}-byte buffer")
        return buf, False

    def run(self, plan: KernelPlan, ctx_addr: int, total: int, nchunks: int) -> None:
        if plan.compiled is None:
            plan.compiled = self.compile(plan)
        pso = plan.compiled
        Metal = self.Metal
        ctx = plan.host_struct.from_address(ctx_addr)
        if plan.loop.kind == "map":
            n = min(total, 1 << 22)  # one GPU thread per element when possible
        else:
            n = nchunks
        csize = -(-total // n)
        n = -(-total // csize)
        args = plan.arg_struct()
        args.total, args.csize = total, csize
        for v in plan.scalars:
            setattr(args, v.name, getattr(ctx, v.name))
        arg_buf = self.dev.newBufferWithBytes_length_options_(
            bytes(args), ctypes.sizeof(args), Metal.MTLResourceStorageModeShared
        )
        flag_buf = self.dev.newBufferWithLength_options_(16, Metal.MTLResourceStorageModeShared)
        info_buf = self.dev.newBufferWithLength_options_(64, Metal.MTLResourceStorageModeShared)
        ctypes.memset(self._address(flag_buf), 0, 16)

        # one Metal buffer per group of overlapping host ranges (see merge_regions)
        pks = [_PkBuf.from_address(getattr(ctx, v.name)) for v in plan.bufs]
        regions, where = merge_regions([(pk.data or 0, pk.nbytes) for pk in pks])
        external = [False] * len(regions)
        written = [False] * len(regions)
        for v, pk, (r, _) in zip(plan.bufs, pks, where):
            external[r] = external[r] or bool(pk.flags & 1)
            written[r] = written[r] or v.uid in plan.written
        device_bufs = [self._buffer(addr, nbytes, shared=not ext) for (addr, nbytes), ext in zip(regions, external)]
        bound = [(device_bufs[r][0], off) for r, off in where]

        cb = self.queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for k, (b, off) in enumerate([(arg_buf, 0), (flag_buf, 0), (info_buf, 0), *bound]):
            enc.setBuffer_offset_atIndex_(b, off, k)
        width = min(int(pso.maxTotalThreadsPerThreadgroup()), n)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(max(width, 1), 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if cb.error() is not None:
            raise RuntimeError(f"Metal failed to run a kernel: {cb.error()}")
        for (addr, nbytes), (buf, zero_copy), was_written in zip(regions, device_bufs, written):
            if was_written and not zero_copy and nbytes > 0:
                ctypes.memmove(addr, self._address(buf), nbytes)
        flag = ctypes.c_int32.from_address(self._address(flag_buf)).value
        if flag:
            info = (ctypes.c_int64 * 4).from_address(self._address(info_buf))
            raise make_runtime_error(plan.errors[flag - 1], list(info))

    @staticmethod
    def _address(buf) -> int:
        contents = buf.contents()
        # PyObjC hands back an objc.varlist; its buffer view gives the address
        view = contents.as_buffer(1)
        return ctypes.addressof(ctypes.c_char.from_buffer(view))


class _PkBuf(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("nbytes", ctypes.c_int64),
        ("refcount", ctypes.c_int64),
        ("flags", ctypes.c_int64),
    ]


_ = (config, used_vars, ctype_of)
