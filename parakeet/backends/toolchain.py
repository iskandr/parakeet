"""Find a C compiler, build shared libraries, and cache them on disk."""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from ..config import config
from ..errors import BackendUnavailable, CCompilerError
from . import runtime_c

_lock = threading.RLock()


def cache_dir() -> Path:
    if config.cache_dir:
        base = Path(config.cache_dir).expanduser()
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "parakeet"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "parakeet" / "Cache"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "parakeet"
    try:
        base.mkdir(parents=True, exist_ok=True)
        probe = base / f".write-test-{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError:
        base = Path(tempfile.gettempdir()) / f"parakeet-cache-{os.getuid() if hasattr(os, 'getuid') else 'user'}"
        base.mkdir(parents=True, exist_ok=True)
    return base


def clear_cache() -> int:
    """Delete every cached compiled module; returns the number of files removed."""
    n = 0
    for f in cache_dir().glob("pk_*"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


@dataclass(frozen=True)
class Toolchain:
    cc: tuple[str, ...]
    kind: str  # "gcc" | "clang" | "msvc"
    version: str
    flags: tuple[str, ...]

    @property
    def ext(self) -> str:
        return {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")


def _run(cmd: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _identify(cc: str) -> tuple[str, str] | None:
    try:
        out = _run([cc, "--version"], 20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout + out.stderr).strip()
    if out.returncode != 0 and "Microsoft" not in text:
        try:
            out = _run([cc], 20)
            text = (out.stdout + out.stderr).strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
    first = text.splitlines()[0] if text else ""
    low = text.lower()
    if "microsoft" in low:
        return "msvc", first
    if "clang" in low:
        return "clang", first
    if "gcc" in low or "free software foundation" in low:
        return "gcc", first
    return None


def _candidates() -> list[str]:
    if config.cc:
        return [config.cc]
    names = ["cc", "gcc", "clang"]
    if os.name == "nt":
        names = ["gcc", "clang", "cl"]
    return names


def _accepts(cc: tuple[str, ...], kind: str, flags: list[str]) -> bool:
    with tempfile.TemporaryDirectory(prefix="pk_probe_") as d:
        src = Path(d) / "t.c"
        src.write_text("int pk_t(void) { return 0; }\n")
        out = Path(d) / ("t.obj" if kind == "msvc" else "t.o")
        if kind == "msvc":
            cmd = [*cc, "/nologo", "/c", str(src), f"/Fo{out}", *flags]
        else:
            cmd = [*cc, "-c", str(src), "-o", str(out), "-Werror", *flags]
        try:
            return _run(cmd, 60).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False


@functools.lru_cache(maxsize=4)
def _toolchain(cc_setting: str | None, opt_level: int, fast_math: bool, cflags: tuple[str, ...]) -> Toolchain:
    tried = []
    for name in _candidates():
        path = shutil.which(name) or (name if os.path.exists(name) else None)
        if path is None:
            tried.append(f"{name}: not found")
            continue
        ident = _identify(path)
        if ident is None:
            tried.append(f"{name}: could not identify compiler")
            continue
        kind, version = ident
        cc = (path,)
        if kind == "msvc":
            flags = [f"/O{min(max(opt_level, 0), 2)}", "/LD", "/nologo", "/fp:precise" if not fast_math else "/fp:fast"]
        else:
            flags = [f"-O{opt_level}", "-std=gnu11", "-fwrapv", "-fno-strict-aliasing", "-fno-math-errno", "-w"]
            if os.name != "nt":
                flags.append("-fPIC")
            else:
                # MinGW: do not depend on compiler runtime DLLs that Python cannot find
                flags += ["-static-libgcc", "-static"]
            flags.append("-ffast-math" if fast_math else "-ffp-contract=off")
            native = "-mcpu=native" if platform.machine().lower() in ("arm64", "aarch64") else "-march=native"
            if _accepts(cc, kind, [native]):
                flags.append(native)
            if not _accepts(cc, kind, [f for f in flags if f != native]):
                tried.append(f"{name}: failed to compile a trivial program")
                continue
            if _accepts(cc, kind, ["-fopenmp-simd"]):
                flags.append("-fopenmp-simd")  # honour "#pragma omp simd" (needs no OpenMP runtime)
        return Toolchain(cc, kind, version, tuple(flags) + tuple(cflags))
    hint = {
        "darwin": "install the Xcode command line tools with `xcode-select --install`",
        "win32": "install MinGW-w64 gcc or LLVM clang and make sure it is on PATH "
        "(or set PARAKEET_CC to the compiler executable)",
    }.get(sys.platform, "install gcc or clang with your package manager (e.g. `apt install gcc`)")
    raise BackendUnavailable(
        "Parakeet needs a C compiler to generate native code, but none was usable ("
        + "; ".join(tried)
        + f"). To fix this, {hint}. You can point Parakeet at a specific compiler with "
        "parakeet.config.cc or the PARAKEET_CC environment variable, or use backend='python' "
        "to run without compiling."
    )


def get_toolchain() -> Toolchain:
    return _toolchain(config.cc, config.opt_level, config.fast_math, tuple(config.cflags))


def supports_simd_pragmas() -> bool:
    return "-fopenmp-simd" in get_toolchain().flags


def _link_cmd(tc: Toolchain, src: Path, out: Path, extra: tuple[str, ...]) -> list[str]:
    if tc.kind == "msvc":
        return [*tc.cc, *tc.flags, *extra, str(src), f"/Fe{out}", f"/Fo{out.with_suffix('.obj')}"]
    cmd = [*tc.cc, *tc.flags, "-shared", str(src), "-o", str(out), *extra]
    if sys.platform.startswith("linux"):
        cmd.append("-lm")
    return cmd


def compile_shared(source: str, extra_flags: tuple[str, ...] = (), tag: str = "mod") -> Path:
    """Compile ``source`` into a shared library; returns the (cached) path."""
    tc = get_toolchain()
    digest = hashlib.sha256(
        "\0".join([source, tc.version, " ".join(tc.cc), " ".join(tc.flags), " ".join(extra_flags)]).encode()
    ).hexdigest()[:24]
    cdir = cache_dir()
    out = cdir / f"pk_{tag}_{digest}{tc.ext}"
    if out.exists() and out.stat().st_size > 0:
        return out
    with _lock:
        if out.exists() and out.stat().st_size > 0:
            return out
        with tempfile.TemporaryDirectory(prefix="pk_build_", dir=cdir) as d:
            src = Path(d) / f"pk_{tag}_{digest}.c"
            src.write_text(source)
            tmp_out = Path(d) / out.name
            cmd = _link_cmd(tc, src, tmp_out, extra_flags)
            try:
                proc = _run(cmd, 600)
            except subprocess.TimeoutExpired:
                raise CCompilerError(f"C compiler timed out: {' '.join(cmd)}") from None
            except OSError as e:
                raise CCompilerError(f"could not run the C compiler ({e}): {' '.join(cmd)}") from None
            if proc.returncode != 0 or not tmp_out.exists():
                kept = cdir / f"failed_{digest}.c"
                try:
                    shutil.copyfile(src, kept)
                except OSError:
                    kept = src
                stderr = (proc.stderr or proc.stdout or "").strip()
                tail = "\n".join(stderr.splitlines()[:40])
                raise CCompilerError(
                    "the C compiler rejected code generated by Parakeet. This is a bug in Parakeet - "
                    "please report it together with the file below.\n"
                    f"  command: {' '.join(cmd)}\n  source kept at: {kept}\n{tail}",
                    str(kept),
                    stderr,
                )
            try:
                os.replace(tmp_out, out)
            except OSError:
                # another process installed (and on Windows, loaded) the same module first
                if not out.exists():
                    raise
    return out


def load(path: Path) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(path))
    except OSError as e:
        raise CCompilerError(f"could not load compiled module {path}: {e}") from None


# ---------------------------------------------------------------------------
# threading runtimes
# ---------------------------------------------------------------------------

PARFOR_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64)


def _openmp_flags(tc: Toolchain) -> list[tuple[str, ...]]:
    if tc.kind == "msvc":
        return [("/openmp",)]
    options: list[tuple[str, ...]] = [("-fopenmp",)]
    if sys.platform == "darwin":
        prefixes = [os.environ.get("CONDA_PREFIX"), "/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
        for p in prefixes:
            if p and os.path.exists(os.path.join(p, "include", "omp.h")):
                lib = os.path.join(p, "lib")
                options.append(
                    ("-Xclang", "-fopenmp", f"-I{os.path.join(p, 'include')}", f"-L{lib}", "-lomp", f"-Wl,-rpath,{lib}")
                )
    return options


@dataclass
class ThreadRuntime:
    name: str
    lib: ctypes.CDLL | None
    parfor: int  # address of pk_rt_parallel_for (0: run chunks serially)


_runtimes: dict[str, ThreadRuntime | str] = {}


def _build_runtime(mode: str) -> ThreadRuntime:
    if mode == "serial":
        return ThreadRuntime("serial", None, 0)
    tc = get_toolchain()
    define = ("/DPK_PAR=%d" if tc.kind == "msvc" else "-DPK_PAR=%d") % runtime_c.PAR_MODES[mode]
    variants: list[tuple[str, ...]]
    if mode == "openmp":
        variants = _openmp_flags(tc)
    elif mode == "gcd":
        if sys.platform != "darwin":
            raise BackendUnavailable("the 'gcd' (Grand Central Dispatch) backend is only available on macOS")
        variants = [()]
    elif mode == "threads":
        if os.name == "nt":
            raise BackendUnavailable(
                "the 'threads' backend uses POSIX threads and is not available on Windows; use backend='openmp'"
            )
        variants = [("-pthread",)]
    else:
        raise ValueError(mode)
    last: Exception | None = None
    for extra in variants:
        try:
            path = compile_shared(runtime_c.RUNTIME, (define, *extra), tag=f"rt_{mode}")
            lib = load(path)
            lib.pk_rt_probe.restype = ctypes.c_int32
            if lib.pk_rt_probe() != 0:
                raise CCompilerError(f"the {mode} threading runtime failed its self-test")
            addr = ctypes.cast(lib.pk_rt_parallel_for, ctypes.c_void_p).value or 0
            return ThreadRuntime(mode, lib, addr)
        except CCompilerError as e:
            last = e
    hint = ""
    if mode == "openmp":
        hint = (
            " OpenMP support was not found: on macOS run `brew install libomp`; on Linux install gcc "
            "(libgomp) or clang with libomp; or use backend='multicore' to let Parakeet pick another "
            "threading strategy."
        )
    raise BackendUnavailable(f"the '{mode}' threading backend is not available on this machine.{hint}") from last


def thread_runtime(mode: str) -> ThreadRuntime:
    """Return the runtime for a threading strategy, building it on first use."""
    with _lock:
        hit = _runtimes.get(mode)
        if isinstance(hit, ThreadRuntime):
            return hit
        if isinstance(hit, str):
            raise BackendUnavailable(hit)
        try:
            rt = _build_runtime(mode)
        except BackendUnavailable as e:
            _runtimes[mode] = str(e)
            raise
        _runtimes[mode] = rt
        return rt


def best_thread_runtime() -> ThreadRuntime:
    """The preferred multi-core strategy on this machine.

    A persistent thread pool is preferred because it needs no extra runtime
    library; OpenMP is used where threads are unavailable (Windows)."""
    order = ["openmp", "serial"] if os.name == "nt" else ["threads", "gcd", "openmp", "serial"]
    for mode in order:
        try:
            return thread_runtime(mode)
        except BackendUnavailable:
            continue
    return thread_runtime("serial")


def available_thread_runtimes() -> list[str]:
    out = []
    for mode in ("threads", "gcd", "openmp"):
        try:
            thread_runtime(mode)
            out.append(mode)
        except BackendUnavailable:
            pass
    return out
