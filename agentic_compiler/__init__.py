"""Agentic Compiler — Runtime-adaptive compilation for Python.

Automatically profiles hot functions, compiles them to optimized
backends (Numba, Rust, CUDA), and hot-swaps them at runtime.
"""

from __future__ import annotations

__version__ = "0.1.0"

from agentic_compiler.core import (
    Compiler,
    Profiler,
    JitBackend,
    CompilationResult,
    GridBackendSelector,
    hot_swap,
    hot_swap_restore,
)

from agentic_compiler.codegen import (
    CodeGenerator,
    NumbaGenerator,
    RustGenerator,
    GeneratedKernel,
    ast_to_numba,
    ast_to_rust,
)

__all__ = [
    "__version__",
    "Compiler",
    "Profiler",
    "JitBackend",
    "CompilationResult",
    "GridBackendSelector",
    "hot_swap",
    "hot_swap_restore",
    "CodeGenerator",
    "NumbaGenerator",
    "RustGenerator",
    "GeneratedKernel",
    "ast_to_numba",
    "ast_to_rust",
]
