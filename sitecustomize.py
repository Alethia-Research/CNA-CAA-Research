# ── Patch 1: torch sub-byte dtypes (torchao compat for torch < 2.6) ──
# torchao uses torch.int1..int7 / uint1..uint7 as dictionary keys.
# On PyTorch < 2.6 these don't exist. We create minimal stubs.
try:
    import torch as _torch
    class _StubDtype:
        """Hashable stand-in for torch.intN / torch.uintN (N=1..7)."""
        __slots__ = ("_name",)
        def __init__(self, name):
            self._name = name
        def __repr__(self):
            return self._name
        def __hash__(self):
            return hash(self._name)
        def __eq__(self, other):
            return isinstance(other, _StubDtype) and self._name == other._name
        def __ne__(self, other):
            return not self.__eq__(other)
        is_floating_point = False
    for _pfx in ("int", "uint"):
        for _i in range(1, 8):
            _a = f"{_pfx}{_i}"
            if not hasattr(_torch, _a):
                setattr(_torch, _a, _StubDtype(f"torch.{_a}"))
    try:
        import torch.utils._pytree as _pt
        if not hasattr(_pt, "register_constant"):
            _pt.register_constant = lambda x: None
    except Exception:
        pass
except Exception:
    pass

# ── Patch 2: Nuclear torchao mock for vLLM subprocesses ──
# vLLM subprocess only inspects model architecture — it never uses torchao.
# If the real torchao fails to import for ANY reason, provide a mock that
# satisfies `import torchao` / `from torchao.xxx import yyy` without crashing.
# NOTE: Do NOT import torch._inductor here — it causes partial init that
# breaks unsloth's access to torch._inductor.config later.
import sys as _sys
from types import ModuleType as _MT
class _TorchaoFallbackFinder:
    """sys.meta_path finder that catches torchao import failures."""
    _trying = set()  # reentrancy guard to prevent infinite recursion
    def find_module(self, fullname, path=None):
        if fullname == "torchao" or fullname.startswith("torchao."):
            if fullname not in _sys.modules and fullname not in self._trying:
                self._trying.add(fullname)
                try:
                    __import__(fullname)
                    return None  # real import worked, don't intercept
                except Exception:
                    return self  # real import failed, provide mock
                finally:
                    self._trying.discard(fullname)
        return None
    def load_module(self, fullname):
        if fullname in _sys.modules:
            return _sys.modules[fullname]
        mod = _MT(fullname)
        mod.__loader__ = self
        mod.__path__ = []
        mod.__file__ = f"<mock:{fullname}>"
        mod.__package__ = fullname.rsplit(".", 1)[0] if "." in fullname else fullname
        mod.__all__ = []
        _sys.modules[fullname] = mod
        return mod
_sys.meta_path.append(_TorchaoFallbackFinder())
