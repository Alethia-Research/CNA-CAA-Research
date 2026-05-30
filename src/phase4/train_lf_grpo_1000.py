# =====================================================================
# LF-GRPO 1000-STEP COGNITIVE MONOLOGUE TRAINING PIPELINE
# Designed for Google Colab Tesla T4 & Single-GPU Environments
# Alethia Research — Periphery Alignment & Central Logic
# =====================================================================

import os
import re
import sys
import json
import shutil
import subprocess
from types import ModuleType
from importlib.machinery import ModuleSpec

# Dynamically write sitecustomize.py to standard site-packages to patch all processes.
# This runs in vLLM subprocesses that are fresh Python interpreters without our patches.
_SITECUSTOMIZE_CONTENT = '''\
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
'''

try:
    import site
    import os
    written = False
    try:
        dirs = site.getsitepackages()
        for d in dirs:
            try:
                target_path = os.path.join(d, "sitecustomize.py")
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(_SITECUSTOMIZE_CONTENT)
                written = True
                print(f"[+] Successfully wrote global sitecustomize.py to {target_path}")
                break
            except Exception:
                continue
    except Exception:
        pass

    if not written:
        with open("sitecustomize.py", "w", encoding="utf-8") as f:
            f.write(_SITECUSTOMIZE_CONTENT)
        print("[+] Successfully wrote local sitecustomize.py fallback.")
except Exception as e:
    print(f"[-] sitecustomize.py writer encountered an issue: {e}")

# =====================================================================
# 1. ENVIRONMENT PROPERTIES & DEPENDENCY RESOLUTION
# =====================================================================

# Bypass Unsloth's aggressive mock-blocking on-the-fly
# Note: Do NOT set UNSLOTH_VLLM_STANDBY — it overrides gpu_memory_utilization
# to 0.8, which OOMs on T4 (14.6 GB). We control memory via vllm_gpu_memory_utilization.

# Import torch first to completely prevent circular import locks in modern PyTorch/Unsloth integrations
import torch

# ── Patch torch missing int/uint sub-byte dtypes ──────────────────────────────
# torchao (imported transitively by transformers -> unsloth) crashes at module
# level if torch.int1 / torch.uint1 … torch.uint7 are absent (PyTorch < 2.6).
# This MUST run before any import of unsloth / torchao / transformers.
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

for _prefix in ["int", "uint"]:
    for _i in range(1, 8):
        _attr = f"{_prefix}{_i}"
        if not hasattr(torch, _attr):
            setattr(torch, _attr, _StubDtype(f"torch.{_attr}"))
try:
    import torch.utils._pytree as _pt
    if not hasattr(_pt, "register_constant"):
        _pt.register_constant = lambda x: None
except Exception:
    pass

# Mock torchvision dynamically using a PEP 451 sys.meta_path finder and MagicMock.
# This prevents CUDA version mismatch crashes in torchvision when imported by HF transformers.
from unittest.mock import MagicMock

class TorchvisionMockFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("torchvision"):
            return ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        mock = MagicMock()
        mock.__name__ = spec.name
        mock.__path__ = []
        return mock

    def exec_module(self, module):
        pass

sys.meta_path.insert(0, TorchvisionMockFinder())

def is_google_colab():
    """Detects if the script is currently running inside Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# The compatible transformers range: >= 4.51.3 (unsloth CompileConfig) and < 5.0 (vLLM 0.7.x compat)
_TRANSFORMERS_PIN = "transformers>=4.51.3,<5.0"

def install_dependencies():
    """Performs a bulletproof environment setup using uv pip."""
    print("[*] Environment check: Executing automated installation...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-q", "uv"], check=True)
        print("[+] Installed uv package manager.")
        
        print("[*] Installing custom Unsloth and HF ecosystems...")
        subprocess.run(["uv", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b"], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--upgrade", "datasets", "accelerate", "bitsandbytes", "tqdm"], check=True)
        
        # Resolve exact vLLM wheel based on system CUDA version to prevent dynamic link mismatch
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cuda_version = (torch.version.cuda or "").replace(".", "")
        if device == "cuda" and cuda_version:
            # Install nvjitlink to fix bitsandbytes "libnvJitLink.so.13" load error in Colab
            print("[*] Installing NVIDIA nvjitlink for bitsandbytes...")
            subprocess.run(["uv", "pip", "install", "-q", "nvidia-nvjitlink-cu12"], check=True)
            
            # Dynamically match against supported wheels.vllm.ai folders: cu118, cu121, cu124, cu128
            major_minor = cuda_version[:3]  # e.g., "128" or "124" or "121" or "118"
            if major_minor in ["128", "124", "121", "118"]:
                target_cuda = major_minor
            elif cuda_version.startswith("12"):
                # Fallbacks for other CUDA 12.x versions
                val = int(cuda_version)
                if val < 124:
                    target_cuda = "121"
                elif val < 128:
                    target_cuda = "124"
                else:
                    target_cuda = "128"
            else:
                target_cuda = "124"  # Default fallback
            
            wheel_url = f"https://wheels.vllm.ai/cu{target_cuda}"
            print(f"[*] Installing vLLM from {wheel_url} matching CUDA {target_cuda}...")
            subprocess.run(["uv", "pip", "install", "-q", "vllm==0.7.2", "--extra-index-url", wheel_url], check=True)
        else:
            print("[!] Fallback: Installing standard vLLM...")
            subprocess.run(["uv", "pip", "install", "-q", "vllm==0.7.2"], check=True)
        
        # FINAL STEP: Force-pin transformers to the vLLM-compatible range.
        # This MUST run last to override whatever version unsloth/TRL pulled in.
        print(f"[*] Pinning {_TRANSFORMERS_PIN} (vLLM 0.7.x + unsloth compat)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _TRANSFORMERS_PIN], check=True)
            
        print("[+] Dependency installation complete!")
        if is_google_colab():
            print("[!] IMPORTANT: Please restart the Colab session (Runtime -> Restart session) to clear cached libraries.")
    except Exception as e:
        print(f"[-] Dependency setup encountered an issue: {e}")

# If we are running inside Google Colab, auto-install missing libraries
if is_google_colab():
    try:
        import unsloth
        import trl
        import datasets
        import vllm
        
        # Unsloth vllm_utils is currently incompatible with vLLM 0.7.3+ (missing tokenizer module)
        vllm_version_parts = [int(x) for x in vllm.__version__.split(".")[:3] if x.isdigit()]
        if len(vllm_version_parts) >= 3 and tuple(vllm_version_parts[:3]) >= (0, 7, 3):
            print(f"[-] vLLM {vllm.__version__} is incompatible with Unsloth. Triggering downgrade to 0.7.2...")
            raise ImportError("vLLM version too new")
            
        print("[+] Core machine learning environments detected.")
    except ImportError:
        install_dependencies()
        # Force restart after fresh install to flush stale C extensions
        # (e.g., PIL _imaging mismatch) and apply newly-written sitecustomize.py
        print("[+] Restarting Python process after fresh install...")
        os.environ["_LF_GRPO_RESTARTED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # ── Auto-downgrade transformers if >= 5.0 ──────────────────────────────────
    # transformers 5.x breaks vLLM 0.7.x subprocesses (ProcessorMixin removed,
    # torchao→torch.int1 crash). Without vLLM, generation is ~10x slower (~26h).
    # Target: >=4.51.3 (unsloth needs CompileConfig), <5.0 (vLLM compat).
    if os.environ.get("_LF_GRPO_RESTARTED") != "1":
        try:
            import transformers as _tf_check
            _tf_major = int(_tf_check.__version__.split(".")[0])
            if _tf_major >= 5:
                print(f"[!] transformers {_tf_check.__version__} detected — incompatible with vLLM 0.7.x.")
                print(f"[*] Auto-downgrading to {_TRANSFORMERS_PIN}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", _TRANSFORMERS_PIN],
                    check=True
                )
                print("[+] Downgrade complete. Restarting Python process to flush old modules...")
                os.environ["_LF_GRPO_RESTARTED"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[!] transformers version check/downgrade encountered an issue: {e}")
    else:
        print("[+] Post-restart: transformers version lock confirmed.")

    # torchao compat is now handled by sitecustomize.py (mock fallback) and
    # on-disk patches to transformers/quantizers/auto.py. No version pinning needed.

# =====================================================================
# 2. SYSTEM PATCHES & DUMMY DEPENDENCIES MOCKING
# =====================================================================

# torch int/uint dtype patch already applied at module top (before unsloth import)

# ── Patch vLLM + transformers source files on disk for subprocess compat ────
# vLLM 0.7.x spawns a subprocess for model inspection. That subprocess imports
# vLLM which imports transformers which imports torchao. The crash chain:
#   vllm/multimodal/processing.py (or inputs/registry.py)
#     → from transformers import ProcessorMixin
#     → transformers.processing_utils → .modeling_utils → .quantizers → torchao
#     → torch.int1 crash (torch < 2.6)  OR  do_bench crash (torchao 0.4)
#
# STRATEGY: Cut the chain at the ROOT by patching transformers/quantizers/auto.py
# to make the torchao import optional. This single patch eliminates ALL torchao
# version-specific crashes in one shot. We also keep the vLLM patches as backup.
def _patch_vllm_for_subprocess():
    """Patch vLLM + transformers source files on disk so subprocesses survive."""
    import importlib

    # ══════════════════════════════════════════════════════════════════════
    # Part A (ROOT FIX): Patch transformers/quantizers/auto.py
    # Guard `from .quantizer_torchao import TorchAoHfQuantizer` so that if
    # torchao is broken (ANY version mismatch), transformers still loads.
    # ══════════════════════════════════════════════════════════════════════
    try:
        spec = importlib.util.find_spec("transformers.quantizers.auto")
        if spec and spec.origin:
            auto_file = spec.origin
            with open(auto_file, "r", encoding="utf-8") as f:
                auto_src = f.read()

            target = "from .quantizer_torchao import TorchAoHfQuantizer"
            guarded = (
                "try:\n"
                "    from .quantizer_torchao import TorchAoHfQuantizer\n"
                "except Exception:\n"
                "    TorchAoHfQuantizer = None  # torchao unavailable — patched for vLLM subprocess compat"
            )

            if target in auto_src and "# torchao unavailable" not in auto_src:
                auto_src = auto_src.replace(target, guarded)
                with open(auto_file, "w", encoding="utf-8") as f:
                    f.write(auto_src)
                print(f"[+] Patched transformers quantizers/auto.py — torchao import is now guarded")
            elif "# torchao unavailable" in auto_src:
                print("[+] transformers quantizers/auto.py torchao guard already applied.")
            else:
                print("[*] transformers quantizers/auto.py: no matching import found — skipping.")
    except Exception as e:
        print(f"[*] transformers quantizers/auto.py patch skipped: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Part B: Patch vllm/inputs/registry.py (ProcessorMixin)
    # ══════════════════════════════════════════════════════════════════════
    _pm_stub = (
        "from transformers import BatchFeature, PretrainedConfig\n"
        "try:\n"
        "    from transformers import ProcessorMixin\n"
        "except ImportError:\n"
        "    class ProcessorMixin: pass  # stub for transformers 4.5x+"
    )
    try:
        import vllm.inputs.registry as _vir
        registry_file = _vir.__file__
        if registry_file:
            with open(registry_file, "r", encoding="utf-8") as f:
                src = f.read()
            original = "from transformers import BatchFeature, PretrainedConfig, ProcessorMixin"
            if original in src:
                src = src.replace(original, _pm_stub)
                with open(registry_file, "w", encoding="utf-8") as f:
                    f.write(src)
                print(f"[+] Patched vLLM inputs/registry.py ProcessorMixin import")
            elif "class ProcessorMixin: pass  # stub" in src:
                print("[+] vLLM inputs/registry.py already patched.")
    except Exception as e:
        print(f"[*] vLLM inputs/registry.py patch skipped: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Part C: Patch vllm/multimodal/processing.py (ProcessorMixin)
    # This file also does `from transformers import ... ProcessorMixin`
    # ══════════════════════════════════════════════════════════════════════
    try:
        spec = importlib.util.find_spec("vllm.multimodal.processing")
        if spec and spec.origin:
            mm_file = spec.origin
            with open(mm_file, "r", encoding="utf-8") as f:
                mm_src = f.read()
            original = "from transformers import BatchFeature, PretrainedConfig, ProcessorMixin"
            if original in mm_src:
                mm_src = mm_src.replace(original, _pm_stub)
                with open(mm_file, "w", encoding="utf-8") as f:
                    f.write(mm_src)
                print(f"[+] Patched vLLM multimodal/processing.py ProcessorMixin import")
            elif "class ProcessorMixin: pass  # stub" in mm_src:
                print("[+] vLLM multimodal/processing.py already patched.")
    except Exception as e:
        print(f"[*] vLLM multimodal/processing.py patch skipped: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Part D: Patch vllm/transformers_utils/processor.py
    # ══════════════════════════════════════════════════════════════════════
    try:
        spec = importlib.util.find_spec("vllm.transformers_utils.processor")
        if spec and spec.origin:
            proc_file = spec.origin
            with open(proc_file, "r", encoding="utf-8") as f:
                proc_src = f.read()
            old_import = "from transformers.processing_utils import ProcessorMixin"
            safe_import = (
                "try:\n"
                "    from transformers.processing_utils import ProcessorMixin\n"
                "except (ImportError, AttributeError):\n"
                "    try:\n"
                "        from transformers import ProcessorMixin\n"
                "    except ImportError:\n"
                "        class ProcessorMixin: pass  # stub"
            )
            if old_import in proc_src and "# stub" not in proc_src:
                proc_src = proc_src.replace(old_import, safe_import)
                with open(proc_file, "w", encoding="utf-8") as f:
                    f.write(proc_src)
                print(f"[+] Patched vLLM transformers_utils/processor.py")
    except Exception as e:
        print(f"[*] vLLM processor.py patch skipped: {e}")

_patch_vllm_for_subprocess()

# Mock missing mergekit and llm_blender dependencies to prevent TRL load failures
for module_name in ["mergekit", "mergekit.config", "mergekit.merge", "llm_blender"]:
    if module_name not in sys.modules:
        dummy = ModuleType(module_name)
        dummy.__spec__ = ModuleSpec(module_name, None)
        if module_name == "mergekit.config":
            dummy.MergeConfiguration = object
        elif module_name == "mergekit.merge":
            dummy.MergeOptions = object
            dummy.run_merge = lambda *args, **kwargs: None
        sys.modules[module_name] = dummy

# Prevent local mergekit directory checks
try:
    import trl.import_utils
    trl.import_utils.is_mergekit_available = lambda: False
except ImportError:
    pass

# =====================================================================
# 3. UNSLOTH ZOO Patches & GRPOTrainer Compilation Hotpatching
# =====================================================================

# Unsloth must be imported first to compile core patches
from unsloth import FastLanguageModel, PatchFastRL
import torch
from transformers import TrainerCallback

# Clear stale compiled cache to force fresh regeneration and bypass potential corruption
_cache_dir = os.path.join(os.getcwd(), "unsloth_compiled_cache")
if os.path.exists(_cache_dir):
    try:
        shutil.rmtree(_cache_dir)
        print("[*] Cleared existing unsloth_compiled_cache for clean regeneration.")
    except Exception as e:
        print(f"[*] Note: Could not clear cache directory: {e}")

# Apply FastRL patch for GRPO to enable ultra-fast vLLM-integrated generation
PatchFastRL("GRPO", FastLanguageModel)

# Trigger the dynamic compilation of UnslothGRPOTrainer.py
print("[*] Initializing compiler for Unsloth GRPOTrainer...")
try:
    from trl import GRPOTrainer
except Exception:
    pass

# Hotpatch the compiled file on disk to fix Unsloth compiler's top_k UnboundLocalError
def hotpatch_unsloth_grpo():
    trainer_file = os.path.join(os.getcwd(), "unsloth_compiled_cache", "UnslothGRPOTrainer.py")
    if os.path.exists(trainer_file):
        print(f"[*] Found compiled trainer file: {trainer_file}. Applying hotpatch...")
        with open(trainer_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        target_line = "if use_vllm and (top_k is None or top_k == 0): top_k = -1"
        replacement_lines = (
            "top_k = kwargs.get('top_k', None)\n"
            "        if use_vllm and (top_k is None or top_k == 0):\n"
            "            top_k = -1\n"
            "            kwargs['top_k'] = -1"
        )
        
        if target_line in content:
            new_content = content.replace(target_line, replacement_lines)
            with open(trainer_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("[+] Successfully hotpatched Unsloth GRPOTrainer's top_k UnboundLocalError!")
        else:
            print("[*] Note: Target line not found in compiled trainer. It might be already patched.")
    else:
        print("[-] Error: Compiled trainer file not found at the expected path.")

hotpatch_unsloth_grpo()

# Flush sys.modules to force a clean import of the hotpatched classes
for k in list(sys.modules.keys()):
    if "trl" in k or k == "UnslothGRPOTrainer" or "unsloth_compiled_cache" in k:
        del sys.modules[k]

# Import the hotpatched, working GRPOConfig and GRPOTrainer
from trl import GRPOConfig, GRPOTrainer
print("[+] GRPOConfig and GRPOTrainer successfully loaded!")

# =====================================================================
# 4. REWARD FUNCTIONS WITH LOOPHOLE FIXES
# =====================================================================

def get_completion_text(comp) -> str:
    """Safely extracts the generated text string from a completion."""
    if isinstance(comp, list):
        for msg in comp:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        if len(comp) > 0 and isinstance(comp[-1], dict):
            return comp[-1].get("content", "")
    elif isinstance(comp, str):
        return comp
    return ""

def extract_xml_answer(text: str) -> str:
    """Extracts the final numerical answer after the </think> tag if present."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    # Remove commas to avoid matching errors (e.g., 1,000 -> 1000)
    text_clean = text.replace(",", "")
    
    # 1. Look for LaTeX \boxed{...}
    boxed_match = re.search(r"\\boxed\{([0-9.-]+)\}", text_clean)
    if boxed_match:
        return boxed_match.group(1)
        
    # 2. Look for the last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text_clean)
    if numbers:
        return numbers[-1]
        
    return ""

def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """
    Standard formatting reward. Validates that the output has exactly
    one open and one closing <think> tag, and non-empty final answer content.
    """
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")
        
        if num_start == 1 and num_end == 1:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            # 1.0 reward for single closed think block with non-empty trailing output
            if start_idx < end_idx and len(comp_text[end_idx + 8 :].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif num_start == 0 or num_end == 0:
            if num_start == 0 and num_end == 0:
                # No think blocks at all
                rewards.append(0.0)
            else:
                # Unclosed/mismatched tag penalty
                rewards.append(0.2)
        else:
            # Multi-block format (exploit prevention)
            rewards.append(0.3)
    return rewards

def is_mathematically_equivalent(s1: str, s2: str) -> bool:
    if not s1 or not s2:
        return False
    # Clean commas to prevent float conversion crashes
    s1_clean = s1.replace(",", "").strip()
    s2_clean = s2.replace(",", "").strip()
    try:
        return float(s1_clean) == float(s2_clean)
    except ValueError:
        return s1_clean.lower() == s2_clean.lower()

def math_correctness_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """Compares the extracted numerical answer with the target ground truth."""
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        
        if extracted and is_mathematically_equivalent(extracted, target_clean):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def p_grpo_format_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """Posterior-GRPO Formatting: zeroes format rewards if answer is incorrect."""
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")
        
        if num_start == 1 and num_end == 1:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            if start_idx < end_idx and len(comp_text[end_idx + 8 :].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif num_start == 0 or num_end == 0:
            if num_start == 0 and num_end == 0:
                rewards.append(0.0)
            else:
                rewards.append(0.2)
        else:
            rewards.append(0.3)
    return rewards

def step_grpo_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Step-GRPO Decaying Reward based on Monologue Word Count to prevent vocabulary evasion.
    Applies a 100-word grace window to give the model "thinking space" for complex math,
    followed by a mild 0.996^(words-100) decay to suppress extreme verbosity.
    Multi-block tag exploits are penalized linearly by 0.5 per extra block.
    Zeroes out if the math answer is incorrect.
    """
    rewards = []
    
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        # 1. Extract thinking monologue content
        think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
        num_start = comp_text.count("<think>")
        
        monologue_content = ""
        if think_blocks:
            monologue_content = think_blocks[0]
        elif num_start > 0:
            # Fallback for unclosed tag
            monologue_content = comp_text.split("<think>")[-1]
            
        # 2. Count actual words inside the monologue
        words = len(monologue_content.split())
        
        # 3. Apply 100-word grace window + mild 0.996 decay
        if words <= 100:
            conciseness_decay = 1.0
        else:
            conciseness_decay = 0.996 ** (words - 100)
            
        # 4. Multi-block penalty (arbitrage prevention)
        block_penalty = 0.0
        if num_start > 1:
            block_penalty = 0.5 * (num_start - 1)
            
        decayed_reward = max(0.0, conciseness_decay - block_penalty)
        rewards.append(decayed_reward)
        
    return rewards

def think_depth_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """Rewards multi-line structured reasoning inside <think> blocks.
    Caps at 1.0 after 6 lines (GSM8K sweet spot is 3-6 steps)."""
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
        if not think_blocks:
            rewards.append(0.0)
            continue
        lines = [l.strip() for l in think_blocks[0].split('\n') if l.strip()]
        rewards.append(min(1.0, len(lines) / 6.0))
    return rewards

def quantity_inventory_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """Rewards explicitly listing quantities and showing computation steps.
    Requires >=4 distinct numbers (filters out trivial problem echoing)."""
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
        if not think_blocks:
            rewards.append(0.0)
            continue
        think_text = think_blocks[0]
        numbers_found = re.findall(r'\d+', think_text)
        has_explicit_steps = bool(re.search(
            r'(total|sum|remaining|result|equals|=)\s*[:=]?\s*\d',
            think_text.lower()
        ))
        has_multiple_quantities = len(set(numbers_found)) >= 4
        score = 0.0
        if has_multiple_quantities: score += 0.5
        if has_explicit_steps: score += 0.5
        rewards.append(score)
    return rewards

def final_format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """Rewards the GSM8K #### <number> answer format specifically."""
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        if re.search(r'####\s*-?\d+(?:\.\d+)?', comp_text):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# ── Tag-Spam Electrified Fence ──────────────────────────────────────────
# Whitelisted tags: ONLY <think> and </think> are allowed.
# Everything else gets punished — HTML, numbered, invented, all of it.
_TAG_WHITELIST = frozenset(["think"])

# Explicit HTML/web tags the model keeps inventing from its instruct pretraining
_HTML_TAGS = frozenset([
    "p", "br", "div", "span", "strong", "b", "i", "u", "em", "a", "img",
    "ul", "ol", "li", "table", "tr", "td", "th", "thead", "tbody",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "code", "blockquote",
    "hr", "nowiki", "sub", "sup", "small", "big", "center", "font",
    "section", "article", "header", "footer", "nav", "main", "aside",
    "details", "summary", "figure", "figcaption", "mark", "del", "ins",
    "underline", "strikethrough", "abbr", "cite", "q", "kbd", "var",
    "samp", "dfn", "ruby", "rt", "rp", "bdi", "bdo", "wbr",
])

# Number words the model uses as tags
_NUMBER_WORDS = frozenset([
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "first", "second", "third",
])

def tag_spam_penalty_fn(prompts, completions, **kwargs) -> list[float]:
    """Aggressive penalty for ANY tag that is not <think> or </think>.
    Catches HTML tags, digit tags (<1>, <60>), step tags (<step1>),
    number-word tags (<one>), and arbitrary invented tags.
    Returns -0.3 per bad tag occurrence, capped at -1.5."""
    penalties = []
    # Pattern matches standard and custom XML tags (including hyphens, dots, colons, and digits)
    tag_pattern = re.compile(r"</?([a-zA-Z0-9_\-.:]+)>")

    for comp in completions:
        comp_text = get_completion_text(comp)
        bad_tag_count = 0
        bad_tag_types = set()

        for match in tag_pattern.finditer(comp_text):
            tag_name = match.group(1).lower()
            if tag_name not in _TAG_WHITELIST:
                bad_tag_types.add(tag_name)
                bad_tag_count += 1

        if bad_tag_count > 0:
            penalty = max(-1.5, -0.3 * bad_tag_count)
        else:
            penalty = 0.0

        penalties.append(penalty)
    return penalties

# Global step tracker for stage shifting
current_step = 0

def get_combined_reward_fn(mode="step-grpo", stage_steps=100, is_resumed=False):
    """
    Combines formatting and correctness rewards and dynamically transitions
    the weights from format-priming (Stage 1) to correctness (Stage 2).
    Also includes the Syntactic Generalization Monitor for emergent tags.
    """
    def reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
        global current_step
        
        # SYNTACTIC GENERALIZATION MONITOR: Detect emergent/invented tags
        from collections import Counter
        invented = []
        for comp in completions:
            text = get_completion_text(comp)
            for tag in re.findall(r"</?(\w+)>", text):
                # Ignore expected tags AND tokenizer special tokens (BOS/EOS/PAD/etc.)
                if tag.lower() not in ["think", "boxed", "s", "pad", "unk", "mask",
                                        "sep", "cls", "bos", "eos", "endoftext"]:
                    invented.append(tag)
        if invented:
            print(f"\n==================================================")
            print(f"[!] EMERGENT TAGS DETECTED (Step {current_step})")
            print(f"    Unique invented tags: {sorted(set(invented))}")
            print(f"    Occurrences: {Counter(invented).most_common()}")
            print(f"==================================================")
            
        # 1. Fetch formatting reward
        if mode == "p-grpo":
            f_rewards = p_grpo_format_reward_fn(prompts, completions, target_answer)
        else:
            f_rewards = format_reward_fn(prompts, completions)
            
        # 2. Fetch correctness/conciseness reward
        if mode == "step-grpo":
            c_rewards = step_grpo_reward_fn(prompts, completions, target_answer)
        else:
            c_rewards = math_correctness_reward_fn(prompts, completions, target_answer)
            
        # 3. Auxiliary rewards
        d_rewards = think_depth_reward_fn(prompts, completions)
        i_rewards = quantity_inventory_reward_fn(prompts, completions)
        ff_rewards = final_format_reward_fn(prompts, completions)

        # 4. Tag spam penalty (electrified fence — always active, always punishing)
        spam_penalties = tag_spam_penalty_fn(prompts, completions)

        # 5. Three-stage weighting (spam penalty weight is constant across all stages)
        w_spam = 1.0  # Additive (penalties are already negative)
        effective_step = current_step + (100 if is_resumed else 0)
        
        if effective_step <= 20:
            # Stage 1 — format boot camp
            w_format, w_correct, w_depth, w_inventory, w_final = 1.0, 0.05, 0.5, 0.3, 0.5
        elif effective_step <= stage_steps + 20:
            # Stage 2 — correctness enters
            w_format, w_correct, w_depth, w_inventory, w_final = 0.3, 1.0, 0.4, 0.3, 0.3
        else:
            # Stage 3 — correctness dominant
            w_format, w_correct, w_depth, w_inventory, w_final = 0.1, 1.0, 0.2, 0.1, 0.2
            
        combined = []
        for f, c, d, i, ff, sp in zip(f_rewards, c_rewards, d_rewards, i_rewards, ff_rewards, spam_penalties):
            combined.append(w_format*f + w_correct*c + w_depth*d + w_inventory*i + w_final*ff + w_spam*sp)
        return combined
    return reward_fn

# =====================================================================
# 5. GRADIENT INSULATION VERIFICATION CALLBACK
# =====================================================================

class StepTrackerCallback(TrainerCallback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.verified = False

    def on_step_end(self, args, state, control, **kwargs):
        global current_step
        current_step = state.global_step

    def on_substep_end(self, args, state, control, **kwargs):
        """
        Fires after the very first backward pass at step 0.
        Verifies:
        1. That frozen parameters (requires_grad=False) received exactly 0.0 gradient norm.
        2. That NO base model layers were left with requires_grad=True by mistake (verifies frozen config).
        Raises RuntimeError on gradient leakage or layout freezing misconfiguration.
        """
        if state.global_step == 0 and not self.verified:
            print("\n" + "=" * 50)
            print("GRADIENT INSULATION VERIFICATION REPORT")
            print("=" * 50)
            
            active_params = []
            frozen_params = []
            config_leaks = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    active_params.append((name, param))
                    # Check if non-LoRA / base model weights have requires_grad set to True
                    if not any(x in name for x in ["lora_", "modules_to_save"]):
                        config_leaks.append(name)
                else:
                    frozen_params.append((name, param))
                    
            print(f"[+] Total trainable parameters: {len(active_params)}")
            print("[*] Sample active parameter gradient norms:")
            sample_active_shown = 0
            for name, param in active_params:
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                if sample_active_shown < 5:
                    print(f"  - {name}: grad_norm = {grad_norm:.6f}")
                    sample_active_shown += 1
                    
            print(f"[+] Total frozen parameters: {len(frozen_params)}")
            print("[*] Sample frozen parameter gradient norms:")
            sample_frozen_shown = 0
            has_leak = False
            leaked_params = []
            
            for name, param in frozen_params:
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                
                if sample_frozen_shown < 5:
                    print(f"  - {name}: grad_norm = {grad_norm:.6f}")
                    sample_frozen_shown += 1
                
                if grad_norm > 1e-8:
                    has_leak = True
                    leaked_params.append((name, grad_norm))
            
            # Fail fast if base model parameters were accidentally left trainable
            if config_leaks:
                print(f"[-] CONFIGURATION LEAK: Trainable base model parameters found!")
                for name in config_leaks[:5]:
                    print(f"  - {name} has requires_grad = True")
                print("=" * 50)
                raise RuntimeError("Gradient insulation configuration leak: Base model layers were left trainable.")
                
            if has_leak:
                print(f"[-] VERIFICATION FAILED: Frozen parameters received gradients!")
                for name, norm in leaked_params[:5]:
                    print(f"  - {name}: grad_norm = {norm:.6f}")
                print("=" * 50)
                raise RuntimeError("Gradient insulation verification failed: Frozen parameters received non-zero gradients.")
            else:
                print("[+] VERIFICATION SUCCESS: 100% gradient insulation and layout configurations confirmed!")
                print("=" * 50 + "\n")
                self.verified = True

# =====================================================================
# 5b. GOOGLE DRIVE CHECKPOINT UPLOAD CALLBACK
# =====================================================================

class DriveUploadCallback(TrainerCallback):
    """Mounts Google Drive and copies each checkpoint after it is saved."""

    def __init__(self, output_dir, drive_dest=None):
        super().__init__()
        self.output_dir = output_dir
        _run = os.path.basename(os.path.abspath(output_dir))
        self.drive_dest = drive_dest or f"/content/drive/MyDrive/grpo_checkpoints/{_run}"
        self._mounted = False
        self._try_mount()

    def _try_mount(self):
        try:
            if os.path.ismount("/content/drive"):
                self._mounted = True
                os.makedirs(self.drive_dest, exist_ok=True)
                print(f"[+] Drive already mounted → {self.drive_dest}")
                return
            from google.colab import drive as _gd
            _gd.mount("/content/drive", force_remount=False)
            self._mounted = True
            os.makedirs(self.drive_dest, exist_ok=True)
            print(f"[+] Drive mounted → {self.drive_dest}")
        except Exception as _e:
            print(f"[!] Drive mount failed ({_e}) — checkpoints local only.")

    def on_save(self, args, state, control, **kwargs):
        if not self._mounted:
            return
        try:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt_dir):
                ckpt_dir = self.output_dir
            zip_name = f"save_step_{state.global_step}"
            zip_base = os.path.join(self.drive_dest, zip_name)
            if os.path.exists(zip_base + ".zip"):
                os.remove(zip_base + ".zip")
            shutil.make_archive(zip_base, "zip", root_dir=ckpt_dir)
            print(f"[+] Step {state.global_step} → {zip_name}.zip uploaded to Drive.")
        except Exception as _e:
            print(f"[!] Drive upload failed at step {state.global_step}: {_e}")

# =====================================================================
# 6. PIPELINE EXECUTION
# =====================================================================

def run_training_pipeline(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    mode="step-grpo",
    max_steps=1000,
    stage_steps=100,
    learning_rate=1.5e-5,
    layers_to_transform="last_4",
    output_dir="./grpo_cot_output",
    save_steps=100,
    limit_train=None,
    num_generations=4,
    no_vllm=False,
    resume_adapter_from=None
):
    print(f"[*] Starting LF-GRPO training pipeline in mode: {mode.upper()}")
    
    # 1. Dataset Preparation
    SYSTEM_PROMPT = """A conversation between User and Assistant. The Assistant is a precise mathematical reasoner.

When solving a math problem, the Assistant MUST:

1. Use <think>...</think> tags to reason step by step BEFORE giving the final answer
2. Inside <think>, identify ALL quantities in the problem before calculating anything
3. Inside <think>, write out EVERY multiplication and subtraction explicitly — never skip steps
4. Pay special attention to problems involving ratios, rates, and "X times faster/more/less" — these always require two operations, not one
5. After </think>, state the final answer as a plain number with no units, symbols, or extra text

The final answer must appear on its own line in this exact format:
#### <number>

Bad example (incomplete think, skipped step):
<think>Multiply sprints by distance.</think>
180
#### 180

Good example (full think, all steps explicit):
<think>
James runs 3 sprints per session.
He runs 3 sessions per week.
Each sprint is 60 meters.
Total = 3 × 3 × 60 = 540 meters.
</think>
James runs 540 meters per week.
#### 540"""

    
    print("[*] Pre-processing GSM8K dataset...")
    from datasets import load_dataset as hf_load_dataset
    raw_gsm = hf_load_dataset("openai/gsm8k", "main", split="train")
    
    if limit_train is not None:
        raw_gsm = raw_gsm.select(range(min(limit_train, len(raw_gsm))))
        
    os.makedirs("data", exist_ok=True)
    dataset_path = "data/gsm8k_train_grpo.jsonl"
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in raw_gsm:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]}
            ]
            
            # Extract clean ground-truth answer
            ans_text = item["answer"]
            target_answer = ans_text.split("####")[-1].strip().replace(",", "")
            
            json_line = {
                "prompt": prompt_messages,
                "target_answer": target_answer
            }
            f.write(json.dumps(json_line) + "\n")
            
    print(f"[+] Loaded {len(raw_gsm)} training samples and saved to {dataset_path}")
    
    # 2. VRAM co-location OOM Safeguards & vLLM check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vllm_available = False
    if device == "cuda" and not no_vllm:
        # ── Version-gate: detect known-incompatible combos before wasting 15s on a
        #    vLLM subprocess crash that always fails.
        _version_str = torch.__version__.split("+")[0]
        _version_parts = [int(x) for x in _version_str.split(".") if x.isdigit()]
        _torch_ver = tuple(_version_parts[:3])

        import transformers as _tf
        _tf_ver = tuple(int(x) for x in _tf.__version__.split(".")[:2] if x.isdigit())

        if _torch_ver < (2, 1, 1):
            print(f"[!] torch {torch.__version__} < 2.1.1: vLLM subprocesses will crash on torch.int1. Disabling vLLM.")
        elif _tf_ver >= (5, 0):
            # transformers 5.x removed/renamed ProcessorMixin and restructured quantizers;
            # vLLM 0.7.x subprocesses import transformers at module level and crash.
            print(f"[!] transformers {_tf.__version__} >= 5.0 is incompatible with vLLM 0.7.x. Disabling vLLM.")
        else:
            try:
                import vllm
                # Also probe the import chain unsloth_zoo.patch_vllm() walks in-process.
                import vllm.transformers_utils.tokenizer  # noqa: F401
                from vllm.utils import is_list_of           # noqa: F401
                vllm_available = True
                print(f"[+] vLLM {vllm.__version__} probe passed — will use vLLM for generation.")
            except Exception as e:
                print(f"[!] vLLM probe failed ({type(e).__name__}: {e}). Falling back to HF generation.")
                vllm_available = False

    use_vllm = not no_vllm and device == "cuda" and vllm_available
    print(f"[+] Hardware detected: {device.upper()}. Using vLLM for generation: {use_vllm}")

    # 3. Model & Tokenizer Loading (with 4-bit Quantization)
    print(f"[*] Loading model {model_name} in 4-bit...")
    def _load_model(fast_inf):
        # FastLanguageModel.from_pretrained expects only model architecture arguments.
        # Passing vLLM engine settings like gpu_memory_utilization directly will raise a TypeError in HF loaders.
        # These belong in GRPOConfig as vllm_gpu_memory_utilization.
        kwargs = dict(
            model_name=model_name,
            max_seq_length=512 + 384,
            load_in_4bit=True,
            fast_inference=fast_inf,
        )
        return FastLanguageModel.from_pretrained(**kwargs)

    # Load WITHOUT fast_inference — Unsloth's embedded vLLM doesn't survive get_peft_model()
    # wrapping. TRL's native use_vllm=True on cuda:0 gives real vLLM generation.
    model, tokenizer = _load_model(False)
    
    # 4. Late-Layer LoRA Adaption (LF-GRPO)
    num_layers = model.config.num_hidden_layers
    if isinstance(layers_to_transform, str) and layers_to_transform.strip().lower().startswith("last_"):
        try:
            n = int(layers_to_transform.strip().lower().split("_")[-1])
            resolved_layers = list(range(num_layers - n, num_layers))
        except ValueError:
            raise ValueError(f"Invalid preset configuration format for layers_to_transform: '{layers_to_transform}'")
    else:
        # Expect comma separated indices, e.g. "24,25,26,27"
        resolved_layers = [int(x.strip()) for x in layers_to_transform.split(",")]
        
    print(f"[+] Spatial configuration: Freezing L0-L{resolved_layers[0]-1}. Adapting periphery layers: {resolved_layers}")
    
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=64,
        lora_alpha=96,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=resolved_layers, # Freezes all other layers automatically
        use_gradient_checkpointing="unsloth"
    )

    if resume_adapter_from:
        abs_resume_path = os.path.abspath(resume_adapter_from)
        print(f"[*] Loading adapter weights from {abs_resume_path}...")
        model.load_adapter(abs_resume_path, adapter_name="default")
        print("[+] Adapter weights loaded — stage-1 weights restored, continuing from step 100.")

    # 5. Load Formatted Dataset
    from datasets import load_dataset
    train_dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 6. Configure GRPO Trainer Args
    # ── Introspect GRPOConfig to only pass kwargs it actually accepts ──
    import inspect as _inspect
    _grpo_params = set(_inspect.signature(GRPOConfig.__init__).parameters)

    training_args_kwargs = dict(
        output_dir=output_dir,
        beta=0.0,  # Disable KL penalty → skips ref model deepcopy (vLLM LLMEngine not picklable)
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=448,
        max_completion_length=448,
        max_steps=max_steps,
        logging_steps=5,
        save_steps=save_steps,
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        report_to="none",
    )

    # TRL native vLLM: explicitly pin to cuda:0 so it doesn't grab non-existent cuda:1.
    # Model loaded with fast_inference=False — Unsloth's embedded vLLM doesn't survive
    # the PEFT wrapping anyway (generate() calls bypass it, falling back to HF at ~23s/call).
    if use_vllm and "use_vllm" in _grpo_params:
        training_args_kwargs["use_vllm"] = True
        # Force vllm_device to cuda:0 if the param exists (avoids "auto" → cuda:1 error)
        if "vllm_device" in _grpo_params:
            training_args_kwargs["vllm_device"] = "cuda:0"
        else:
            print("[!] GRPOConfig has no 'vllm_device' param — TRL may grab cuda:1. Watch for error.")
        _vllm_optional = {
            "vllm_gpu_memory_utilization": 0.5,
            "vllm_dtype": "float16",  # T4 compute capability 7.5 < 8.0; bfloat16 unsupported
            "vllm_mode": "colocate",
            "vllm_enforce_eager": True,
        }
        for k, v in _vllm_optional.items():
            if k in _grpo_params:
                training_args_kwargs[k] = v
            else:
                print(f"[*] GRPOConfig has no '{k}' param — skipping.")
    elif "use_vllm" in _grpo_params:
        training_args_kwargs["use_vllm"] = False

    training_args = GRPOConfig(**training_args_kwargs)

    # vLLM cannot load bitsandbytes pre-quantized weights (.absmax keys in safetensors).
    # Unsloth downloads the bnb-4bit variant even for non-bnb model IDs and stores that
    # path in model.config._name_or_path. vLLM needs an fp16 model path instead.
    if use_vllm:
        # Derive canonical fp16 model name: strip unsloth prefix → original publisher
        _vllm_model_name = model_name
        if "unsloth/" in model_name.lower():
            _base = model_name.split("/", 1)[1]
            if _base.lower().startswith("qwen"):
                _vllm_model_name = f"Qwen/{_base}"
            elif _base.lower().startswith("llama"):
                _vllm_model_name = f"meta-llama/{_base}"
            elif _base.lower().startswith("mistral"):
                _vllm_model_name = f"mistralai/{_base}"
        print(f"[*] vLLM fp16 inference model: {_vllm_model_name}")

        # Set on every config layer PEFT might proxy through
        for _cfg in [
            getattr(model, "config", None),
            getattr(getattr(model, "base_model", None), "config", None),
            getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
        ]:
            if _cfg is not None:
                _cfg._name_or_path = _vllm_model_name

        # Belt-and-suspenders: patch grpo_trainer's LLM reference so vLLM always
        # receives the fp16 path regardless of what TRL derives from model.config.
        import trl.trainer.grpo_trainer as _grpo_m
        _OrigLLM = getattr(_grpo_m, "LLM", None)
        if _OrigLLM is not None:
            _fp16_for_vllm = _vllm_model_name
            def _safe_llm(*_a, **_kw):
                _mn = _a[0] if _a else _kw.get("model", "")
                if isinstance(_mn, str) and _mn != _fp16_for_vllm:
                    print(f"[*] vLLM: overriding model {_mn!r} → {_fp16_for_vllm!r}")
                    _a = (_fp16_for_vllm,) + (_a[1:] if _a else ())
                    _kw.pop("model", None)
                return _OrigLLM(*_a, **_kw)
            _grpo_m.LLM = _safe_llm
    else:
        _vllm_model_name = model_name

    # ── Patch PEFT model with attributes TRL's GRPOTrainer expects from
    #    a bare PreTrainedModel but that PEFT's __getattr__ chain hides. ──
    _trl_expected_attrs = {
        "warnings_issued": {},
        "_keep_in_fp32_modules": [],
    }
    for attr_name, default_val in _trl_expected_attrs.items():
        if not hasattr(model, attr_name):
            setattr(model, attr_name, default_val)
    # ── Patch reference model creation for vLLM compat ──────────────────────
    # TRL's GRPOTrainer tries to deepcopy(model) to create a reference model,
    # but vLLM's LLMEngine attached to the model can't be pickled.
    # For LoRA, the base model (adapters disabled) IS the reference — no copy needed.
    if use_vllm:
        # Patch create_reference_model WHERE IT IS CALLED (grpo_trainer module namespace),
        # not in the source module — grpo_trainer already holds a local reference at import time.
        import trl.trainer.grpo_trainer as _grpo_mod
        import trl.models.modeling_base as _tmb
        _orig_create_ref = getattr(_grpo_mod, "create_reference_model", None) or _tmb.create_reference_model
        def _safe_create_reference_model(model, *args, **kwargs):
            try:
                return _orig_create_ref(model, *args, **kwargs)
            except (RuntimeError, TypeError) as e:
                if "pickle" in str(e).lower() or "LLMEngine" in str(e):
                    print("[*] Reference model creation skipped (vLLM LLMEngine not picklable).")
                    print("[*] With LoRA, the frozen base model serves as the implicit reference.")
                    return None
                raise
        _grpo_mod.create_reference_model = _safe_create_reference_model
        _tmb.create_reference_model = _safe_create_reference_model

    # Timing probe: measure generate() wall time to distinguish generation vs backward bottleneck
    import time as _time
    _orig_gen = model.generate
    def _timed_generate(*_a, **_kw):
        _t0 = _time.time()
        _out = _orig_gen(*_a, **_kw)
        print(f"[TIMING] generate(): {_time.time()-_t0:.2f}s", flush=True)
        return _out
    model.generate = _timed_generate

    # Also pass ref_model=None directly if the trainer accepts it
    _trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[get_combined_reward_fn(mode=mode, stage_steps=stage_steps, is_resumed=bool(resume_adapter_from))],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[StepTrackerCallback(model), DriveUploadCallback(output_dir)]
    )
    _trainer_params = set(_inspect.signature(GRPOTrainer.__init__).parameters)
    if "ref_model" in _trainer_params:
        _trainer_kwargs["ref_model"] = None

    trainer = GRPOTrainer(**_trainer_kwargs)

    # vLLM PEFT weight sync patch.
    # TRL's GRPOTrainer._prepare_inputs syncs full state_dict to vLLM each step.
    # With PEFT, state_dict keys are prefixed `base_model.model.` and contain
    # `.lora_A.default.weight` / `.lora_B.default.weight` keys that vLLM rejects
    # (ValueError: There is no module or parameter named 'base_model').
    # We intercept load_weights, merge LoRA A/B into the fp16 base (from HF cache
    # — already downloaded by vLLM during init), and only push HF-style merged keys.
    if use_vllm:
        try:
            import torch as _th
            import os as _os
            import glob as _glob
            from safetensors import safe_open as _safe_open
            from huggingface_hub import snapshot_download as _snap_dl

            _sd_dir = _snap_dl(
                _vllm_model_name,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            )
            _sd_files = sorted(_glob.glob(_os.path.join(_sd_dir, "*.safetensors")))
            _fp16_base = {}
            for _f in _sd_files:
                with _safe_open(_f, framework="pt", device="cpu") as _st:
                    for _k in _st.keys():
                        _fp16_base[_k] = _st.get_tensor(_k)
            print(f"[+] Loaded fp16 base: {len(_sd_files)} safetensors, {len(_fp16_base)} tensors")
            _fp16_base_gpu = {}  # lazily promoted: only the ~28 adapted-layer tensors

            _llm_model = trainer.llm.llm_engine.model_executor.driver_worker.model_runner.model
            _orig_load_weights = _llm_model.load_weights
            _lora_scale = 96.0 / 64.0  # lora_alpha / r — must match get_peft_model config
            _step_counter = [0]
            _PREFIXES = ("base_model.model.", "base_model.")

            def _peft_load_weights(weights):
                wd = dict(weights)
                lora_A_map = {}
                lora_B_map = {}
                for name, tensor in wd.items():
                    clean = name
                    for _p in _PREFIXES:
                        if clean.startswith(_p):
                            clean = clean[len(_p):]
                            break
                    if ".lora_A.default.weight" in clean:
                        base_key = clean.replace(".lora_A.default.weight", ".weight")
                        lora_A_map[base_key] = tensor
                    elif ".lora_B.default.weight" in clean:
                        base_key = clean.replace(".lora_B.default.weight", ".weight")
                        lora_B_map[base_key] = tensor

                updates = []
                for base_key, A in lora_A_map.items():
                    if base_key not in lora_B_map or base_key not in _fp16_base:
                        continue
                    B = lora_B_map[base_key]
                    if base_key not in _fp16_base_gpu:
                        _fp16_base_gpu[base_key] = _fp16_base[base_key].to(_th.float16).cuda()
                    base = _fp16_base_gpu[base_key]
                    A_g = A.detach().to(_th.float16).cuda()
                    B_g = B.detach().to(_th.float16).cuda()
                    merged = base + (B_g @ A_g) * _lora_scale
                    updates.append((base_key, merged))

                _step_counter[0] += 1
                if _step_counter[0] <= 3:
                    print(
                        f"[*] vLLM sync #{_step_counter[0]}: "
                        f"{len(updates)} LoRA-merged weights pushed "
                        f"({len(lora_A_map)} A / {len(lora_B_map)} B detected)"
                    )
                if not updates:
                    return
                return _orig_load_weights(iter(updates))

            _llm_model.load_weights = _peft_load_weights
            print("[+] Patched vLLM load_weights — LoRA merge from fp16 cache enabled.")
        except Exception as _e:
            print(f"[!] vLLM PEFT load_weights patch failed: {_e}")
            import traceback as _tb
            _tb.print_exc()

    # Custom LR schedule: warmup 10 → flat peak → gentle decay
    import types
    def custom_create_scheduler(self, num_training_steps: int, optimizer=None):
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(step: int):
            if step < 10:
                return float(step) / 10.0
            elif step < 220:
                return 1.0
            elif step < max_steps:
                return 1.0 - float(step - 220) / float(max_steps - 220)
            else:
                return 0.0
        self.lr_scheduler = LambdaLR(self.optimizer if optimizer is None else optimizer, lr_lambda)
        self._created_lr_scheduler = True
        return self.lr_scheduler
    trainer.create_scheduler = types.MethodType(custom_create_scheduler, trainer)

    # 7. Run LF-GRPO Training Loop
    print(f"[*] Launching LF-GRPO training loop for {max_steps} steps...")
    trainer.train()
    
    # 8. Save LoRA Adapters
    final_save_path = os.path.join(output_dir, "final_lora")
    print(f"[*] Saving final LoRA adapters to {final_save_path}...")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("[+] LF-GRPO Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LF-GRPO 1000-Step Training Pipeline.")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-1.5B-Instruct", help="Base model")
    parser.add_argument("--mode", type=str, choices=["standard", "p-grpo", "step-grpo"], default="step-grpo", help="GRPO Mode")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps")
    parser.add_argument("--stage_steps", type=int, default=100, help="Stage 1 (format priming) steps")
    parser.add_argument("--learning_rate", type=float, default=1.5e-5, help="Learning rate")
    parser.add_argument("--layers_to_transform", type=str, default="last_4", help="Layers to adapt (comma-separated indices or 'last_4')")
    parser.add_argument("--output_dir", type=str, default="./grpo_cot_output", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--limit_train", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt")
    parser.add_argument("--no_vllm", action="store_true", help="Disable vLLM")
    parser.add_argument("--resume_adapter_from", type=str, default=None,
                        help="HF repo or local path to load LoRA adapter weights before training")

    args, _ = parser.parse_known_args()

    run_training_pipeline(
        model_name=args.model_name,
        mode=args.mode,
        max_steps=args.max_steps,
        stage_steps=args.stage_steps,
        learning_rate=args.learning_rate,
        layers_to_transform=args.layers_to_transform,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        limit_train=args.limit_train,
        num_generations=args.num_generations,
        no_vllm=args.no_vllm,
        resume_adapter_from=args.resume_adapter_from,
    )
