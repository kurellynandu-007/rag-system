"""
RAG Environment Verification Script
Runs inside .venv -- checks interpreter, imports, versions,
and encodes one test sentence on CPU.
Does NOT ingest documents or start Streamlit.
"""
import io, sys
# Force UTF-8 stdout so Unicode does not crash on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import math

# ── 1. Interpreter check ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Interpreter")
print(f"  Executable : {sys.executable}")
print(f"  Python     : {sys.version.split()[0]}")
in_venv = (
    hasattr(sys, "real_prefix")
    or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
)
print(f"  In venv    : {in_venv}")
if not in_venv:
    print("  WARNING: Not running inside a virtual environment!")

# ── 2. Memory snapshot helper ─────────────────────────────────────────────────
def mem_gb():
    """Return (used_pct, free_gb) via WMI — Windows only."""
    try:
        import subprocess, re
        out = subprocess.check_output(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_OperatingSystem) | "
             "Select-Object TotalVisibleMemorySize,FreePhysicalMemory | "
             "ConvertTo-Json"],
            stderr=subprocess.DEVNULL, timeout=10
        ).decode()
        import json
        d = json.loads(out)
        total = d["TotalVisibleMemorySize"]
        free  = d["FreePhysicalMemory"]
        used_pct = round((total - free) / total * 100, 1)
        free_gb  = round(free / 1_048_576, 2)
        return used_pct, free_gb
    except Exception:
        return None, None

used_pct, free_gb = mem_gb()
print(f"  RAM before imports: {used_pct}% used, {free_gb} GB free")
print()

# ── 3. Package imports & versions ────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Imports & Versions")

results = {}

def try_import(label, module_name, version_attr="__version__"):
    try:
        mod = __import__(module_name)
        ver = getattr(mod, version_attr, "unknown")
        print(f"  [OK] {label:<25} version={ver}")
        results[label] = ("OK", ver)
        return mod
    except Exception as e:
        print(f"  [FAIL] {label:<23} ERROR: {e}")
        results[label] = ("FAIL", str(e))
        return None

chromadb_mod   = try_import("chromadb",            "chromadb")
st_mod         = try_import("sentence_transformers","sentence_transformers")
pypdf_mod      = try_import("pypdf",               "pypdf")
google_mod     = try_import("google.genai",        "google.genai", "version") or \
                 try_import("google.genai",        "google.genai", "__version__")
streamlit_mod  = try_import("streamlit",           "streamlit")
dotenv_mod     = try_import("dotenv",              "dotenv")

print()

# ── 4. Check torch (CPU) ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: PyTorch device check")
try:
    import torch
    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}  (expected: False)")
    device = "cpu"
    print(f"  Using device  : {device}")
except Exception as e:
    print(f"  [FAIL] torch import: {e}")
    sys.exit(1)
print()

# ── 5. Load embedding model once, encode one sentence ────────────────────────
print("=" * 60)
print("STEP 4: Load SentenceTransformer (all-MiniLM-L6-v2) on CPU")
print("  (first run downloads ~90 MB to HuggingFace cache)")

if st_mod is None:
    print("  [SKIP] sentence_transformers failed to import — cannot test embedding.")
    sys.exit(1)

used_pct, free_gb = mem_gb()
print(f"  RAM before model load: {used_pct}% used, {free_gb} GB free")
if free_gb is not None and free_gb < 0.8:
    print(f"  [ABORT] Only {free_gb} GB free — memory too low to safely load model.")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = "all-MiniLM-L6-v2"
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"  Model loaded: {MODEL_NAME}")
except Exception as e:
    print(f"  [FAIL] Model load error: {e}")
    sys.exit(1)

used_pct, free_gb = mem_gb()
print(f"  RAM after model load : {used_pct}% used, {free_gb} GB free")
print()

# ── 6. Encode one test sentence ───────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Encode one test sentence (batch_size=8)")

TEST_SENTENCE = "This is a lightweight RAG system test on CPU."
BATCH_SIZE = 8  # configured for later embedding jobs

try:
    import numpy as np
    vector = model.encode(
        TEST_SENTENCE,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        device="cpu",
    )
    dim = vector.shape[0]
    all_finite = bool(np.all(np.isfinite(vector)))
    sample = [round(float(x), 6) for x in vector[:5]]

    print(f"  Sentence   : {TEST_SENTENCE!r}")
    print(f"  Dimension  : {dim}")
    print(f"  All finite : {all_finite}")
    print(f"  First 5 values: {sample}")

    if not all_finite:
        print("  [FAIL] Vector contains NaN or Inf values!")
        sys.exit(1)
    if dim == 0:
        print("  [FAIL] Zero-length vector returned!")
        sys.exit(1)

    print("  [OK] Embedding test PASSED")
except Exception as e:
    print(f"  [FAIL] Encoding error: {e}")
    sys.exit(1)

# ── 7. Final memory snapshot ──────────────────────────────────────────────────
print()
print("=" * 60)
used_pct, free_gb = mem_gb()
print(f"RAM after encode: {used_pct}% used, {free_gb} GB free")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
all_ok = True
for label, (status, detail) in results.items():
    mark = "[PASS]" if status == "OK" else "[FAIL]"
    print(f"  {mark} {label:<25} {status}  ({detail})")
    if status != "OK":
        all_ok = False

print()
if all_ok:
    print("  ALL IMPORT CHECKS PASSED")
else:
    print("  SOME IMPORTS FAILED — see details above")

print(f"  Embedding model : {MODEL_NAME}  dim={dim}  finite={all_finite}")
print(f"  Batch size hint : {BATCH_SIZE}  (reuse model object, don't reload)")
print("=" * 60)
print("Verification complete. No documents ingested. Streamlit not started.")
