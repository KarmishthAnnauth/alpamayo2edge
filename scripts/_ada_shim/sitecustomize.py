"""Cap this process's share of a SHARED GPU. Imported automatically by CPython
when it is on sys.path; only active when `A2E_MEM_FRAC` is set, so it is inert
everywhere except scripts/03b_ada_rehearsal.sh.

The Ada carries other people's long-running work and nothing arbitrates it (no
SLURM). Without a cap, an underestimate of our own peak silently starves them.
With one, we OOM instead - a loud failure that costs us a rehearsal rather than
someone else's 25-day demo server.
"""
import os

_frac = os.environ.get("A2E_MEM_FRAC")
if _frac:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(_frac), 0)
            _t = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"sitecustomize: capped at {float(_frac):.0%} of "
                  f"{_t:.1f} GiB = {float(_frac) * _t:.1f} GiB", flush=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"sitecustomize: memory cap NOT applied ({e})", flush=True)
