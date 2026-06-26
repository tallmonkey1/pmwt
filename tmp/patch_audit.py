"""Patch the audit script to fix the f-string syntax error."""
from pathlib import Path

path = Path("/home/user/hutrh/extracted/repo/unpacked/options-engine/audit_simulation_correctness.py")
src = path.read_text()
old = """    out.append(AuditResult(
        "Reproducibility (HybridSimulator, same seed)",
        same_spot and same_var,
        f\"max |\\u0394spot|={float(np.max(np.abs(paths_a.spot - paths_b.spot))):.3e}, \"
        f\"max |\\u0394var|={float(np.max(np.abs(paths_a.variance - paths_b.variance)):.3e)}\",
    ))"""
new = """    delta_spot = float(np.max(np.abs(paths_a.spot - paths_b.spot)))
    delta_var = float(np.max(np.abs(paths_a.variance - paths_b.variance)))
    out.append(AuditResult(
        "Reproducibility (HybridSimulator, same seed)",
        same_spot and same_var,
        f\"max |\\u0394spot|={delta_spot:.3e}, max |\\u0394var|={delta_var:.3e}\",
    ))"""
if old in src:
    src = src.replace(old, new)
    path.write_text(src)
    print("PATCHED")
else:
    print("NOT FOUND")
