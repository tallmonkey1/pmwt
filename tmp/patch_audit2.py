"""Patch the audit script to fix the f-string syntax error (lines 105-106)."""
from pathlib import Path

path = Path("/home/user/hutrh/extracted/repo/unpacked/options-engine/audit_simulation_correctness.py")
src = path.read_text()

# Replace the problematic block (use exact characters from the file)
old_lines = [
    '    out.append(AuditResult(',
    '        "Reproducibility (HybridSimulator, same seed)",',
    '        same_spot and same_var,',
    '        f"max |\u0394spot|={float(np.max(np.abs(paths_a.spot - paths_b.spot))):.3e}, "',
    '        f"max |\u0394var|={float(np.max(np.abs(paths_a.variance - paths_b.variance)):.3e)}",',
    '    ))',
]
new_lines = [
    '    delta_spot = float(np.max(np.abs(paths_a.spot - paths_b.spot)))',
    '    delta_var = float(np.max(np.abs(paths_a.variance - paths_b.variance)))',
    '    out.append(AuditResult(',
    '        "Reproducibility (HybridSimulator, same seed)",',
    '        same_spot and same_var,',
    '        f"max |\u0394spot|={delta_spot:.3e}, max |\u0394var|={delta_var:.3e}",',
    '    ))',
]

old_block = "\n".join(old_lines)
new_block = "\n".join(new_lines)

if old_block in src:
    src = src.replace(old_block, new_block)
    path.write_text(src)
    print("PATCHED")
else:
    print("NOT FOUND")
    # Try line-by-line
    for i in range(len(src.split('\n'))):
        chunk = "\n".join(src.split('\n')[i:i+6])
        if old_block in chunk:
            print(f"Found at line {i+1}")
            break
