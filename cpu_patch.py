"""Make VoxCPM2 run on the CPU: PyTorch's CPU attention needs the attention mask with at least two
dimensions (on CUDA a one-dimensional one works). One line in the installed voxcpm package; it
means the same thing. Run it once after installing or upgrading voxcpm:  python cpu_patch.py"""
import os
import sys

import voxcpm

path = os.path.join(os.path.dirname(voxcpm.__file__), "modules", "minicpm4", "model.py")
src = open(path, encoding="utf-8").read()
MARK = "attn_mask = attn_mask[None, :]"
if MARK in src:
    print(f"Already patched: {path}")
    sys.exit(0)
lines = src.split("\n")
# the patch goes in front of the attention call of forward_step that uses attn_mask
for i, line in enumerate(lines):
    if "def forward_step" in line:
        for j in range(i, len(lines)):
            if "scaled_dot_product_attention" in lines[j] and "attn_mask" in "\n".join(lines[j:j + 8]):
                indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                lines.insert(j, indent + "# CPU attention needs a >=2-D mask (CUDA broadcasts 1-D); same meaning")
                lines.insert(j + 1, indent + MARK)
                break
        break
else:
    sys.exit("forward_step not found: this voxcpm version may not need the patch, or it changed. Not patched.")
if MARK not in "\n".join(lines):
    sys.exit("The attention call was not found where expected. Not patched.")
open(path + ".orig", "w", encoding="utf-8").write(src)
open(path, "w", encoding="utf-8").write("\n".join(lines))
print(f"Patched {path} (original kept as model.py.orig)")
