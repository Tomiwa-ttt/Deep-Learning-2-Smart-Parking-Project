"""
generate_architecture_diagram.py

Renders the updated two-pipeline system architecture (object detector for
uncalibrated photos + classifier/geometry check for calibrated lots) as a
plain matplotlib diagram, since no graphviz binary is available in this
environment.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

BOX_STYLE = dict(boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.3)


def box(ax, xy, w, h, text, fc="#24282f", ec="#4a5058", tc="white", fontsize=10.5):
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, **BOX_STYLE)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fontsize, weight="medium")
    return (x, y, w, h)


def arrow(ax, a, b, label=None, label_offset=(0.15, 0)):
    (ax0, ay0), (bx0, by0) = a, b
    fa = FancyArrowPatch((ax0, ay0), (bx0, by0), arrowstyle="-|>", mutation_scale=14,
                          color="#8b909c", linewidth=1.3, shrinkA=6, shrinkB=6)
    ax.add_patch(fa)
    if label:
        mx, my = (ax0 + bx0) / 2 + label_offset[0], (ay0 + by0) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="left", va="center", fontsize=9, color="#5a5f68")


fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis("off")

# Path A: general object detector (any photo)
box(ax, (2.4, 10), 4.0, 0.9, "Any lot photo\n(upload, no calibration)", fc="#333842")
box(ax, (2.4, 8.4), 4.0, 0.9, "Object Detector\n16x16 grid, 2 classes\n(empty_spot / occupied_spot)", fc="#3b5bfd")

# Path B: calibrated classifier + geometry
box(ax, (7.6, 10), 4.0, 0.9, "Calibrated lot photo\n(known spot boundaries)", fc="#333842")
box(ax, (7.6, 8.4), 4.0, 0.9, "Per-spot crop (OpenCV)")
box(ax, (7.6, 6.8), 4.0, 0.9, "CNN Occupancy Classifier")
box(ax, (7.6, 5.2), 4.4, 0.9, "Geometry Check\n(IoU vs. marked spot, if occupied)")

# shared
box(ax, (5.0, 3.2), 4.6, 0.9, "REST API (Flask)", fc="#e0a338", tc="#14161a")
box(ax, (5.0, 1.4), 4.6, 0.9, "ParkCheck demo UI\n(upload + sample lots)", fc="#333842")

arrow(ax, (2.4, 9.55), (2.4, 8.85))
arrow(ax, (7.6, 9.55), (7.6, 8.85))
arrow(ax, (7.6, 7.95), (7.6, 7.25))
arrow(ax, (7.6, 6.35), (7.6, 5.65))

arrow(ax, (2.4, 7.95), (5.0, 3.65))
arrow(ax, (7.6, 4.75), (5.0, 3.65))
arrow(ax, (5.0, 2.75), (5.0, 1.85))

plt.tight_layout()
plt.savefig("report_assets/architecture_v2.png", dpi=150, facecolor="white")
print("Saved report_assets/architecture_v2.png")
