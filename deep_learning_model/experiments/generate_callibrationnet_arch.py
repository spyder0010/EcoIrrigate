"""
CalibrationNet 3D Architecture Diagram
========================================

Generates a publication-quality pseudo-3D architecture diagram of the
CalibrationNet MLP model used for sensor calibration in EcoIrrigate.

Output
------
  results/figures/fig_calibrationnet_3d.png  — 300 DPI PNG

Usage
-----
    python experiments/generate_callibrationnet_arch.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── colour helpers ─────────────────────────────────────────────────────────────
def rgb(r, g, b): return (r/255, g/255, b/255)

C_INPUT  = rgb( 30,144,255)   # dodger-blue  – Input
C_DENSE  = rgb( 25,100,210)   # mid-blue     – Dense 256/128/64
C_OUTPUT = rgb( 32,178,120)   # teal-green   – Output + Moisture
C_FARM   = rgb(180, 30, 80)   # crimson      – Farm embedding
C_TOP    = rgb( 90,170,255)   # light-blue   – 3-D top face (blue boxes)
C_TOP_G  = rgb( 60,200,150)   # light-green  – 3-D top face (output box)
C_SIDE   = rgb( 15, 70,170)   # dark-blue    – 3-D right face (blue boxes)
C_SIDE_G = rgb( 18,120, 88)   # dark-green   – 3-D right face (output box)
C_FARM_S = rgb(130, 20, 60)   # dark crimson – farm right face
C_BG     = rgb(245,248,252)   # near-white   – background


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE: 3-D box
# Draws three faces (right side, top, front) to simulate depth.
# Labels are placed as fractions of box height so they never overflow.
# ══════════════════════════════════════════════════════════════════════════════
def box3d(ax, x, y, w, h, dx=0.3, dy=0.2,
          face_c=C_DENSE, top_c=C_TOP, side_c=C_SIDE,
          label="", sublabel="", lfs=11, sfs=8.5, label_color="white"):
    """
    x, y    : bottom-left corner of the FRONT face
    w, h    : width and height of the front face
    dx, dy  : 3-D offset (right/up) for depth illusion
    face_c  : front face colour
    top_c   : top face colour  (lighter shade)
    side_c  : right face colour (darker shade)
    label   : bold primary text (layer name)
    sublabel: secondary text (e.g. "BatchNorm")
    lfs/sfs : label / sublabel font sizes
    """
    # 1. Drop-shadow
    ax.add_patch(mpatches.FancyBboxPatch(
        (x + 0.06, y - 0.06), w, h,
        boxstyle="round,pad=0.015",
        facecolor=rgb(160, 160, 160), edgecolor="none",
        alpha=0.25, zorder=1))

    # 2. Right side face (parallelogram)
    xs = [x+w,    x+w+dx, x+w+dx, x+w]
    ys = [y,      y+dy,   y+h+dy, y+h]
    ax.fill(xs, ys, color=side_c, zorder=2)
    ax.plot(xs + [xs[0]], ys + [ys[0]], color="white", lw=0.4, zorder=3)

    # 3. Top face (parallelogram)
    xt = [x,   x+dx,   x+w+dx, x+w]
    yt = [y+h, y+h+dy, y+h+dy, y+h]
    ax.fill(xt, yt, color=top_c, zorder=2)
    ax.plot(xt + [xt[0]], yt + [yt[0]], color="white", lw=0.4, zorder=3)

    # 4. Front face (rounded rectangle)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015",
        facecolor=face_c, edgecolor="white",
        linewidth=1.0, zorder=2))

    # 5. Labels — fractional positioning prevents overflow at any box size
    cx_, cy_ = x + w / 2, y + h / 2
    ax.text(cx_, cy_ + h * 0.12, label,
            ha="center", va="center",
            fontsize=lfs, fontweight="bold",
            color=label_color, zorder=5)
    ax.text(cx_, cy_ - h * 0.16, sublabel,
            ha="center", va="center",
            fontsize=sfs, color=label_color,
            alpha=0.88, linespacing=1.4, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# ARROW HELPERS
# ══════════════════════════════════════════════════════════════════════════════
ARROW_PROPS = dict(arrowstyle="->", color="#333",
                   lw=1.5, mutation_scale=18, zorder=6)

def harrow(ax, x0, x1, y, **kw):
    """Straight horizontal arrow."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(**{**ARROW_PROPS, **kw}))

def arrow_diag(ax, x0, y0=None, x1=None, y1=None, **kw):
        """Draw an arrow along a straight or multi-segment (L-shaped) path.

        Usage:
          arrow_diag(ax, x0, y0, x1, y1, **kw)
                - legacy single-segment call (keeps backwards compatibility)

          arrow_diag(ax, [(x0,y0),(x1,y1),(x2,y2),...], **kw)
                - draw a polyline through the provided points and place an
                  arrowhead on the final segment. Useful for L-shaped or
                  multi-point arrows.

        Additional kwargs are forwarded to `arrowprops` (e.g. color, lw).
        """
        # Build points list from either a single iterable or four coords
        if isinstance(x0, (list, tuple)) and y0 is None:
                pts = list(x0)
        else:
                pts = [(x0, y0), (x1, y1)]

        # Draw intermediate segments (no arrowheads)
        if len(pts) > 1:
                for i in range(len(pts) - 1):
                        x_start, y_start = pts[i]
                        x_end, y_end = pts[i + 1]
                        # For the last segment, draw an annotated arrow; otherwise a simple line
                        if i < len(pts) - 2:
                                ax.plot([x_start, x_end], [y_start, y_end],
                                                color=kw.get('color', ARROW_PROPS.get('color')),
                                                lw=kw.get('lw', ARROW_PROPS.get('lw')),
                                                zorder=kw.get('zorder', ARROW_PROPS.get('zorder')),
                                                solid_capstyle='round')
                        else:
                                ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                                                        arrowprops=dict(**{**ARROW_PROPS, **kw}))
        else:
                # Degenerate: single point — nothing to draw
                return


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# X positions are computed incrementally so boxes never collide.
# ══════════════════════════════════════════════════════════════════════════════
DX, DY = 0.30, 0.20        # 3-D depth offsets

# Box dimensions (width × height)
BW_IN,  BH_IN  = 1.50, 2.60
BW_256, BH_256 = 1.80, 3.00
BW_128, BH_128 = 1.65, 2.65
BW_64,  BH_64  = 1.50, 2.30
BW_OUT, BH_OUT = 1.30, 1.90
BW_MOI, BH_MOI = 1.20, 1.00
BW_FM,  BH_FM  = 1.30, 1.30

GAP = 0.30    # horizontal gap between box right edge and next arrow

Y0 = 2.60     # Y-baseline of the main row
FY = Y0 - 1.80  # Farm box top (below input column)

# X positions — incremental so no overlap possible
X_IN  = 0.50
X_256 = X_IN  + BW_IN  + DX + GAP + 0.55
X_128 = X_256 + BW_256 + DX + GAP + 0.40
X_64  = X_128 + BW_128 + DX + GAP + 0.35
X_OUT = X_64  + BW_64  + DX + GAP + 0.35
X_MOI = X_OUT + BW_OUT + DX + GAP + 0.40

# Centre-y of each main box
CY_IN  = Y0 + BH_IN  / 2
CY_256 = Y0 + BH_256 / 2
CY_128 = Y0 + BH_128 / 2
CY_64  = Y0 + BH_64  / 2
CY_OUT = Y0 + BH_OUT / 2


# ══════════════════════════════════════════════════════════════════════════════
# CANVAS — auto-fitted to content (zero wasted whitespace)
# ══════════════════════════════════════════════════════════════════════════════
RIGHT = X_MOI + BW_MOI + DX + 0.30
TOP   = Y0 + BH_256 + DY + 0.90    # title + subtitle
BOT   = FY - 0.65                   # footnote

# global downward shift (data-units). Increase to move everything further down.
SHIFT_DOWN = 1.0

SCALE = 0.78   # data-units → inches (controls overall figure size)
fig, ax = plt.subplots(figsize=((RIGHT - 0.20) * SCALE,
                                (TOP   - BOT)   * SCALE))
ax.set_facecolor(C_BG)
fig.patch.set_facecolor(C_BG)
ax.set_xlim(0.20, RIGHT)
# Keep the original bottom so small elements (Farm) remain visible,
# and extend the top by SHIFT_DOWN so the whole scene appears shifted down.
ax.set_ylim(BOT, TOP + SHIFT_DOWN)
ax.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# DRAW BOXES
# ══════════════════════════════════════════════════════════════════════════════
box3d(ax, X_IN, Y0, BW_IN, BH_IN, dx=DX, dy=DY,
      face_c=C_INPUT, top_c=C_TOP, side_c=C_SIDE,
      label="Input", sublabel="A2: 2 feat\nA6: 6 feat", lfs=11, sfs=8.5)

box3d(ax, X_256, Y0, BW_256, BH_256, dx=DX, dy=DY,
      label="Dense 256", sublabel="BatchNorm", lfs=10, sfs=9)

box3d(ax, X_128, Y0, BW_128, BH_128, dx=DX, dy=DY,
      label="Dense 128", sublabel="BatchNorm", lfs=10, sfs=9)

box3d(ax, X_64, Y0, BW_64, BH_64, dx=DX, dy=DY,
      label="Dense 64", sublabel="BatchNorm", lfs=10, sfs=9)

box3d(ax, X_OUT, Y0, BW_OUT, BH_OUT, dx=DX, dy=DY,
      face_c=C_OUTPUT, top_c=C_TOP_G, side_c=C_SIDE_G,
      label="Output", sublabel="1 unit", lfs=11, sfs=8.5)

box3d(ax, X_IN, FY, BW_FM, BH_FM, dx=0.22, dy=0.16,
      face_c=C_FARM, top_c=rgb(210, 60, 100), side_c=C_FARM_S,
      label="Farm", sublabel="Embed 16d", lfs=11, sfs=8.5)


# ══════════════════════════════════════════════════════════════════════════════
# MOISTURE % OUTPUT LABEL (flat teal-bordered box)
# ══════════════════════════════════════════════════════════════════════════════
MX = X_MOI
MY = Y0 + (BH_OUT - BH_MOI) / 2    # vertically centred with Output box
ax.add_patch(mpatches.FancyBboxPatch(
    (MX, MY), BW_MOI, BH_MOI,
    boxstyle="round,pad=0.06",
    facecolor=C_BG, edgecolor=C_OUTPUT, linewidth=2.0, zorder=4))
ax.text(MX + BW_MOI/2, MY + BH_MOI*0.65, "Moisture",
        ha="center", va="center",
        fontsize=11, fontweight="bold", color=C_OUTPUT, zorder=5)
ax.text(MX + BW_MOI/2, MY + BH_MOI*0.28, "%",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color=C_OUTPUT, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN-FLOW ARROWS (straight, horizontal)
# ══════════════════════════════════════════════════════════════════════════════
harrow(ax, X_IN  + BW_IN  + GAP*0.5,       X_256 - GAP*0.5,        CY_IN)
harrow(ax, X_256 + BW_256 + GAP*0.5,       X_128 - GAP*0.5,        CY_256)
harrow(ax, X_128 + BW_128 + GAP*0.5,       X_64  - GAP*0.5,        CY_128)
harrow(ax, X_64  + BW_64  + GAP*0.5,       X_OUT - GAP*0.5,        CY_64)
harrow(ax, X_OUT + BW_OUT + DX + GAP*0.5,  MX    - GAP*0.5,        CY_OUT)


# ══════════════════════════════════════════════════════════════════════════════
# FARM EMBEDDING MERGE  (diagonal arrow + ⊕ circle)
# ══════════════════════════════════════════════════════════════════════════════
PX = X_IN + BW_FM * 0.65    # ⊕ x — aligned inside Input box base
PY = Y0 - 0.45              # ⊕ y — just below Input box





# ⊕ → Dense 256 (straight L-shaped segments)
# Exit from near the ⊕, travel right to the farm box's right side,
# then vertically to the level of the Dense 256 entry, then right to target.
arrow_diag(ax, [
        (PX + 0.5, PY + 0.05 - 0.5),
        (X_IN + BW_FM + 0.08 +0.6, PY + 0.05 - 0.45),          # right side of farm box (slightly offset)
        (X_IN + BW_FM + 0.08 +0.6, CY_IN),              # move vertically to Dense 256 lane
        (X_256 - GAP*0.5, CY_IN)                   # enter Dense 256 left face
], color="#333", lw=1.5, mutation_scale=18, zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
# DIMENSION ANNOTATIONS (above top face)
# ══════════════════════════════════════════════════════════════════════════════
def dim_lbl(x_left, bh, txt):
    ax.text(x_left + 0.06, Y0 + bh + DY + 0.08, txt,
            fontsize=8, color=rgb(110, 110, 110),
            ha="left", va="bottom", zorder=6)

dim_lbl(X_256, BH_256, "→22 or 18→")
dim_lbl(X_128, BH_128, "→256→")
dim_lbl(X_64,  BH_64,  "→128→")
dim_lbl(X_OUT, BH_OUT, "→64→")
dim_lbl(X_MOI, BH_MOI, "→1→")


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION LABELS (teal-bordered, below dense boxes)
# ══════════════════════════════════════════════════════════════════════════════
ACT_Y = Y0 - 0.20

def act_lbl(cx):
    ax.text(cx, ACT_Y, "ReLU + Dropout(0.3)",
            ha="center", va="top", fontsize=8, color=C_OUTPUT, zorder=6,
            bbox=dict(boxstyle="round,pad=0.30",
                      facecolor=C_BG, edgecolor=C_OUTPUT, lw=0.9))

act_lbl(X_256 + BW_256 / 2)
act_lbl(X_128 + BW_128 / 2)
act_lbl(X_64  + BW_64  / 2)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE & SUBTITLE
# ══════════════════════════════════════════════════════════════════════════════
MID     = (0.20 + RIGHT) / 2
TITLE_Y = Y0 + BH_256 + DY + 0.62 + 0.5

ax.text(MID, TITLE_Y,
        "CalibrationNet Architecture",
        ha="center", fontsize=17, fontweight="bold",
        color=rgb(20, 20, 20), zorder=6)
ax.text(MID, TITLE_Y - 0.32,
        "Sensor Calibration: Raw ADC → Volumetric Moisture %  |  ~50K Parameters",
        ha="center", fontsize=9, color=rgb(100, 100, 100), zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTNOTE (training config)
# ══════════════════════════════════════════════════════════════════════════════
ax.text(MID, FY - 0.42,
        "Loss: Huber (δ=1.0)  |  Optimizer: AdamW (lr=1e-3, wd=1e-4)  |  Cosine Annealing",
        ha="center", fontsize=8.5, color=rgb(70, 70, 70), zorder=6,
        bbox=dict(boxstyle="round,pad=0.40",
                  facecolor=C_BG, edgecolor=rgb(175, 175, 175), lw=1.0))


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(fig_dir, exist_ok=True)

plt.savefig(os.path.join(fig_dir, 'fig_calibrationnet_3d.png'), dpi=300, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("    Done.")
