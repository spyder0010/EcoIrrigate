"""
================================================================================
EcoIrrigate: End-to-End System Architecture Figure
================================================================================
Purpose
-------
Generates a publication-quality pseudo-3D system architecture diagram of the
EcoIrrigate smart irrigation framework, following Elsevier journal figure
standards (DejaVu Sans font, 300 DPI, tight bounding box, CMYK-safe colours).

Output
------
  ecoirrigate_final_v2.png         — 300 DPI PNG ready for journal submission
  ecoirrigate_final_v2.png.meta.json — accessibility / caption metadata

Architecture overview (left → right, top → bottom)
---------------------------------------------------
  IoT Sensors ──► Arduino R4 ──► METAR (Pressure)
          │
          ▼
  Data Pipeline (Cleaning → Feature Eng. → 92 Features → Temporal Split)
          │
          ▼
  ┌──────────────────────────────────────────────────┐
  │  CalibrationNet │ ForecastingNet │ MultiTaskNet  │
  └──────────┬─────────────┬────────────────┬────────┘
             │             │                │
        Calibration      Forecast       (feeds both)
          Output          Output
             │             │
             └──────┬───────┘
                    ▼
         Irrigation Decision Engine ──► Smart Irrigation

Dependencies: matplotlib >= 3.5, colorsys (stdlib), json (stdlib)
================================================================================
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import colorsys, json


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ELSEVIER PUBLICATION TYPOGRAPHY
# ------------------------------------------------------------------------------
# Elsevier requires a sans-serif typeface (Arial/Helvetica preferred;
# DejaVu Sans is the closest freely available substitute), minimum 6 pt
# in-figure text, and 300 DPI for raster PNG/TIFF submissions.
# Setting rcParams globally means every text call inherits these values.
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   18,
    'figure.dpi':  300,
    'savefig.dpi': 300,
    'text.color':  '#111111',
})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COLOUR UTILITY FUNCTIONS
# ------------------------------------------------------------------------------
# The pseudo-3-D effect is achieved by drawing three faces per box:
#   Front face  — full component colour
#   Top face    — lightened variant (simulates overhead illumination)
#   Right face  — darkened variant (simulates shadow)
# HLS space is used (not RGB mixing) to preserve hue and saturation.
# ══════════════════════════════════════════════════════════════════════════════
def _lighten(hex_col, a=0.32):
    """Increase HLS Lightness by *a* to simulate a lit top face."""
    h = hex_col.lstrip('#')
    r,g,b = [int(h[i:i+2],16)/255 for i in (0,2,4)]
    hh,l,s = colorsys.rgb_to_hls(r,g,b)
    r2,g2,b2 = colorsys.hls_to_rgb(hh, min(1.0,l+a), s)
    return '#{:02x}{:02x}{:02x}'.format(int(r2*255),int(g2*255),int(b2*255))

def _darken(hex_col, a=0.28):
    """Decrease HLS Lightness by *a* to simulate a shadowed right face."""
    h = hex_col.lstrip('#')
    r,g,b = [int(h[i:i+2],16)/255 for i in (0,2,4)]
    hh,l,s = colorsys.rgb_to_hls(r,g,b)
    r2,g2,b2 = colorsys.hls_to_rgb(hh, max(0.0,l-a), s)
    return '#{:02x}{:02x}{:02x}'.format(int(r2*255),int(g2*255),int(b2*255))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COLOUR PALETTE
# ------------------------------------------------------------------------------
# Each key maps to one architecture component. Colours are:
#   (a) visually distinct across all 10 components
#   (b) dark enough to render white text at WCAG AA contrast
#   (c) distinguishable in deuteranopia/protanopia (no two adjacent boxes
#       share the same hue family)
# ══════════════════════════════════════════════════════════════════════════════
C = dict(
    iot      = '#5C2E0E',  # dark brown  — IoT hardware layer
    arduino  = '#3B5268',  # slate blue  — microcontroller / firmware
    metar    = '#1B8080',  # teal        — external weather data source
    pipeline = '#3A7D44',  # forest green — data processing pipeline
    calib_nn = '#1565C0',  # royal blue  — CalibrationNet (MLP)
    fore_nn  = '#6A1B9A',  # deep purple — ForecastingNet (BiLSTM)
    multi_nn = '#BF360C',  # burnt orange — MultiTaskNet (joint)
    calib_o  = '#00695C',  # dark teal   — calibration output / metrics
    fore_o   = '#4527A0',  # indigo      — forecast output / metrics
    decision = '#1B5E20',  # dark green  — irrigation decision engine
)
WHITE = '#ffffff'  # text on coloured boxes
DARK  = '#2a2a2a'  # arrow lines — softer than pure black for print
BG    = '#EFEFEF'  # light grey canvas — avoids harsh white in print


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GEOMETRY CONSTANTS & BOTTOM-UP CANVAS SIZING
# ------------------------------------------------------------------------------
# All units are Matplotlib data-units (= inches at the given figsize).
# D_EX / S_EX control 3-D extrusion; equal values give a 45° viewing angle.
#
# Canvas height H_TOT is derived bottom-up from band heights + gaps so
# there is NEVER empty whitespace. Adding/removing a band auto-reflows.
#
#   BDY  = PAD                          ← badges sit at canvas bottom
#   R2Y  = BDY + BDH + VGAP            ← neural-net row above badges
#   PPY  = R2Y + BH2 + D_EX + VGAP     ← pipeline above NN row
#   R1Y  = PPY + PPH + D_EX + VGAP     ← input row above pipeline
#   TY   = R1Y + BH1 + D_EX + VGAP     ← title above input row
#   H_TOT= TY  + 1.55                  ← title text needs ~1.5 units headroom
# ══════════════════════════════════════════════════════════════════════════════
D_EX=0.28; S_EX=0.28
PAD=0.55;  VGAP=0.60
BDH=1.20;  BDW=4.20
BH2=1.75;  BW2=4.10
PPH=1.18;  PPW=17.20; PPX=0.50
BH1=1.62;  BW1=4.00

BDY  = PAD
R2Y  = BDY + BDH  + VGAP
PPY  = R2Y + BH2  + D_EX + VGAP
R1Y  = PPY + PPH  + D_EX + VGAP
TY   = R1Y + BH1  + D_EX + VGAP
H_TOT= TY  + 1.55
W_TOT= 30.0

IX=0.50; AX=5.30; MX=10.10
CNX=0.50; FNX=5.40; MNX=10.30
OX=19.20; OW=4.50; OH=BH2
COY=PPY; FOY=R2Y                     # output boxes align with pipeline/NN rows
DEX=24.50; DEW=3.80
DEY=FOY; DEH=(COY+OH+D_EX)-FOY      # decision engine spans both output rows
SIX=DEX+DEW+S_EX+1.05


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CANVAS INITIALISATION
# ------------------------------------------------------------------------------
# A single Axes fills the Figure. Spines/ticks are off (this is a diagram).
# xlim/ylim anchor the coordinate system to the geometry constants above.
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(W_TOT, H_TOT))
ax.set_xlim(0,W_TOT); ax.set_ylim(0,H_TOT)
ax.axis('off')
ax.set_facecolor(BG); fig.patch.set_facecolor(BG)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DRAWING PRIMITIVE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def box3d(x, y, w, h, fc, d=D_EX, s=S_EX, lw=1.1, zo=3):
    """
    Pseudo-3-D box: three Polygon patches (front, top, right).
    Returns (cx, cy) — front-face centre for label placement.
    """
    for verts, fc2 in [
        ([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],              fc),
        ([[x,y+h],[x+w,y+h],[x+w+s,y+h+d],[x+s,y+h+d]], _lighten(fc)),
        ([[x+w,y],[x+w,y+h],[x+w+s,y+h+d],[x+w+s,y+d]], _darken(fc)),
    ]:
        ax.add_patch(mpatches.Polygon(verts, closed=True,
                     facecolor=fc2, edgecolor='#111111', lw=lw, zorder=zo))
    return x+w/2, y+h/2

def label(cx, cy, l1, l2='', fs1=20.0, fs2=15.0, zo=5):
    """
    Centred white label (bold title + optional subtitle) inside a box.
    Two-line layout: l1 sits +0.22 above centre; l2 sits −0.25 below.
    """
    if l2:
        ax.text(cx, cy+0.22, l1, ha='center', va='center',
                fontsize=fs1, fontweight='bold', color=WHITE, zorder=zo)
        ax.text(cx, cy-0.25, l2, ha='center', va='center',
                fontsize=fs2, color=WHITE, zorder=zo)
    else:
        ax.text(cx, cy, l1, ha='center', va='center',
                fontsize=fs1, fontweight='bold', color=WHITE, zorder=zo)

def elbow(pts, col=DARK, lw=2.0, ms=17):
    """
    Orthogonal (Manhattan-routed) multi-segment arrow through waypoints.
    All segments except the last are plain lines; only the final segment
    carries an arrowhead. This prevents mid-route arrowheads and guarantees
    no curved artefacts when routing around box boundaries.
    pts: list of (x,y) — arrow tip is at pts[-1].
    """
    for i in range(len(pts)-2):
        ax.plot([pts[i][0],pts[i+1][0]], [pts[i][1],pts[i+1][1]],
                color=col, lw=lw, solid_capstyle='round', zorder=7)
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                mutation_scale=ms,
                                connectionstyle='arc3,rad=0'), zorder=8)

def straight(x1,y1,x2,y2, col=DARK, lw=2.0, ms=17):
    """Single straight arrow for unobstructed horizontal connections."""
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                mutation_scale=ms,
                                connectionstyle='arc3,rad=0'), zorder=8)

def badge(x, y, w, h, t1, t2, col):
    """
    Rounded stat badge (white fill, coloured border).
    White fill + coloured text ensures greyscale legibility.
    t1 = primary stat (bold), t2 = descriptor label.
    """
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.10',
                  facecolor=WHITE, edgecolor=col, lw=2.0, zorder=7))
    ax.text(x+w/2, y+h*0.65, t1, ha='center', va='center',
            fontsize=24.0, fontweight='bold', color=col, zorder=8)
    ax.text(x+w/2, y+h*0.28, t2, ha='center', va='center',
            fontsize=18.0, color=col, zorder=8)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TITLE & SUBTITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(W_TOT/2, TY+1.10, 'EcoIrrigate: End-to-End System Architecture',
        ha='center', va='center', fontsize=26, fontweight='bold',
        color='#111111', zorder=9)
ax.text(W_TOT/2, TY+0.48, 'From IoT Sensor Data to Intelligent Irrigation Decisions',
        ha='center', va='center', fontsize=18, color='#444444', zorder=9)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ROW 1: INPUT DATA SOURCES
# ------------------------------------------------------------------------------
# IoT Sensors → Arduino R4 → METAR. Straight arrows connect them left-to-right.
# A vertical elbow descends from Arduino (the data aggregation node) to the
# top of the pipeline bar.
# ══════════════════════════════════════════════════════════════════════════════
cx_i,cy_i = box3d(IX,R1Y,BW1,BH1,C['iot'])
label(cx_i,cy_i,'IoT Sensors','Capacitive + Temp')
cx_a,cy_a = box3d(AX,R1Y,BW1,BH1,C['arduino'])
label(cx_a,cy_a,'Arduino R4','ESP32-S3')
cx_m,cy_m = box3d(MX,R1Y,BW1,BH1,C['metar'])
label(cx_m,cy_m,'METAR','Pressure Data')

straight(IX+BW1+S_EX+0.06, R1Y+BH1/2+0.08, AX, R1Y+BH1/2+0.08)
straight(AX+BW1+S_EX+0.06, R1Y+BH1/2+0.08, MX, R1Y+BH1/2+0.08)
elbow([(AX+BW1/2, R1Y),(AX+BW1/2, PPY+PPH+D_EX+0.10)])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — DATA PIPELINE BAR
# ------------------------------------------------------------------------------
# Four sequential preprocessing stages: Cleaning → Feature Engineering →
# 92 Features → Temporal Split (chronological split prevents data leakage).
# One wide bar conveys that all three NNs share the same processed dataset.
# ══════════════════════════════════════════════════════════════════════════════
cx_pp,cy_pp = box3d(PPX,PPY,PPW,PPH,C['pipeline'])
ax.text(cx_pp, cy_pp+0.07,
        'Data Pipeline:  Cleaning  →  Feature Engineering  →  92 Features  →  Temporal Split',
        ha='center', va='center', fontsize=18.0, fontweight='bold', color=WHITE, zorder=5)
elbow([(PPX+PPW/2, PPY),(PPX+PPW/2, R2Y+BH2+D_EX+0.10)])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ROW 2: NEURAL NETWORK MODELS
# ------------------------------------------------------------------------------
# CalibrationNet  — MLP (256→128→64): maps raw sensor readings to calibrated
#                   values using reference measurement pairs.
# ForecastingNet  — BiLSTM + Attention: multi-step soil moisture forecasting
#                   from 1h to 24h ahead.
# MultiTaskNet    — Joint model performing both calibration and forecasting in
#                   a shared representation (task-synergy experiment).
# All three share the same y-baseline to emphasise they are parallel variants.
# ══════════════════════════════════════════════════════════════════════════════
cx_cn,cy_cn = box3d(CNX,R2Y,BW2,BH2,C['calib_nn'])
label(cx_cn,cy_cn,'CalibrationNet','MLP  256 → 128 → 64')
cx_fn,cy_fn = box3d(FNX,R2Y,BW2,BH2,C['fore_nn'])
label(cx_fn,cy_fn,'ForecastingNet','BiLSTM + Attn')
cx_mn,cy_mn = box3d(MNX,R2Y,BW2,BH2,C['multi_nn'])
label(cx_mn,cy_mn,'MultiTaskNet','Joint Calib + Forecast')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — OUTPUT METRIC BOXES
# ------------------------------------------------------------------------------
# Calibration box — R² for sensor arrays A2 (0.956) and A6 (0.912).
#                   Placed at pipeline level (COY=PPY): calibration is an
#                   intermediate processing step before irrigation decisions.
# Forecast box    — 1h→24h horizon; 1.7× performance degradation at 24h.
#                   Placed at NN level (FOY=R2Y): it is a direct model output.
# ══════════════════════════════════════════════════════════════════════════════
cx_co,cy_co = box3d(OX,COY,OW,OH,C['calib_o'])
label(cx_co,cy_co,'Calibration','R²=0.956 (A2)   R²=0.912 (A6)')
cx_fo,cy_fo = box3d(OX,FOY,OW,OH,C['fore_o'])
label(cx_fo,cy_fo,'Forecast','1 h → 24 h    1.7× degradation')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — DECISION ENGINE + SMART IRRIGATION TERMINAL LABEL
# ------------------------------------------------------------------------------
# Decision Engine height spans from bottom of Forecast output to top of
# Calibration output — visually shows it consumes both streams.
# '✶' + plain text for the terminal output avoids over-weighting an outcome
# node with the same visual mass as a processing component.
# ══════════════════════════════════════════════════════════════════════════════
cx_de,cy_de = box3d(DEX,DEY,DEW,DEH,C['decision'],d=0.30,s=0.30)
label(cx_de,cy_de,'Irrigation','Decision Engine',fs1=24.0,fs2=18.0)
ax.text(SIX, cy_de+0.26, '✶', ha='center', va='center',
        fontsize=26, color='#1B5E20', fontweight='bold', zorder=9)
ax.text(SIX, cy_de-0.44, 'Smart\nIrrigation', ha='center', va='center',
        fontsize=20, color='#1B5E20', fontweight='bold',
        linespacing=1.45, zorder=9)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — ARROW ROUTING: NNs → OUTPUT BOXES
# ------------------------------------------------------------------------------
# ROUTING STRATEGY OVERVIEW
# --------------------------
# Two different exit strategies are used depending on the model:
#
#   CalibrationNet & ForecastingNet → EXIT FROM BOTTOM FACE
#   --------------------------------------------------------
#   Each arrow drops from the bottom-centre of its NN box, travels down to
#   a dedicated sub-lane below the NN row, then turns right and enters the
#   corresponding output box on its left face.
#
#   Sub-lane assignments (horizontal corridors below the NN row):
#     SLA = R2Y + 0.55   sub-lane A: CalibrationNet → Calibration output
#     SLB = R2Y + 1.10   sub-lane B: ForecastingNet → Forecast output
#   Using two distinct y-depths keeps the two bottom-exit paths vertically
#   separated so they never overlap.
#
#   Bottom-exit 4-point elbow pattern (for each single-task NN):
#     pt1  bottom-centre of NN box      (cx_nn, R2Y)
#     pt2  sub-lane y directly below    (cx_nn, SL_)
#     pt3  sub-lane at output box edge  (OX,    SL_)
#     pt4  left-face entry of output    (OX,    OY + OH*frac)
#
#   MultiTaskNet → EXIT FROM RIGHT FACE (unchanged)
#   ------------------------------------------------
#   MultiTaskNet feeds BOTH output boxes. It exits the right face via lane C
#   and splits into two branches (upper +0.28 y → Calibration,
#   lower −0.28 y → Forecast).
# ══════════════════════════════════════════════════════════════════════════════

# Sub-lane y-depths (below the NN row baseline R2Y)
SLA = R2Y - 0.35   # CalibrationNet sub-lane
SLB = R2Y - 0.25   # ForecastingNet sub-lane

# Lane C for MultiTaskNet right-exit (unchanged)
RAIL = OX - 0.55
LC   = RAIL - 0.55

# ── CalibrationNet → Calibration output  (bottom exit) ──────────────────────
elbow([(cx_cn,              R2Y),           # pt1: exit bottom face of CalibrationNet
    (cx_cn,              SLA),              # pt2: drop to sub-lane A
       (OX - 0.75,          SLA),           # pt3: travel right along sub-lane A
       (OX - 0.75,         COY+OH*0.35),    # pt4: travel up along sub-lane A
       (OX,                COY+OH*0.35)])   # pt5: enter Calibration output left face

# ── ForecastingNet → Forecast output  (bottom exit) ─────────────────────────
elbow([(cx_fn,          R2Y),               # pt1: exit bottom face of ForecastingNet
       (cx_fn,          SLB),               # pt2: drop to sub-lane B
       (OX - 0.5,         SLB),             # pt3: travel right along sub-lane B
       (OX - 0.5,         FOY+OH*0.65),     # pt4: travel up along sub-lane B
       (OX,             FOY+OH*0.65)])      # pt5: enter Forecast output left face

# ── MultiTaskNet → Calibration output  (right exit, upper branch) ───────────
elbow([(MNX+BW2+S_EX+0.06, cy_mn+0.28),
       (LC,                 cy_mn+0.28),
       (LC,                 COY+OH*0.7),
       (OX,                 COY+OH*0.7)])

# ── MultiTaskNet → Forecast output  (right exit, lower branch) ──────────────
elbow([(MNX+BW2+S_EX+0.06, cy_mn-0.28),
       (LC,                 cy_mn-0.28),
       (LC,                 FOY+OH*0.35),
       (OX,                 FOY+OH*0.35)])



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — ARROW ROUTING: OUTPUT BOXES → DECISION ENGINE
# ------------------------------------------------------------------------------
# Both output boxes share a single detour lane (DETOUR = DEX − 0.55).
# Calibration enters the Decision Engine upper half (DEH*0.75);
# Forecast enters the lower half (DEH*0.25) — reflecting their roles.
# ══════════════════════════════════════════════════════════════════════════════
DETOUR=DEX-0.55
elbow([(OX+OW+S_EX+0.06,COY+OH/2),(DETOUR,COY+OH/2),
       (DETOUR,DEY+DEH*0.75),(DEX,DEY+DEH*0.75)])
elbow([(OX+OW+S_EX+0.06,FOY+OH/2),(DETOUR,FOY+OH/2),
       (DETOUR,DEY+DEH*0.25),(DEX,DEY+DEH*0.25)])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — ARROW: DECISION ENGINE → SMART IRRIGATION
# ══════════════════════════════════════════════════════════════════════════════
straight(DEX+DEW+0.30+0.06, cy_de, SIX-0.55, cy_de)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — STAT BADGES (BOTTOM ROW)
# ------------------------------------------------------------------------------
# Five badges span the full canvas width with equal spacing:
#   GAP_B = (W_TOT − 2×MARGIN − 5×BDW) / 4
# Colour mirrors corresponding architecture component for cross-referencing.
# ══════════════════════════════════════════════════════════════════════════════
MARGIN=0.50; GAP_B=(W_TOT-MARGIN*2-BDW*5)/4
COLS_B=[C['calib_nn'],C['fore_nn'],C['multi_nn'],C['decision'],C['calib_o']]
TOPS=['21,312','110 Days','9 ML','35 Ablation','Water Savings']
BOTS=['Readings','2 Farms','Baselines','Runs','> 30%']
for i,(t,b,c) in enumerate(zip(TOPS,BOTS,COLS_B)):
    badge(MARGIN+i*(BDW+GAP_B), BDY, BDW, BDH, t, b, c)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — EXPORT (300 DPI)
# ══════════════════════════════════════════════════════════════════════════════

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(fig_dir, exist_ok=True)

plt.savefig(os.path.join(fig_dir, 'fig_system_overview_3d.png'), dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("    Done.")