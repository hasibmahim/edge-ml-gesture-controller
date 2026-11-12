"""Draws the architecture diagram for the group 13 plan (architecture.png and architecture.pdf).

Run: python make_architecture.py   (needs matplotlib)
Solid boxes: code written and tested on the PC. Dashed boxes: still to build.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
EDGE, CLOUD, DATA, INK = "#1f5fa8", "#d9731a", "#6b6b6b", "#222222"
BAND = ("#eef3fa", "#f4f4f4", "#f6f1ea")

fig, ax = plt.subplots(figsize=(11, 6.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")


def band(x, w, title, colour):
    ax.add_patch(Rectangle((x, 1), w, 56, facecolor=colour, edgecolor="#c8c8c8", lw=0.8))
    ax.text(x + w / 2, 58.6, title, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)


def box(x, y, w, h, title, body="", planned=False, edge=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                facecolor="white", edgecolor=edge, lw=1.1,
                                linestyle=(0, (4, 2)) if planned else "-"))
    if body:
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=8.2, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 4.1, body, ha="center", va="top", fontsize=6.9, color=INK, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.2, weight="bold", color=INK)


def arrow(p, q, colour=INK, label="", dashed=False, lx=0.0, ly=0.0, rad=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=colour, lw=1.3,
                                 linestyle=(0, (4, 2)) if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha=ha, va="center",
                fontsize=6.8, color=colour, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


# Layers
band(1, 36, "Sensor and device layer: Pico W 1 + TKJ HAT", BAND[0])
band(39, 21, "Network layer", BAND[1])
band(62, 37, "Data management and application layer", BAND[2])

# Device layer (Pico W 1)
box(3.5, 46, 31, 9, "ICM-42670 IMU",
    "I2C 0x69, 400 kHz, 50 Hz\n3-axis accel (±8 g) + 3-axis gyro (±1000 °/s)")
box(3.5, 34, 31, 9, "Segmenter (motion trigger)",
    "|gyro| > 60 °/s starts a 1.5 s window\n(75 samples × 6 axes, 0.2 s before onset)")
box(3.5, 22, 15, 9, "Features + tree", "40 features\ndecision tree\n(edge mode)", edge=EDGE)
box(20.5, 22, 14, 9, "Buttons", "SW1: record sample\nSW2: edge ↔ cloud")
box(3.5, 12, 31, 7, "Application logic",
    "confidence ≥ 0.6, cooldown, mode, latency metrics")
box(3.5, 2.5, 31, 7, "Local outputs", "OLED text · RGB LED colour · buzzer beep")

arrow((19, 46), (19, 43))
arrow((11, 34), (11, 31))
arrow((11, 22), (11, 19), EDGE)
arrow((27.5, 22), (27.5, 19))
arrow((19, 12), (19, 9.5))

# Network layer
box(40.5, 27, 18, 22, "MQTT broker",
    "Mosquitto, Wi-Fi 2.4 GHz\n\ngesture/<dev>/event\ngesture/<dev>/raw\ngesture/<dev>/label\n"
    "gesture/<dev>/metrics\ngesture/<dev>/mode\ngesture/<dev>/status\n(retained + last will)\n\n"
    "stretch: TLS (8883)\n+ username / password")
box(40.5, 3, 18, 11, "Pico W 2 (actuator)", "exercise 3 device\nLED on / off\niot/pico2/command")

# Pico 1 <-> broker
arrow((34.5, 38.5), (40.5, 44), CLOUD, "raw window\n903 B (cloud)", dashed=True, ly=1.8)
arrow((34.5, 17.5), (40.5, 33.5), EDGE, "event ≈ 70 B\n(edge)", lx=-0.6, ly=1.6)
arrow((40.5, 27.5), (34.5, 13), CLOUD, "label\n(cloud)", dashed=True, lx=1.6, ly=-2.6)
arrow((49.5, 27), (49.5, 14), EDGE, "flick up / down\n→ command", lx=0.2)

# Data management / application layer
box(64, 45, 33, 10, "Cloud classifier (Python, PC)",
    "same features.py + model.py as the Pico\nreplies \"seq,label,conf\"", edge=CLOUD)
box(64, 30, 10, 9, "Node-RED", "flow:\nparse, tag")
box(75.5, 30, 10, 9, "InfluxDB", "events,\nmetrics")
box(87, 30, 10, 9, "Grafana", "gestures/min,\nlatency, bytes")
box(64, 16, 33, 9, "Evaluation logger, plots, web console",
    "eval_logger.py, plots.py: bytes, latency, timeouts\nweb_ui.py: live view and mode switch")
box(64, 4, 33, 8.5, "Training (PC, offline)",
    "capture.py → train.py (depth 3/5/8, leave-one-person-out)\n→ export_tree.py → model.py copied to Pico")

arrow((58.5, 46), (64, 50), CLOUD, "raw", dashed=True, ly=1.6, rad=-0.1)
arrow((64, 47.5), (58.5, 43), CLOUD, "label", dashed=True, ly=-1.6, rad=-0.1)
arrow((58.5, 35), (64, 34.5), DATA, "events,\nmetrics", ly=2.6)
arrow((74, 34.5), (75.5, 34.5), DATA)
arrow((85.5, 34.5), (87, 34.5), DATA)
arrow((58.5, 29), (64, 21), DATA, "metrics", lx=1.5, ly=1.0)

# Legend
lx, ly = 64.5, -2.2
ax.plot([lx, lx + 4], [ly, ly], color=EDGE, lw=1.4)
ax.text(lx + 5, ly, "edge path", fontsize=7, va="center")
ax.plot([lx + 14, lx + 18], [ly, ly], color=CLOUD, lw=1.4, linestyle=(0, (4, 2)))
ax.text(lx + 19, ly, "cloud path", fontsize=7, va="center")
ax.add_patch(Rectangle((1, ly - 1.1), 3.5, 2.2, facecolor="white", edgecolor=INK, lw=1.0))
ax.text(5.3, ly, "written and PC-tested", fontsize=7, va="center")
ax.add_patch(Rectangle((22, ly - 1.1), 3.5, 2.2, facecolor="white", edgecolor=INK, lw=1.0,
                       linestyle=(0, (4, 2))))
ax.text(26.3, ly, "to build (weeks 4-6)", fontsize=7, va="center")
ax.set_ylim(-4, 62)

fig.tight_layout(pad=0.3)
fig.savefig(HERE / "architecture.png", dpi=200)
fig.savefig(HERE / "architecture.pdf")
print("wrote", HERE / "architecture.png")
