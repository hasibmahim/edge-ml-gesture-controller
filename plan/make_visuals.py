"""Generates visual diagrams and figures for the Group 13 repository and docs.

Outputs in ../docs/figures/:
  - hardware_overview.png: Front & back photos of the Group 13 controller + GPIO/I2C pinout diagram
  - gesture_waveforms.png: 5-class 6-axis IMU signatures + 0.2s pre-trigger & 60 deg/s onset gate
  - edge_vs_cloud_comparison.png: MQTT payload breakdown (Edge vs Hybrid vs Cloud) & LOPO baselines
"""

import math
import os
import random
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "docs" / "figures"
sys.path.insert(0, str(ROOT / "pico"))

import features  # noqa: E402


def _gauss(t, c, w):
    return math.exp(-((t - c) / w) ** 2)


def synth_window(label, seed=4):
    rng = random.Random(seed)
    n, fs = features.WINDOW, features.SAMPLE_RATE_HZ
    w = [[0.0] * n for _ in range(6)]
    c = 0.65
    phase = 0.3
    for i in range(n):
        t = i / fs
        ax, ay, az, gx, gy, gz = 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        if i >= features.PRE_TRIGGER:
            if label == "idle":
                ax = 0.2 * math.sin(2 * math.pi * 0.5 * t + phase)
                gy = 85 * math.sin(2 * math.pi * 0.6 * t + phase)
            elif label == "shake":
                env = _gauss(t, 0.78, 0.42)
                ax = 1.6 * env * math.sin(2 * math.pi * 5 * t)
                gz = 320 * env * math.cos(2 * math.pi * 5 * t)
            elif label == "circle":
                ax = 0.85 * math.sin(2 * math.pi * 1.1 * t)
                ay = 0.85 * math.cos(2 * math.pi * 1.1 * t)
                gx = 135 * math.cos(2 * math.pi * 1.1 * t)
                gy = -135 * math.sin(2 * math.pi * 1.1 * t)
            else:
                s = 1 if label == "flick_up" else -1
                gx = s * (410 * _gauss(t, c, 0.08) - 160 * _gauss(t, c + 0.22, 0.10))
                az = 1 + s * (1.25 * _gauss(t, c, 0.08) - 0.65 * _gauss(t, c + 0.16, 0.08))
        vals = (ax, ay, az, gx, gy, gz)
        for k in range(6):
            w[k][i] = vals[k] + rng.gauss(0, 0.02 if k < 3 else 2.2)
    return w


def make_gesture_waveforms():
    classes = [
        ("shake", "Shake\n(5 Hz oscillation)"),
        ("circle", "Circle\n(90° phase ax/ay)"),
        ("flick_up", "Flick Up\n(Peak before dip)"),
        ("flick_down", "Flick Down\n(Dip before peak)"),
        ("idle", "Idle Movement\n(Slow drift, ignored)"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(13.5, 5.2), sharex=True)
    t = [i / features.SAMPLE_RATE_HZ for i in range(features.WINDOW)]
    t_onset = features.PRE_TRIGGER / features.SAMPLE_RATE_HZ

    for col, (lbl, title) in enumerate(classes):
        w = synth_window(lbl, seed=10 + col)
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Pre-trigger shaded region
        for a in (ax_top, ax_bot):
            a.axvspan(0, t_onset, color="#f59e0b", alpha=0.12)
            a.axvline(t_onset, color="#d97706", linestyle="--", lw=1.0)
            a.grid(True, linestyle=":", alpha=0.45)

        ax_top.plot(t, w[0], color="#dc2626", lw=1.3, label="ax")
        ax_top.plot(t, w[1], color="#16a34a", lw=1.3, label="ay")
        ax_top.plot(t, w[2], color="#2563eb", lw=1.3, label="az")
        ax_top.set_ylim(-2.4, 2.6)
        ax_top.set_title(title, fontsize=9.5, weight="bold")

        ax_bot.plot(t, w[3], color="#ea580c", lw=1.3, label="gx")
        ax_bot.plot(t, w[4], color="#9333ea", lw=1.3, label="gy")
        ax_bot.plot(t, w[5], color="#ca8a04", lw=1.3, label="gz")
        ax_bot.axhline(60, color="#d97706", linestyle=":", lw=0.9)
        ax_bot.axhline(-60, color="#d97706", linestyle=":", lw=0.9)
        ax_bot.set_ylim(-460, 460)
        ax_bot.set_xlabel("Time (s)", fontsize=8.5)

        if col == 0:
            ax_top.set_ylabel("Acceleration (g)", fontsize=9, weight="bold")
            ax_bot.set_ylabel("Angular Velocity (°/s)", fontsize=9, weight="bold")
            ax_top.legend(loc="upper right", fontsize=7.5, frameon=True)
            ax_bot.legend(loc="upper right", fontsize=7.5, frameon=True)

    fig.suptitle(
        "Motion-Triggered 1.50 s Windows (75 Samples @ 50 Hz, 0.20 s Pre-Trigger Buffer Before |ω| > 60 °/s Onset)",
        fontsize=11,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = OUT / "gesture_waveforms.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote", out_path)


def make_edge_vs_cloud_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Left: MQTT Traffic per Window (Recognised Gesture vs Idle Movement)
    categories = ["Recognised Gesture\n(shake / circle / flick)", "Ordinary Idle Movement\n(classified as 'idle')"]
    edge_bytes = [70, 0]
    hybrid_bytes = [70, 0]  # high-confidence on-device path
    cloud_bytes = [1011, 941]

    x = [0, 1]
    width = 0.25
    b1 = ax1.bar([i - width for i in x], edge_bytes, width, label="Edge Mode (On-Pico)", color="#1f5fa8")
    b2 = ax1.bar(x, hybrid_bytes, width, label="Hybrid Mode (Conf ≥ 0.85)", color="#0ea5e9")
    b3 = ax1.bar([i + width for i in x], cloud_bytes, width, label="Cloud Mode (Raw Stream)", color="#d9731a")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 22,
                f"{int(h)} B",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylabel("MQTT Traffic per Window (Bytes, Topic + Payload)", fontsize=9, weight="bold")
    ax1.set_ylim(0, 1320)
    ax1.set_title("Scenario 1: Network Traffic per Window (93%–100% Reduction)", fontsize=10, weight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper center", fontsize=8, ncol=1)

    # Right: Model Complexity & Baselines (Synthetic Pipeline Verification)
    models = ["Majority\nBaseline", "Gyro Energy\nThreshold", "Tree\nDepth 3", "Tree\nDepth 5", "Tree\nDepth 8"]
    f1_scores = [0.067, 0.188, 1.000, 1.000, 1.000]
    colors = ["#94a3b8", "#64748b", "#10b981", "#1f5fa8", "#6366f1"]
    bars = ax2.bar(models, [v * 100 for v in f1_scores], color=colors, width=0.52)
    ax2.axhline(80, color="#dc2626", linestyle="--", lw=1.1, label="Rule 5 Usability Threshold (80% Macro-F1)")

    for bar, val in zip(bars, f1_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
        )

    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Leave-One-Person-Out Macro-F1 (%)", fontsize=9, weight="bold")
    ax2.set_title("Scenario 2: Pipeline Verification vs. Baselines (Synthetic Set)", fontsize=10, weight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out_path = OUT / "edge_vs_cloud_comparison.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote", out_path)


def make_hardware_overview():
    front_jpg = OUT / "hardware_front.jpg"
    back_jpg = OUT / "hardware_back.jpg"

    fig = plt.figure(figsize=(13.6, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.95, 1.95], wspace=0.12)

    ax_f = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[0, 2])

    if front_jpg.exists():
        ax_f.imshow(mpimg.imread(str(front_jpg)))
    ax_f.set_title("Front: TKJ HAT in Handheld Case\n(OLED, SW1/SW2, IMU, RGB LED)", fontsize=9, weight="bold")
    ax_f.axis("off")

    if back_jpg.exists():
        ax_b.imshow(mpimg.imread(str(back_jpg)))
    ax_b.set_title("Back: Raspberry Pi Pico WH\n(Wi-Fi, USB, Piezo Buzzer)", fontsize=9, weight="bold")
    ax_b.axis("off")

    # Right panel: Pinout & Multi-Node Schematic
    ax_p.set_xlim(0, 100)
    ax_p.set_ylim(0, 100)
    ax_p.axis("off")
    ax_p.set_title("Hardware Pin Mapping & Multi-Node Link", fontsize=9.5, weight="bold")

    ax_p.add_patch(
        FancyBboxPatch((2, 28), 96, 66, boxstyle="round,pad=1.2,rounding_size=2", facecolor="#eef4fb", edgecolor="#1f5fa8", lw=1.3)
    )
    ax_p.text(50, 89, "Node 1: Raspberry Pi Pico WH + TKJ HAT", ha="center", va="center", fontsize=9.5, weight="bold", color="#0f172a")

    rows = [
        ("I2C0 Bus (400 kHz)", "SDA = GP12, SCL = GP13"),
        ("ICM-42670 IMU", "I2C 0x69 (±8 g, ±1000 °/s @ 50 Hz)"),
        ("SSD1306 OLED (128×64)", "I2C 0x3C (gesture, conf, mode, lat)"),
        ("RGB LED (PWM)", "GP18 (R), GP19 (G), GP20 (B)"),
        ("Piezo Buzzer (PWM)", "GP17 (short beep on gesture)"),
        ("Buttons SW1 / SW2", "GP2 (record) / GP22 (mode switch)"),
        ("CYW43439 Wi-Fi", "GP23, GP24, GP25, GP29 (no clash)"),
    ]
    y = 79
    for left, right in rows:
        ax_p.add_patch(Rectangle((4.5, y - 3.5), 91, 6.5, facecolor="white", edgecolor="#cbd5e1", lw=0.7))
        ax_p.text(6.5, y, left, ha="left", va="center", fontsize=7.8, weight="bold", color="#1e293b")
        ax_p.text(93.5, y, right, ha="right", va="center", fontsize=7.6, color="#334155")
        y -= 7.8

    ax_p.add_patch(
        FancyBboxPatch((2, 4), 96, 19, boxstyle="round,pad=1.2,rounding_size=2", facecolor="#f0fdf4", edgecolor="#16a34a", lw=1.3)
    )
    ax_p.text(50, 17.5, "Node 2: Second Pico W Actuator (Course Kit)", ha="center", va="center", fontsize=9, weight="bold", color="#14532d")
    ax_p.text(
        50,
        9.5,
        "Subscribes to MQTT topic iot/pico2/command\nflick_up → 'on' (LED ON)   |   flick_down → 'off' (LED OFF)",
        ha="center",
        va="center",
        fontsize=8,
        color="#166534",
    )

    out_path = OUT / "hardware_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_gesture_waveforms()
    make_edge_vs_cloud_comparison()
    make_hardware_overview()
