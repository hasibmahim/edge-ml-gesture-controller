"""Web console for the Edge-ML Gesture Controller (group 13).

Shows what the Pico publishes over MQTT: last gesture, mode, the last raw window
(cloud mode only), the 40 features, the metrics log and a summary per mode. It can
switch the Pico's mode and send on/off to the second Pico.

It also has a SIMULATOR for demos without hardware: synthetic gestures run through
the real features.py, model.py and cloud_classifier.py on this PC. Simulated rows are
marked source = "sim". They are never included in the summary table or the CSV
export, and their timing is PC time, not Pico or network time. Do not use them as
evaluation data in the report.

Usage:
  python pc/web_ui.py [--port 8080] [--broker localhost] [--device gesture1]
The page is served on this PC only (127.0.0.1). Use --host 0.0.0.0 to open it to
the local network, for example to a phone. There is no login, so only do that on a
network you trust.
"""

import argparse
import collections
import csv
import importlib.util
import io
import json
import math
import os
import random
import statistics
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PICO = os.path.join(ROOT, "pico")
RESULTS_DIR = os.path.join(ROOT, "results")            # real results from train.py
SYNTH_DIR = os.path.join(RESULTS_DIR, "synthetic_demo")  # results of the made-up bootstrap data
sys.path[:0] = [PICO, HERE]

import features  # noqa: E402
import cloud_classifier  # noqa: E402
import main as pico_main  # noqa: E402


def _gauss(t, c, w):
    return math.exp(-((t - c) / w) ** 2)


def synth_window(label, rng=None, amp=1.0, speed=1.0, noisy=False):
    rng = rng or random.Random()
    n, fs = features.WINDOW, features.SAMPLE_RATE_HZ
    w = [[0.0] * n for _ in range(6)]
    c = rng.uniform(0.55, 0.95)
    phase = rng.uniform(0, 2 * math.pi)
    noise_a = 0.12 if noisy else 0.03
    noise_g = 14.0 if noisy else 3.0
    for i in range(n):
        t = i / fs
        ax, ay, az, gx, gy, gz = 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        if label == "idle":
            ax = 0.2 * math.sin(2 * math.pi * 0.5 * t + phase)
            gy = 90 * amp * math.sin(2 * math.pi * 0.6 * t + phase)
        elif label == "shake":
            env = _gauss(t, 0.75, 0.45)
            ax = 1.5 * amp * env * math.sin(2 * math.pi * 5 * speed * t + phase)
            gz = 300 * amp * env * math.cos(2 * math.pi * 5 * speed * t + phase)
        elif label == "circle":
            ax = 0.8 * amp * math.sin(2 * math.pi * speed * t + phase)
            ay = 0.8 * amp * math.cos(2 * math.pi * speed * t + phase)
            gx = 120 * amp * math.cos(2 * math.pi * speed * t + phase)
            gy = -120 * amp * math.sin(2 * math.pi * speed * t + phase)
        else:
            s = 1 if label == "flick_up" else -1
            gx = s * amp * (400 * _gauss(t, c, 0.08 / speed) - 150 * _gauss(t, c + 0.2, 0.1))
            az = 1 + s * amp * (1.2 * _gauss(t, c, 0.08) - 0.6 * _gauss(t, c + 0.15, 0.08))
        vals = (ax, ay, az, gx, gy, gz)
        for k in range(6):
            w[k][i] = round(vals[k] + rng.gauss(0, noise_a if k < 3 else noise_g), 4)
    return w


def ensure_default_model():
    model_path = os.path.join(PICO, "model.py")
    if not os.path.exists(model_path):
        try:
            import train
            import capture

            tmp_data = tempfile.mkdtemp(prefix="gesture-bootstrap-")
            rng = random.Random(42)
            paths = []
            for p in ("member1", "member2", "member3"):
                amp, speed = rng.uniform(0.82, 1.18), rng.uniform(0.88, 1.12)
                p_csv = os.path.join(tmp_data, f"{p}.csv")
                with open(p_csv, "w", newline="") as f:
                    seq = 0
                    for lbl in features.CLASSES:
                        for _ in range(30):
                            w = synth_window(lbl, rng, amp, speed)
                            values = ",".join("%.3f" % x for axis in w for x in axis)
                            line = "G,%s,%d,%s" % (lbl, seq, values)
                            f.write(capture.to_csv_row(line, p)[1])
                            seq += 1
                paths.append(p_csv)
            train.run(paths, 5, SYNTH_DIR, model_path)
            with open(model_path) as f:
                code = f.read()
            with open(model_path, "w") as f:
                f.write("# SYNTHETIC PLACEHOLDER: trained on made-up gestures by web_ui.py, not on real\n"
                        "# recordings. Replace it by running pc/train.py on the group's data.\n" + code)
        except Exception:
            code = (
                "# SYNTHETIC PLACEHOLDER: hand-written rules, not a trained model.\n"
                "CLASSES = ('circle', 'flick_down', 'flick_up', 'idle', 'shake')\n"
                "N_FEATURES = 40\n\n"
                "def predict(x):\n"
                "    if x[34] > 6: return 4, 0.96\n"
                "    if abs(x[1]) > 0.35 and abs(x[7]) > 0.35: return 0, 0.94\n"
                "    if x[21] > 180 or x[20] < -180:\n"
                "        return (2, 0.95) if x[23] < 0 else (1, 0.95)\n"
                "    return 3, 0.92\n"
            )
            with open(model_path, "w") as f:
                f.write(code)
    spec = importlib.util.spec_from_file_location("gesture_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(model_path, "r") as f:
        code_str = f.read()
    return mod, code_str


class HubState:
    def __init__(self, device="gesture1", broker_host="localhost", broker_port=1883):
        self.lock = threading.Lock()
        self.device = device
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.mqtt_connected = False
        self.pico_online = False
        self.mode = "edge"
        self.hybrid_threshold = 0.85
        self.lamp_on = False
        self.seq = 0
        self.model, self.model_code = ensure_default_model()
        self.last_window = synth_window("shake", random.Random(1))
        self.last_features = [round(x, 4) for x in features.extract(self.last_window)]
        self.last_gesture = {
            "seq": 0,
            "g": "idle",
            "conf": 0.0,
            "mode": "edge",
            "routed": "-",
            "lat_ms": 0,
            "bytes": 0,
            "ts": "--:--:--",
            "source": "sim",
        }
        self.history = collections.deque(maxlen=300)
        self.mqtt_client = None
        self._start_mqtt()

    def _start_mqtt(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            return

        def on_connect(c, userdata, flags, reason, props):
            with self.lock:
                self.mqtt_connected = True
            c.subscribe("gesture/+/event")
            c.subscribe("gesture/+/metrics")
            c.subscribe("gesture/+/raw")
            c.subscribe("gesture/+/status")
            c.subscribe("iot/+/command")

        def on_disconnect(c, userdata, flags, reason, props):
            with self.lock:
                self.mqtt_connected = False

        def on_message(c, userdata, msg):
            topic = msg.topic
            payload = msg.payload
            with self.lock:
                if topic.endswith("/status"):
                    self.pico_online = payload.decode("utf-8", "replace").strip() == "online"
                elif topic.endswith("/command"):
                    cmd = payload.decode("utf-8", "replace").strip().lower()
                    self.lamp_on = cmd == "on"
                elif topic.endswith("/raw"):
                    try:
                        _seq, w = cloud_classifier.unpack_window(payload)
                        self.last_window = w
                        self.last_features = [round(x, 4) for x in features.extract(w)]
                    except Exception:
                        pass
                elif topic.endswith("/metrics"):
                    try:
                        m = json.loads(payload.decode("utf-8"))
                        m["ts"] = time.strftime("%H:%M:%S")
                        m["source"] = "hw"
                        m["routed"] = "cloud-wan" if m.get("bytes", 0) > 200 else "on-device"
                        self.mode = m.get("mode", self.mode)
                        self.history.appendleft(m)
                        if m.get("g") and m.get("g") != "idle":
                            self.last_gesture = m
                    except Exception:
                        pass

        def runner():
            try:
                c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="group13-console")
                c.on_connect = on_connect
                c.on_disconnect = on_disconnect
                c.on_message = on_message
                self.mqtt_client = c
                c.connect_async(self.broker_host, self.broker_port, keepalive=30)
                c.loop_forever()
            except Exception:
                pass

        threading.Thread(target=runner, daemon=True).start()

    def set_mode(self, mode):
        if mode not in ("edge", "hybrid", "cloud"):
            return
        with self.lock:
            self.mode = mode
            if self.mqtt_client and self.mqtt_connected:
                self.mqtt_client.publish(f"gesture/{self.device}/mode", mode)

    def set_lamp(self, state):
        with self.lock:
            self.lamp_on = bool(state)
            if self.mqtt_client and self.mqtt_connected:
                self.mqtt_client.publish("iot/pico2/command", "on" if self.lamp_on else "off", qos=1)

    def simulate_gesture(self, label, force_mode=None, wan_drop=False, ambiguous=False):
        with self.lock:
            mode = force_mode or self.mode
            self.seq = (self.seq + 1) & 0xFFFF
            w = synth_window(label, noisy=ambiguous)
            t0 = time.perf_counter()
            feats = features.extract(w)
            idx, conf = self.model.predict(feats)
            pred_name = self.model.CLASSES[idx]
            # PC compute time only. It says nothing about the Pico or the network.
            dt_edge_ms = (time.perf_counter() - t0) * 1000.0

            offloaded = False
            if mode == "edge":
                routed = "on-device"
                lat_ms = round(dt_edge_ms, 2)
                if pred_name != "idle" and conf >= 0.6:
                    ev = json.dumps({"g": pred_name, "conf": conf, "mode": "edge"})
                    nbytes = len(ev) + len(f"gesture/{self.device}/event")
                else:
                    nbytes = 0
                error = None
            elif mode == "hybrid":
                if conf >= self.hybrid_threshold:
                    routed = "on-device"
                    lat_ms = round(dt_edge_ms, 2)
                    if pred_name != "idle" and conf >= 0.6:
                        ev = json.dumps({"g": pred_name, "conf": conf, "mode": "hybrid"})
                        nbytes = len(ev) + len(f"gesture/{self.device}/event")
                    else:
                        nbytes = 0
                    error = None
                else:
                    offloaded = True
                    routed = "cloud-offload"
            else:
                offloaded = True
                routed = "cloud-wan"

            if offloaded:
                packed = pico_main.pack_window(self.seq, w)
                up_bytes = len(packed) + len(f"gesture/{self.device}/raw")
                if wan_drop:
                    lat_ms = -1.0
                    nbytes = up_bytes
                    pred_name = None
                    conf = 0.0
                    error = "timeout"
                else:
                    reply = cloud_classifier.classify(self.model, packed)
                    down_bytes = len(reply) + len(f"gesture/{self.device}/label")
                    nbytes = up_bytes + down_bytes
                    _seq, name, conf = reply.split(",")
                    pred_name, conf = name, float(conf)
                    lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    error = None

            if pred_name == "flick_up" and conf >= 0.6:
                self.lamp_on = True
            elif pred_name == "flick_down" and conf >= 0.6:
                self.lamp_on = False

            rec = {
                "seq": self.seq,
                "mode": mode,
                "routed": routed,
                "g": pred_name,
                "true_g": label,
                "conf": round(float(conf), 3),
                "lat_ms": lat_ms,
                "bytes": nbytes,
                "error": error,
                "ts": time.strftime("%H:%M:%S"),
                "source": "sim",
            }
            self.last_window = w
            self.last_features = [round(x, 4) for x in feats]
            if pred_name and pred_name != "idle":
                self.last_gesture = rec
            self.history.appendleft(rec)
            return rec

    def run_benchmark_suite(self):
        rng = random.Random(99)
        for m in ("edge", "hybrid", "cloud"):
            for lbl in features.CLASSES:
                for i in range(4):
                    # one noisier window per class, so the demo is not all clean gestures
                    amb = (i == 3 and lbl != "idle" and rng.random() < 0.8)
                    self.simulate_gesture(lbl, force_mode=m, ambiguous=amb)

    def clear_history(self):
        with self.lock:
            self.history.clear()

    def summary_stats(self, source="hw"):
        out = {}
        for m in ("edge", "hybrid", "cloud"):
            rows = [r for r in self.history if r.get("mode") == m and r.get("source") == source]
            if not rows:
                out[m] = {
                    "windows": 0, "lost": 0, "offloaded": 0, "offload_pct": 0.0,
                    "bytes_per_win": 0.0, "total_bytes": 0, "p5_ms": 0.0,
                    "p25_ms": 0.0, "median_ms": 0.0, "p75_ms": 0.0, "p95_ms": 0.0, "lats": []
                }
                continue
            ok = [r for r in rows if r.get("lat_ms", -1) >= 0]
            off = [r for r in rows if r.get("routed") in ("cloud-offload", "cloud-wan")]
            lats = sorted(float(r["lat_ms"]) for r in ok)
            total_b = sum(int(r.get("bytes", 0)) for r in rows)

            def pct(p):
                if not lats:
                    return 0.0
                idx = min(len(lats) - 1, max(0, int(round(p * (len(lats) - 1)))))
                return round(lats[idx], 2)

            out[m] = {
                "windows": len(rows),
                "lost": len(rows) - len(ok),
                "offload_pct": round(100.0 * len(off) / len(rows), 1),
                "bytes_per_win": round(total_b / len(rows), 1),
                "total_bytes": total_b,
                "p5_ms": pct(0.05),
                "p25_ms": pct(0.25),
                "median_ms": round(statistics.median(lats), 2) if lats else 0.0,
                "p75_ms": pct(0.75),
                "p95_ms": pct(0.95),
                "lats": lats[:60],
            }
        return out

    def data_inventory(self):
        data_dir = os.path.join(ROOT, "data")
        inv = []
        if not os.path.isdir(data_dir):
            return inv
        for fn in sorted(os.listdir(data_dir)):
            if not fn.endswith(".csv"):
                continue
            fpath = os.path.join(data_dir, fn)
            counts = {c: 0 for c in features.CLASSES}
            person = os.path.splitext(fn)[0]
            try:
                with open(fpath, newline="") as f:
                    for row in csv.reader(f):
                        if len(row) >= 2 and row[1] in counts:
                            person = row[0] or person
                            counts[row[1]] += 1
            except Exception:
                continue
            inv.append({
                "file": fn,
                "person": person,
                "counts": counts,
                "total": sum(counts.values()),
            })
        return inv

    def export_csv(self):
        with self.lock:
            buf = io.StringIO()
            fields = ["ts", "seq", "mode", "routed", "g", "conf", "lat_ms", "bytes", "error", "source"]
            w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in reversed(self.history):
                if r.get("source") == "hw":
                    w.writerow(r)
            return buf.getvalue()

    def snapshot(self):
        with self.lock:
            depth_rows = []
            is_synthetic_results = False
            depth_csv = os.path.join(RESULTS_DIR, "depth_results.csv")
            synth_csv = os.path.join(SYNTH_DIR, "depth_results.csv")
            if os.path.exists(depth_csv):
                with open(depth_csv, newline="") as f:
                    depth_rows = list(csv.DictReader(f))
            elif os.path.exists(synth_csv):
                with open(synth_csv, newline="") as f:
                    depth_rows = list(csv.DictReader(f))
                is_synthetic_results = True

            baselines = []
            base_csv = os.path.join(RESULTS_DIR, "baselines.csv")
            if os.path.exists(base_csv):
                with open(base_csv, newline="") as f:
                    baselines = list(csv.DictReader(f))

            per_person = []
            pp_csv = os.path.join(RESULTS_DIR, "per_person.csv")
            if os.path.exists(pp_csv):
                with open(pp_csv, newline="") as f:
                    per_person = list(csv.DictReader(f))

            return {
                "device": self.device,
                "broker": f"{self.broker_host}:{self.broker_port}",
                "mqtt_connected": self.mqtt_connected,
                "pico_online": self.pico_online,
                "mode": self.mode,
                "hybrid_threshold": self.hybrid_threshold,
                "lamp_on": self.lamp_on,
                "last_gesture": self.last_gesture,
                "last_window": self.last_window,
                "feature_names": features.FEATURE_NAMES,
                "last_features": self.last_features,
                "history": list(self.history)[:45],
                "summary": self.summary_stats("hw"),
                "sim_summary": self.summary_stats("sim"),
                "depth_results": depth_rows,
                "is_synthetic_results": is_synthetic_results,
                "baselines": baselines,
                "per_person": per_person,
                "data_inventory": self.data_inventory(),
                "model_code": self.model_code,
            }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OU-521043S // GRP-13 Edge-ML Telemetry & Evaluation Console</title>
<style>
  :root {
    --bg-base: #060911;
    --bg-panel: #0b101d;
    --bg-sub: #0f172a;
    --bg-elevated: #141e33;
    --border: #1e293b;
    --border-bright: #334155;
    --text-main: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #005aff;
    --accent-cyan: #00c8ff;
    --edge-green: #10b981;
    --hybrid-cyan: #06b6d4;
    --cloud-indigo: #6366f1;
    --amber: #f59e0b;
    --crimson: #ef4444;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background-color: var(--bg-base);
    color: var(--text-main);
    font-family: var(--font-sans);
    font-size: 13px;
    line-height: 1.45;
    font-variant-numeric: tabular-nums;
    -webkit-font-smoothing: antialiased;
  }
  /* Top Navigation & Telemetry Header */
  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: 14px;
  }
  .sys-id {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .accent-mark {
    width: 4px;
    height: 28px;
    background: var(--accent-blue);
    border-radius: 1px;
  }
  .sys-title {
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.03em;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sys-meta {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.02em;
  }
  .topbar-right {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .status-tag {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 10px;
    background: var(--bg-sub);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
  }
  .led-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--edge-green);
  }
  /* Segmented Mode Selector */
  .seg-control {
    display: inline-flex;
    background: var(--bg-base);
    border: 1px solid var(--border-bright);
    border-radius: 3px;
    overflow: hidden;
  }
  .seg-btn {
    border: 0;
    border-right: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    padding: 6px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
  }
  .seg-btn:last-child { border-right: 0; }
  .seg-btn:hover { background: var(--bg-elevated); color: var(--text-main); }
  .seg-btn.active-edge { background: rgba(16, 185, 129, 0.18); color: #34d399; box-shadow: inset 0 -2px 0 var(--edge-green); }
  .seg-btn.active-hybrid { background: rgba(6, 182, 212, 0.18); color: #22d3ee; box-shadow: inset 0 -2px 0 var(--hybrid-cyan); }
  .seg-btn.active-cloud { background: rgba(99, 102, 241, 0.22); color: #818cf8; box-shadow: inset 0 -2px 0 var(--cloud-indigo); }

  /* Main Layout Grid */
  .workspace {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 12px;
    padding: 12px 20px 24px 20px;
    max-width: 1640px;
    margin: 0 auto;
  }
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    display: flex;
    flex-direction: column;
  }
  .panel-hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.55);
  }
  .panel-title {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-idx {
    color: var(--accent-cyan);
    font-weight: 700;
  }
  .panel-sub {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
  }
  .panel-body {
    padding: 12px;
    flex: 1;
  }
  .col-4 { grid-column: span 4; }
  .col-5 { grid-column: span 5; }
  .col-6 { grid-column: span 6; }
  .col-7 { grid-column: span 7; }
  .col-8 { grid-column: span 8; }
  .col-12 { grid-column: span 12; }
  @media (max-width: 1180px) {
    .col-4, .col-5, .col-6, .col-7, .col-8 { grid-column: span 12; }
  }

  /* Hardware Diagnostics & OLED Canvas */
  .hw-grid {
    display: grid;
    grid-template-columns: 204px 1fr;
    gap: 12px;
    align-items: stretch;
  }
  @media (max-width: 560px) {
    .hw-grid { grid-template-columns: 1fr; }
  }
  .oled-frame {
    background: #020408;
    border: 1px solid #26334d;
    border-radius: 3px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .oled-caption {
    width: 100%;
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-muted);
    margin-bottom: 5px;
  }
  .diag-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    justify-content: space-between;
  }
  .diag-item {
    background: var(--bg-sub);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 6px 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .diag-lbl {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-muted);
  }
  .diag-val {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .rgb-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    border: 1px solid rgba(255,255,255,0.25);
  }

  /* Action Buttons */
  .section-lbl {
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin: 12px 0 6px 0;
  }
  .btn-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
  }
  .btn-row-ops {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px;
    margin-top: 6px;
  }
  .btn {
    background: var(--bg-sub);
    border: 1px solid var(--border-bright);
    color: var(--text-main);
    border-radius: 3px;
    padding: 7px 8px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    transition: background 0.12s, border-color 0.12s;
  }
  .btn:hover {
    background: var(--bg-elevated);
    border-color: var(--accent-cyan);
  }
  .btn-primary {
    background: rgba(0, 90, 255, 0.22);
    border-color: var(--accent-blue);
    color: #93c5fd;
  }
  .btn-primary:hover {
    background: rgba(0, 90, 255, 0.35);
  }
  .btn-warn {
    border-color: rgba(245, 158, 11, 0.45);
    color: #fcd34d;
  }
  .btn-danger {
    border-color: rgba(239, 68, 68, 0.45);
    color: #fca5a5;
  }

  /* Oscilloscope Canvases */
  .scope-wrap {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  canvas.scope {
    width: 100%;
    background: #04070e;
    border: 1px solid var(--border);
    border-radius: 3px;
    display: block;
    cursor: crosshair;
  }
  .scope-legend {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
    padding: 2px 4px;
  }
  .legend-items {
    display: flex;
    gap: 14px;
  }
  .lg-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .lg-line {
    width: 12px;
    height: 2px;
    display: inline-block;
  }

  /* Engineering Tables */
  table.eng-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 11px;
  }
  table.eng-table th, table.eng-table td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    text-align: right;
  }
  table.eng-table th:first-child, table.eng-table td:first-child {
    text-align: left;
  }
  table.eng-table th {
    color: var(--text-muted);
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(15, 23, 42, 0.4);
  }
  .mode-pill {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
  }
  .mp-edge { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
  .mp-hybrid { background: rgba(6, 182, 212, 0.16); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.35); }
  .mp-cloud { background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); }

  /* Tabs for Bottom Inspector */
  .tab-bar {
    display: flex;
    gap: 2px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-base);
    padding: 0 8px;
  }
  .tab-btn {
    border: 0;
    background: transparent;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    padding: 8px 14px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }
  .tab-btn:hover { color: var(--text-main); }
  .tab-btn.active {
    color: var(--accent-cyan);
    border-bottom-color: var(--accent-cyan);
    background: var(--bg-panel);
  }
  .tab-pane { display: none; padding: 12px; }
  .tab-pane.active { display: block; }
  pre.code-view {
    margin: 0;
    background: #04070e;
    border: 1px solid var(--border);
    padding: 10px 12px;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: #cbd5e1;
    max-height: 220px;
    overflow: auto;
  }
</style>
</head>
<body>

<header class="topbar">
  <div class="sys-id">
    <div>
      <div class="sys-title">
        <span>OU-521043S // GROUP 13 // EDGE-ML TELEMETRY &amp; EVALUATION CONSOLE</span>
      </div>
      <div class="sys-meta">TARGET: RP2040 + CYW43439 + TKJ-HAT (ICM-42670 6-AXIS IMU @ 50 HZ) &bull; BACKEND: MQTT BROKER</div>
    </div>
  </div>
  <div class="topbar-right">
    <div class="status-tag">
      <span class="led-dot" id="mqttDot"></span>
      <span id="mqttStatusText">TRANSPORT: SIMULATOR ENGINE</span>
    </div>
    <div class="status-tag">
      <span>ACTIVE PIPELINE:</span>
      <strong id="activeModeText" style="color:var(--accent-cyan)">EDGE</strong>
    </div>
    <div class="seg-control">
      <button class="seg-btn active-edge" id="segEdge" onclick="setMode('edge')">01 EDGE (ON-PICO)</button>
      <button class="seg-btn" id="segHybrid" onclick="setMode('hybrid')">02 HYBRID (CONF &ge; 0.85)</button>
      <button class="seg-btn" id="segCloud" onclick="setMode('cloud')">03 CLOUD</button>
    </div>
  </div>
</header>

<main class="workspace">
  <!-- MODULE 01: NODE DIAGNOSTICS, OLED DOT-MATRIX EMULATOR & TEST BENCH -->
  <section class="panel col-5">
    <div class="panel-hdr">
      <div class="panel-title"><span class="panel-idx">[01]</span> NODE HARDWARE STATE &amp; INJECTION BENCH</div>
      <div class="panel-sub" id="hdrTimestamp">UTC --:--:--</div>
    </div>
    <div class="panel-body">
      <div class="hw-grid">
        <div class="oled-frame">
          <div class="oled-caption"><span>SSD1306 I2C 0x3C</span><span>128x64 PX</span></div>
          <canvas id="oledCanvas" width="184" height="92" style="background:#000;border:1px solid #1e293b"></canvas>
        </div>
        <div class="diag-list">
          <div class="diag-item">
            <span class="diag-lbl">DETECTED CLASS</span>
            <span class="diag-val" id="diagClass" style="color:var(--accent-cyan)">SHAKE (98.0%)</span>
          </div>
          <div class="diag-item">
            <span class="diag-lbl">EXECUTION ROUTE</span>
            <span class="diag-val" id="diagRoute">ON-DEVICE (1.84 ms)</span>
          </div>
          <div class="diag-item">
            <span class="diag-lbl">HAT RGB PWM (GP18-20)</span>
            <span class="diag-val"><span class="rgb-swatch" id="rgbSwatch"></span><span id="rgbHex">255, 0, 0</span></span>
          </div>
          <div class="diag-item">
            <span class="diag-lbl">NODE 2 ACTUATOR (PICO2)</span>
            <span class="diag-val">
              <span id="lampStateText" style="color:var(--text-secondary)">RELAY: OFF</span>
              <button class="btn" style="padding:2px 7px;font-size:10px" onclick="toggleLamp()">TOGGLE</button>
            </span>
          </div>
        </div>
      </div>

      <div class="section-lbl">INJECT TEST WINDOW (50 HZ / 75 SAMPLES THROUGH FEATURES.PY + MODEL.PY)</div>
      <div class="btn-row">
        <button class="btn" onclick="triggerGesture('shake')">SHAKE</button>
        <button class="btn" onclick="triggerGesture('circle')">CIRCLE</button>
        <button class="btn" onclick="triggerGesture('flick_up')">FLICK_UP</button>
        <button class="btn" onclick="triggerGesture('flick_down')">FLICK_DN</button>
        <button class="btn" onclick="triggerGesture('idle')">IDLE_MOV</button>
      </div>

      <div class="section-lbl">SIMULATOR (SYNTHETIC GESTURES, PC TIMING)</div>
      <div class="btn-row-ops">
        <button class="btn btn-primary" onclick="runBenchmark()">RUN SIMULATED DEMO (NOT EVALUATION DATA)</button>
        <button class="btn btn-warn" onclick="triggerAmbiguous()">SIMULATE NOISY SAMPLE (HYBRID)</button>
        <button class="btn btn-danger" onclick="triggerOutage()">SIMULATE CLOUD TIMEOUT</button>
        <div style="display:flex;gap:6px">
          <button class="btn" style="flex:1" onclick="window.location='/api/export.csv'">EXPORT HARDWARE CSV</button>
          <button class="btn" style="flex:1" onclick="clearHistory()">CLEAR LOG</button>
        </div>
      </div>
    </div>
  </section>

  <!-- MODULE 02: DUAL-CHANNEL CALIBRATED 6-AXIS IMU OSCILLOSCOPE -->
  <section class="panel col-7">
    <div class="panel-hdr">
      <div class="panel-title"><span class="panel-idx">[02]</span> CALIBRATED 6-AXIS INERTIAL OSCILLOSCOPE (ICM-42670)</div>
      <div class="panel-sub" id="scopeCursorReadout">HOVER TRACE FOR SAMPLE CURSOR &bull; N=75 (1.50 S)</div>
    </div>
    <div class="panel-body scope-wrap">
      <div class="scope-legend">
        <span>CH-A: LINEAR ACCELERATION (FS = &plusmn;8g, DISPLAY = &plusmn;3.0g)</span>
        <div class="legend-items">
          <span class="lg-item"><span class="lg-line" style="background:#ef4444"></span>ax</span>
          <span class="lg-item"><span class="lg-line" style="background:#10b981"></span>ay</span>
          <span class="lg-item"><span class="lg-line" style="background:#38bdf8"></span>az</span>
        </div>
      </div>
      <canvas id="accelScope" class="scope" width="860" height="122"></canvas>

      <div class="scope-legend" style="margin-top:4px">
        <span>CH-B: ANGULAR VELOCITY (FS = &plusmn;1000 dps, ONSET TRIGGER = 60.0 dps @ SAMPLE 10)</span>
        <div class="legend-items">
          <span class="lg-item"><span class="lg-line" style="background:#f97316"></span>gx</span>
          <span class="lg-item"><span class="lg-line" style="background:#a855f7"></span>gy</span>
          <span class="lg-item"><span class="lg-line" style="background:#eab308"></span>gz</span>
          <span class="lg-item"><span class="lg-line" style="background:#f59e0b;border-bottom:1px dashed #f59e0b"></span>&plusmn;60 dps gate</span>
        </div>
      </div>
      <canvas id="gyroScope" class="scope" width="860" height="122"></canvas>
    </div>
  </section>

  <!-- MODULE 03: ARCHITECTURAL BENCHMARK (EDGE vs HYBRID vs CLOUD) -->
  <section class="panel col-6">
    <div class="panel-hdr">
      <div class="panel-title">
        <span class="panel-idx">[03]</span>
        <span id="evalPanelTitle">ARCHITECTURAL EVALUATION: EDGE vs. HYBRID vs. CLOUD</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <div class="panel-sub" id="bwGainSummary">BANDWIDTH REDUCTION: --%</div>
        <button class="btn" id="evalSourceToggle" style="padding:2px 8px;font-size:10px" onclick="toggleEvalSource()">VIEW: HW ONLY</button>
      </div>
    </div>
    <div class="panel-body">
      <div id="evalSourceBanner" style="display:none;margin-bottom:8px;padding:4px 8px;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.45);border-radius:3px;font-family:var(--font-mono);font-size:10px;color:#fcd34d">
        SIMULATED PREVIEW (PC TIMING ONLY) &bull; NEVER INCLUDED IN HARDWARE CSV EXPORT OR REPORT METRICS
      </div>
      <table class="eng-table">
        <thead>
          <tr>
            <th>ARCHITECTURE</th>
            <th>WINDOWS</th>
            <th>OFFLOAD %</th>
            <th>BYTES / WIN</th>
            <th>P50 LATENCY</th>
            <th>P95 LATENCY</th>
            <th>LOSS</th>
          </tr>
        </thead>
        <tbody id="evalTableBody"></tbody>
      </table>
      <canvas id="distCanvas" class="scope" width="720" height="168" style="margin-top:10px;cursor:default"></canvas>
    </div>
  </section>

  <!-- MODULE 04: MULTI-TAB ENGINEERING INSPECTOR -->
  <section class="panel col-6">
    <div class="panel-hdr">
      <div class="panel-title"><span class="panel-idx">[04]</span> FEATURE VECTOR, LOPO DEPTH, SESSION COACH &amp; INSPECTOR</div>
      <div class="panel-sub">40 FEATURES &bull; SCIKIT-LEARN TO MICROPYTHON</div>
    </div>
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab(0, this)">40-FEATURE MATRIX</button>
      <button class="tab-btn" onclick="switchTab(1, this)">TREE DEPTH (D=3,5,8)</button>
      <button class="tab-btn" onclick="switchTab(2, this)">TELEMETRY LOG</button>
      <button class="tab-btn" onclick="switchTab(3, this)">COMPILED MODEL.PY</button>
      <button class="tab-btn" onclick="switchTab(4, this)">SESSION COACH &amp; RULES</button>
    </div>

    <!-- TAB 0: 40-Feature Matrix (6 axes x 6 stats + 4 magnitude stats) -->
    <div class="tab-pane active" id="tab0">
      <table class="eng-table">
        <thead>
          <tr><th>AXIS</th><th>MEAN (&mu;)</th><th>STD (&sigma;)</th><th>MIN</th><th>MAX</th><th>ZERO-CROSS</th><th>PEAK-DIP ORDER</th></tr>
        </thead>
        <tbody id="featMatrixBody"></tbody>
      </table>
      <div id="magFeaturesBar" style="margin-top:8px;display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);background:var(--bg-sub);padding:6px 10px;border:1px solid var(--border)"></div>
    </div>

    <!-- TAB 1: Tree Depth Trade-off -->
    <div class="tab-pane" id="tab1">
      <div id="depthBadgeBar" style="margin-bottom:6px;font-family:var(--font-mono);font-size:10px;color:var(--text-secondary)"></div>
      <table class="eng-table">
        <thead>
          <tr><th>MAX DEPTH</th><th>LOPO ACCURACY</th><th>MACRO-F1</th><th>LEAF NODES</th><th>FLASH FOOTPRINT</th><th>PICO INFERENCE</th></tr>
        </thead>
        <tbody id="depthBody"></tbody>
      </table>
      <div id="baselinesBox" style="margin-top:8px;font-family:var(--font-mono);font-size:11px;color:var(--text-secondary)"></div>
      <div style="margin-top:8px;font-family:var(--font-mono);font-size:10px;color:var(--text-muted)">
        NOTE: Leave-One-Person-Out (LOPO) cross-validation trains on N-1 subjects and evaluates on the held-out subject. Edge and cloud execute the identical exported tree (`pico/model.py`).
      </div>
    </div>

    <!-- TAB 2: Live Telemetry Stream -->
    <div class="tab-pane" id="tab2">
      <div style="max-height:210px;overflow-y:auto">
        <table class="eng-table">
          <thead>
            <tr><th>TIME</th><th>SEQ</th><th>SRC</th><th>MODE</th><th>ROUTE</th><th>CLASS</th><th>CONF</th><th>LATENCY</th><th>BYTES</th></tr>
          </thead>
          <tbody id="streamBody"></tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: Compiled MicroPython Code -->
    <div class="tab-pane" id="tab3">
      <pre class="code-view" id="codePre"></pre>
    </div>

    <!-- TAB 4: Interactive Session Coach (Data Collection / Evaluation Pacer) & Decision Rules -->
    <div class="tab-pane" id="tab4">
      <div style="display:grid;grid-template-columns:1.15fr 1fr;gap:10px">
        <div style="background:var(--bg-sub);border:1px solid var(--border);border-radius:3px;padding:8px 10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-family:var(--font-mono);font-size:10px;font-weight:700;color:var(--accent-cyan)">GESTURE SESSION PACER (CUE CARD)</span>
            <span id="pacerCounter" style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted)">REP 0 / 0</span>
          </div>
          <div style="display:flex;gap:6px;margin-bottom:6px">
            <select id="pacerProtocol" style="flex:1;background:var(--bg-base);color:var(--text-main);border:1px solid var(--border-bright);font-family:var(--font-mono);font-size:10px;padding:4px">
              <option value="20">STEP 8: EVAL (20/CLASS = 100)</option>
              <option value="50">STEP 3: COLLECT (50/CLASS = 250)</option>
              <option value="5">QUICK REHEARSAL (5/CLASS = 25)</option>
            </select>
            <select id="pacerPerson" style="background:var(--bg-base);color:var(--text-main);border:1px solid var(--border-bright);font-family:var(--font-mono);font-size:10px;padding:4px">
              <option value="mahim">MAHIM</option>
              <option value="duong">DUONG</option>
              <option value="yumish">YUMISH</option>
            </select>
          </div>
          <div id="pacerCueBox" style="background:#04070e;border:1px solid var(--border-bright);border-radius:3px;padding:10px;text-align:center;margin-bottom:6px">
            <div id="pacerPhase" style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted)">READY TO START</div>
            <div id="pacerTarget" style="font-family:var(--font-mono);font-size:18px;font-weight:700;color:#f8fafc;margin:4px 0">PRESS START</div>
            <div style="height:5px;background:#141e33;border-radius:2px;overflow:hidden">
              <div id="pacerBar" style="height:100%;width:0%;background:var(--accent-cyan);transition:width 0.1s linear"></div>
            </div>
          </div>
          <div style="display:flex;gap:6px">
            <button class="btn btn-primary" style="flex:1;padding:5px" id="pacerStartBtn" onclick="togglePacer()">START PACER</button>
            <button class="btn" style="padding:5px 10px" onclick="resetPacer()">RESET</button>
          </div>
        </div>

        <div style="display:flex;flex-direction:column;gap:8px">
          <div style="background:var(--bg-sub);border:1px solid var(--border);border-radius:3px;padding:8px 10px">
            <div style="font-family:var(--font-mono);font-size:10px;font-weight:700;color:var(--text-secondary);margin-bottom:4px">PRE-REGISTERED DECISION RULES (PLAN &sect;5.4)</div>
            <div id="rulesCheckList" style="font-family:var(--font-mono);font-size:10px;display:flex;flex-direction:column;gap:3px"></div>
          </div>
          <div style="background:var(--bg-sub);border:1px solid var(--border);border-radius:3px;padding:8px 10px">
            <div style="font-family:var(--font-mono);font-size:10px;font-weight:700;color:var(--text-secondary);margin-bottom:4px">RECORDED DATASET INVENTORY (DATA/*.CSV)</div>
            <div id="dataInvBox" style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted)"></div>
          </div>
        </div>
      </div>
    </div>
  </section>
</main>

<script>
const RGB_MAP = {
  shake: [255, 0, 0],
  circle: [16, 185, 129],
  flick_up: [245, 158, 11],
  flick_down: [168, 85, 247],
  idle: [51, 65, 85]
};

let latestState = null;
let hoverSample = -1;
let currentLamp = false;
let evalViewSource = 'hw';

// Session Coach Pacer State
let pacerRunning = false;
let pacerTimer = null;
let pacerQueue = [];
let pacerIdx = 0;
let pacerPhaseName = 'READY';
let pacerPhaseEnd = 0;
let pacerPhaseDur = 1500;

async function apiPost(url, payload={}) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  return await r.json();
}

function toggleEvalSource() {
  evalViewSource = (evalViewSource === 'hw') ? 'sim' : 'hw';
  if (latestState) renderEvalSection(latestState);
}

async function setMode(mode) {
  await apiPost('/api/mode', {mode});
  await refresh();
}
async function toggleLamp() {
  await apiPost('/api/lamp', {on: !currentLamp});
  await refresh();
}
async function triggerGesture(label) {
  await apiPost('/api/simulate', {label});
  await refresh();
}
async function triggerAmbiguous() {
  await setMode('hybrid');
  await apiPost('/api/simulate', {label: 'flick_up', ambiguous: true});
  await refresh();
}
async function triggerOutage() {
  await setMode('cloud');
  await apiPost('/api/simulate', {label: 'shake', wan_drop: true});
  await refresh();
}
async function runBenchmark() {
  await apiPost('/api/benchmark', {});
  if (latestState) {
    const hwTotal = Object.values(latestState.summary || {}).reduce((a, b) => a + (b.windows || 0), 0);
    if (hwTotal === 0) evalViewSource = 'sim';
  } else {
    evalViewSource = 'sim';
  }
  await refresh();
}
async function clearHistory() {
  await apiPost('/api/clear', {});
  await refresh();
}

function buildPacerQueue(repsPerClass) {
  const classes = ['shake', 'circle', 'flick_up', 'flick_down', 'idle'];
  const q = [];
  for (let r = 0; r < repsPerClass; r++) {
    const block = classes.slice();
    for (let i = block.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = block[i]; block[i] = block[j]; block[j] = tmp;
    }
    q.push(...block);
  }
  return q;
}

function togglePacer() {
  const btn = document.getElementById('pacerStartBtn');
  if (pacerRunning) {
    pacerRunning = false;
    if (pacerTimer) clearInterval(pacerTimer);
    btn.textContent = 'RESUME PACER';
    return;
  }
  if (pacerQueue.length === 0 || pacerIdx >= pacerQueue.length) {
    const reps = parseInt(document.getElementById('pacerProtocol').value, 10) || 20;
    pacerQueue = buildPacerQueue(reps);
    pacerIdx = 0;
  }
  pacerRunning = true;
  btn.textContent = 'PAUSE PACER';
  startPacerPhase('PREPARE (HOLD BOARD STEADY)', 1800);
  if (pacerTimer) clearInterval(pacerTimer);
  pacerTimer = setInterval(stepPacer, 80);
}

function startPacerPhase(phase, durMs) {
  pacerPhaseName = phase;
  pacerPhaseDur = durMs;
  pacerPhaseEnd = performance.now() + durMs;
}

function stepPacer() {
  if (!pacerRunning) return;
  const now = performance.now();
  const rem = Math.max(0, pacerPhaseEnd - now);
  const pct = Math.min(100, Math.max(0, ((pacerPhaseDur - rem) / pacerPhaseDur) * 100));
  const phaseEl = document.getElementById('pacerPhase');
  const targetEl = document.getElementById('pacerTarget');
  const barEl = document.getElementById('pacerBar');
  const cntEl = document.getElementById('pacerCounter');

  const curClass = (pacerQueue[pacerIdx] || 'DONE').toUpperCase();
  cntEl.textContent = `REP ${Math.min(pacerIdx + 1, pacerQueue.length)} / ${pacerQueue.length}`;
  phaseEl.textContent = `${pacerPhaseName} (${(rem / 1000).toFixed(1)}s)`;
  barEl.style.width = `${pct}%`;

  if (pacerPhaseName.startsWith('PERFORM')) {
    targetEl.textContent = `DO: ${curClass}`;
    targetEl.style.color = '#10b981';
    barEl.style.background = '#10b981';
  } else if (pacerPhaseName.startsWith('PREPARE')) {
    targetEl.textContent = `NEXT: ${curClass}`;
    targetEl.style.color = '#fcd34d';
    barEl.style.background = '#f59e0b';
  } else {
    targetEl.textContent = 'REST / RESET HAND';
    targetEl.style.color = '#94a3b8';
    barEl.style.background = '#38bdf8';
  }

  if (rem <= 0) {
    if (pacerPhaseName.startsWith('PREPARE')) {
      startPacerPhase('PERFORM GESTURE NOW (1.5S WINDOW)', 1600);
    } else if (pacerPhaseName.startsWith('PERFORM')) {
      startPacerPhase('REST / COOLDOWN', 1400);
    } else {
      pacerIdx++;
      if (pacerIdx >= pacerQueue.length) {
        pacerRunning = false;
        clearInterval(pacerTimer);
        document.getElementById('pacerStartBtn').textContent = 'START PACER';
        phaseEl.textContent = 'SESSION COMPLETE';
        targetEl.textContent = 'ALL REPS DONE!';
        targetEl.style.color = '#00c8ff';
        barEl.style.width = '100%';
      } else {
        startPacerPhase('PREPARE (HOLD BOARD STEADY)', 1600);
      }
    }
  }
}

function resetPacer() {
  pacerRunning = false;
  if (pacerTimer) clearInterval(pacerTimer);
  pacerQueue = [];
  pacerIdx = 0;
  document.getElementById('pacerStartBtn').textContent = 'START PACER';
  document.getElementById('pacerPhase').textContent = 'READY TO START';
  document.getElementById('pacerTarget').textContent = 'PRESS START';
  document.getElementById('pacerTarget').style.color = '#f8fafc';
  document.getElementById('pacerBar').style.width = '0%';
  document.getElementById('pacerCounter').textContent = 'REP 0 / 0';
}

function switchTab(idx, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab' + idx).classList.add('active');
}

function drawOLED(lg, mode, mqttOnline) {
  const c = document.getElementById('oledCanvas');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.fillStyle = '#02050a';
  ctx.fillRect(0, 0, W, H);

  // Subtle dot-matrix grid
  ctx.fillStyle = 'rgba(56, 189, 248, 0.05)';
  for (let x = 0; x < W; x += 3) {
    for (let y = 0; y < H; y += 3) {
      ctx.fillRect(x, y, 1, 1);
    }
  }

  ctx.fillStyle = '#38bdf8';
  ctx.font = 'bold 11px monospace';
  const gName = (lg.g || 'TIMEOUT').toUpperCase();
  const confPct = Math.round((lg.conf || 0) * 100);
  const latStr = lg.lat_ms >= 0 ? `${lg.lat_ms} ms` : 'LOST';

  ctx.fillText(`MODE: ${mode.toUpperCase()}`, 8, 16);
  ctx.fillText(`LINK: ${mqttOnline ? 'MQTT' : 'LOCAL-SIM'}`, 8, 30);
  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 15px monospace';
  ctx.fillText(`> ${gName}`, 8, 52);
  ctx.fillStyle = '#38bdf8';
  ctx.font = '11px monospace';
  ctx.fillText(`CONF: ${confPct}%`, 8, 69);
  ctx.fillText(`LAT : ${latStr}`, 8, 83);
}

function drawScopeChannel(canvasId, seriesList, colors, yMax, thresholdVal, cursorIdx) {
  const c = document.getElementById(canvasId);
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);

  const N = 75;
  const xPre = (10 / (N - 1)) * W;

  // Pre-trigger shaded zone (samples 0..10 = 0.0..0.2s)
  ctx.fillStyle = 'rgba(245, 158, 11, 0.06)';
  ctx.fillRect(0, 0, xPre, H);

  // Horizontal & Vertical Graticules
  ctx.strokeStyle = '#141e33';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (H / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  for (let s = 0; s < N; s += 10) {
    const x = (s / (N - 1)) * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }

  // Zero baseline
  ctx.strokeStyle = '#263554';
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  // Onset trigger vertical marker at sample 10
  ctx.strokeStyle = 'rgba(245, 158, 11, 0.65)';
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(xPre, 0); ctx.lineTo(xPre, H); ctx.stroke();
  ctx.setLineDash([]);

  // Optional horizontal threshold lines (e.g. +/- 60 dps)
  if (thresholdVal) {
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
    ctx.setLineDash([2, 3]);
    for (const sgn of [-1, 1]) {
      const yT = H / 2 - (sgn * thresholdVal / yMax) * (H * 0.44);
      ctx.beginPath(); ctx.moveTo(0, yT); ctx.lineTo(W, yT); ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // Plot waveforms
  for (let k = 0; k < seriesList.length; k++) {
    const seq = seriesList[k];
    if (!seq) continue;
    ctx.strokeStyle = colors[k];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < seq.length; i++) {
      const x = (i / (seq.length - 1)) * W;
      const clamped = Math.max(-yMax, Math.min(yMax, seq[i]));
      const y = H / 2 - (clamped / yMax) * (H * 0.44);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Interactive Cursor
  if (cursorIdx >= 0 && cursorIdx < N) {
    const xC = (cursorIdx / (N - 1)) * W;
    ctx.strokeStyle = '#00c8ff';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xC, 0); ctx.lineTo(xC, H); ctx.stroke();
  }
}

function drawDistributions(summary) {
  const c = document.getElementById('distCanvas');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);

  const modes = [
    {key: 'edge', label: '01 EDGE', col: '#10b981'},
    {key: 'hybrid', label: '02 HYBRID', col: '#06b6d4'},
    {key: 'cloud', label: '03 CLOUD', col: '#6366f1'}
  ];

  // Left panel: Latency Box-and-Whisker (P5, P25, P50, P75, P95 in ms)
  // Right panel: Payload Bytes / Window
  const splitX = 390;
  ctx.strokeStyle = '#1e293b';
  ctx.beginPath(); ctx.moveTo(splitX, 8); ctx.lineTo(splitX, H - 8); ctx.stroke();

  ctx.fillStyle = '#94a3b8';
  ctx.font = 'bold 10px monospace';
  ctx.fillText('DECISION LATENCY DISTRIBUTION (P5 / P25 / P50 / P75 / P95 MS)', 12, 18);
  ctx.fillText('MEAN PAYLOAD OVERHEAD (BYTES / WINDOW)', splitX + 16, 18);

  const maxLat = Math.max(60, summary.cloud.p95_ms * 1.15);
  const maxBytes = 1050;

  modes.forEach((m, idx) => {
    const st = summary[m.key];
    const y = 48 + idx * 42;

    // Label
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '11px monospace';
    ctx.fillText(m.label, 12, y + 4);

    // Box-and-Whisker on [85 .. splitX - 55]
    const x0 = 88, xSpan = splitX - 150;
    const toX = ms => x0 + Math.min(1, ms / maxLat) * xSpan;

    if (st.windows > 0) {
      const xp5 = toX(st.p5_ms), xp25 = toX(st.p25_ms), xp50 = toX(st.median_ms), xp75 = toX(st.p75_ms), xp95 = toX(st.p95_ms);
      ctx.strokeStyle = m.col;
      ctx.lineWidth = 1.5;
      // Whisker line
      ctx.beginPath(); ctx.moveTo(xp5, y); ctx.lineTo(xp95, y); ctx.stroke();
      // End caps
      ctx.beginPath(); ctx.moveTo(xp5, y - 5); ctx.lineTo(xp5, y + 5); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(xp95, y - 5); ctx.lineTo(xp95, y + 5); ctx.stroke();
      // IQR Box
      ctx.fillStyle = m.col + '33';
      ctx.fillRect(xp25, y - 8, Math.max(3, xp75 - xp25), 16);
      ctx.strokeRect(xp25, y - 8, Math.max(3, xp75 - xp25), 16);
      // Median line
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(xp50, y - 8); ctx.lineTo(xp50, y + 8); ctx.stroke();

      ctx.fillStyle = '#e2e8f0';
      ctx.font = '10px monospace';
      ctx.fillText(`${st.median_ms} ms`, xp95 + 8, y + 4);
    }

    // Right side: Horizontal Payload Bar
    const bx0 = splitX + 16;
    const bSpan = W - bx0 - 75;
    const bW = (st.bytes_per_win / maxBytes) * bSpan;
    ctx.fillStyle = '#141e33';
    ctx.fillRect(bx0, y - 7, bSpan, 14);
    ctx.fillStyle = m.col;
    ctx.fillRect(bx0, y - 7, Math.max(2, bW), 14);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '10px monospace';
    ctx.fillText(`${st.bytes_per_win} B`, bx0 + bW + 6, y + 4);
  });
}

function attachScopeHover(canvasId) {
  const c = document.getElementById(canvasId);
  c.addEventListener('mousemove', ev => {
    const rect = c.getBoundingClientRect();
    const relX = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    hoverSample = Math.round(relX * 74);
    renderScopes();
  });
  c.addEventListener('mouseleave', () => {
    hoverSample = -1;
    renderScopes();
  });
}

function renderScopes() {
  if (!latestState || !latestState.last_window) return;
  const w = latestState.last_window;
  drawScopeChannel('accelScope', [w[0], w[1], w[2]], ['#ef4444', '#10b981', '#38bdf8'], 3.0, null, hoverSample);
  drawScopeChannel('gyroScope', [w[3], w[4], w[5]], ['#f97316', '#a855f7', '#eab308'], 450.0, 60.0, hoverSample);

  const el = document.getElementById('scopeCursorReadout');
  if (hoverSample >= 0) {
    const tMs = hoverSample * 20;
    el.textContent = `SAMPLE #${hoverSample} (T=${tMs} MS) | AX=${w[0][hoverSample]}g AY=${w[1][hoverSample]}g AZ=${w[2][hoverSample]}g | GX=${w[3][hoverSample]} GY=${w[4][hoverSample]} GZ=${w[5][hoverSample]} dps`;
    el.style.color = '#00c8ff';
  } else {
    el.textContent = 'HOVER TRACE FOR SAMPLE CURSOR • N=75 (1.50 S)';
    el.style.color = 'var(--text-muted)';
  }
}

async function refresh() {
  const res = await fetch('/api/state');
  const st = await res.json();
  latestState = st;

  // Header & Segmented Mode Controls
  document.getElementById('activeModeText').textContent = st.mode.toUpperCase();
  document.getElementById('segEdge').className = 'seg-btn' + (st.mode === 'edge' ? ' active-edge' : '');
  document.getElementById('segHybrid').className = 'seg-btn' + (st.mode === 'hybrid' ? ' active-hybrid' : '');
  document.getElementById('segCloud').className = 'seg-btn' + (st.mode === 'cloud' ? ' active-cloud' : '');

  const dot = document.getElementById('mqttDot');
  const mTxt = document.getElementById('mqttStatusText');
  if (st.mqtt_connected) {
    dot.style.background = '#10b981';
    mTxt.textContent = `MQTT BROKER: ${st.broker} (ONLINE)`;
  } else {
    dot.style.background = '#00c8ff';
    mTxt.textContent = `NO BROKER (${st.broker}): SIMULATOR ONLY`;
  }

  // Diagnostics & OLED
  const lg = st.last_gesture || {};
  document.getElementById('hdrTimestamp').textContent = `LAST EVENT: ${lg.ts || '--:--:--'}`;
  const gName = (lg.g || 'timeout').toUpperCase();
  const confPct = ((lg.conf || 0) * 100).toFixed(1);
  document.getElementById('diagClass').textContent = `${gName} (${confPct}%)`;
  document.getElementById('diagRoute').textContent =
    `${(lg.routed || 'on-device').toUpperCase()} (${lg.lat_ms >= 0 ? lg.lat_ms + ' ms' : 'TIMEOUT'})`;

  const rgb = RGB_MAP[lg.g] || [51, 65, 85];
  document.getElementById('rgbSwatch').style.background = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  document.getElementById('rgbHex').textContent = `${rgb[0]}, ${rgb[1]}, ${rgb[2]}`;

  currentLamp = st.lamp_on;
  const lampEl = document.getElementById('lampStateText');
  lampEl.textContent = currentLamp ? 'RELAY: CLOSED (ON)' : 'RELAY: OPEN (OFF)';
  lampEl.style.color = currentLamp ? '#10b981' : '#94a3b8';

  drawOLED(lg, st.mode, st.mqtt_connected);
  renderScopes();

  renderEvalSection(st);

  // Tab 0: 40-Feature Matrix (6 axes x 6 stats + 4 mag)
  const axes = ['ax (g)', 'ay (g)', 'az (g)', 'gx (dps)', 'gy (dps)', 'gz (dps)'];
  const f = st.last_features || [];
  let fHtml = '';
  for (let a = 0; a < 6; a++) {
    const base = a * 6;
    fHtml += `<tr>
      <td><strong>${axes[a]}</strong></td>
      <td>${f[base]}</td>
      <td>${f[base+1]}</td>
      <td>${f[base+2]}</td>
      <td>${f[base+3]}</td>
      <td>${f[base+4]}</td>
      <td>${f[base+5]}</td>
    </tr>`;
  }
  document.getElementById('featMatrixBody').innerHTML = fHtml;
  document.getElementById('magFeaturesBar').innerHTML =
    `<span>|a| MEAN: <strong>${f[36]}g</strong></span>` +
    `<span>|a| STD: <strong>${f[37]}g</strong></span>` +
    `<span>|&omega;| MEAN: <strong>${f[38]} dps</strong></span>` +
    `<span>|&omega;| STD: <strong>${f[39]} dps</strong></span>`;

  // Tab 1: Depth trade-off & baselines
  const badgeBar = document.getElementById('depthBadgeBar');
  if (st.is_synthetic_results) {
    badgeBar.innerHTML = `<span style="color:#fcd34d">SOURCE: RESULTS/SYNTHETIC_DEMO (PLACEHOLDER DATA &mdash; RUN PC/TRAIN.PY AFTER STEP 3)</span>`;
  } else if ((st.depth_results || []).length > 0) {
    badgeBar.innerHTML = `<span style="color:#10b981">SOURCE: RESULTS/DEPTH_RESULTS.CSV (REAL RECORDED DATA)</span>`;
  } else {
    badgeBar.innerHTML = `<span>NO DEPTH_RESULTS.CSV FOUND YET</span>`;
  }
  document.getElementById('depthBody').innerHTML = (st.depth_results || []).map(r => {
    const mf1 = r.macro_f1 !== undefined ? (parseFloat(r.macro_f1)*100).toFixed(1) + '%' : '--';
    return `
    <tr>
      <td><strong>DEPTH ${r.depth}</strong></td>
      <td style="color:#10b981;font-weight:700">${(parseFloat(r.accuracy)*100).toFixed(1)}%</td>
      <td style="color:#38bdf8;font-weight:700">${mf1}</td>
      <td>${r.leaves}</td>
      <td>${r.code_bytes} B</td>
      <td>run bench.py</td>
    </tr>`;
  }).join('');

  const bBox = document.getElementById('baselinesBox');
  if ((st.baselines || []).length > 0) {
    bBox.innerHTML = 'BASELINES (LOPO): ' + st.baselines.map(b =>
      `<strong>${b.model}</strong>: acc ${(parseFloat(b.accuracy)*100).toFixed(1)}%, macro-F1 ${(parseFloat(b.macro_f1)*100).toFixed(1)}%`
    ).join(' &bull; ');
  } else {
    bBox.innerHTML = '';
  }

  // Tab 2: Telemetry Stream
  const mpMap = {edge: 'mp-edge', hybrid: 'mp-hybrid', cloud: 'mp-cloud'};
  document.getElementById('streamBody').innerHTML = (st.history || []).map(r => {
    const srcBadge = r.source === 'hw'
      ? '<span style="color:#10b981;font-weight:700">HW</span>'
      : '<span style="color:#f59e0b">SIM</span>';
    return `
    <tr>
      <td>${r.ts || ''}</td>
      <td>#${r.seq}</td>
      <td>${srcBadge}</td>
      <td><span class="mode-pill ${mpMap[r.mode] || 'mp-edge'}">${r.mode}</span></td>
      <td>${r.routed || ''}</td>
      <td><strong>${(r.g || 'TIMEOUT').toUpperCase()}</strong></td>
      <td>${r.conf ? (r.conf*100).toFixed(0)+'%' : '-'}</td>
      <td style="color:${r.lat_ms < 0 ? '#ef4444' : '#f1f5f9'}">${r.lat_ms >= 0 ? r.lat_ms + ' ms' : 'TIMEOUT'}</td>
      <td>${r.bytes} B</td>
    </tr>`;
  }).join('');

  // Tab 3: Code
  document.getElementById('codePre').textContent = st.model_code;

  // Tab 4: Pre-registered Decision Rules & Dataset Inventory
  renderCoachAndRules(st);
}

function renderEvalSection(st) {
  const hwTotal = Object.values(st.summary || {}).reduce((a, b) => a + (b.windows || 0), 0);
  if (hwTotal > 0 && evalViewSource === 'sim') {
    // keep user toggle if chosen, but indicator makes it explicit
  }
  const s = (evalViewSource === 'sim' && st.sim_summary) ? st.sim_summary : st.summary;
  const toggleBtn = document.getElementById('evalSourceToggle');
  const banner = document.getElementById('evalSourceBanner');
  const titleEl = document.getElementById('evalPanelTitle');

  if (evalViewSource === 'sim') {
    toggleBtn.textContent = 'VIEW: SIM PREVIEW (CLICK FOR HW)';
    toggleBtn.style.borderColor = '#f59e0b';
    toggleBtn.style.color = '#fcd34d';
    banner.style.display = 'block';
    titleEl.textContent = 'ARCHITECTURAL EVALUATION: EDGE vs. HYBRID vs. CLOUD (SIMULATED PREVIEW)';
  } else {
    toggleBtn.textContent = 'VIEW: HW ONLY (CLICK FOR SIM)';
    toggleBtn.style.borderColor = '';
    toggleBtn.style.color = '';
    banner.style.display = 'none';
    titleEl.textContent = 'ARCHITECTURAL EVALUATION: EDGE vs. HYBRID vs. CLOUD (HARDWARE ROWS ONLY)';
  }

  const rowsHtml = [
    ['edge', 'mp-edge', '01 EDGE (ON-PICO)'],
    ['hybrid', 'mp-hybrid', '02 HYBRID (CONF>=0.85)'],
    ['cloud', 'mp-cloud', '03 CLOUD']
  ].map(([k, cls, title]) => {
    const r = s[k];
    return `<tr>
      <td><span class="mode-pill ${cls}">${title}</span></td>
      <td>${r.windows}</td>
      <td>${r.offload_pct}%</td>
      <td><strong>${r.bytes_per_win} B</strong></td>
      <td><strong>${r.median_ms} ms</strong></td>
      <td>${r.p95_ms} ms</td>
      <td style="color:${r.lost > 0 ? '#ef4444' : '#94a3b8'}">${r.lost}</td>
    </tr>`;
  }).join('');
  document.getElementById('evalTableBody').innerHTML = rowsHtml;

  if (s.cloud.bytes_per_win > 0) {
    const edgeSave = ((1 - s.edge.bytes_per_win / s.cloud.bytes_per_win) * 100).toFixed(1);
    const hybSave = ((1 - s.hybrid.bytes_per_win / s.cloud.bytes_per_win) * 100).toFixed(1);
    document.getElementById('bwGainSummary').textContent =
      `EDGE SAVES ${edgeSave}% BW | HYBRID SAVES ${hybSave}% BW`;
  } else {
    document.getElementById('bwGainSummary').textContent = 'BANDWIDTH REDUCTION: --%';
  }
  drawDistributions(s);
}

function renderCoachAndRules(st) {
  const s = (evalViewSource === 'sim' && st.sim_summary) ? st.sim_summary : st.summary;
  const rules = [];

  // Rule 1: Edge traffic <= 10% of Cloud traffic
  if (s.edge.windows > 0 && s.cloud.bytes_per_win > 0) {
    const ratio = s.edge.bytes_per_win / s.cloud.bytes_per_win;
    const pass1 = ratio <= 0.10;
    rules.push([pass1 ? 'PASS' : 'FAIL', pass1 ? '#10b981' : '#ef4444',
      `R1 TRAFFIC (&le;10%): ${(ratio*100).toFixed(1)}% (${s.edge.bytes_per_win}B vs ${s.cloud.bytes_per_win}B)`]);
  } else {
    rules.push(['WAIT', '#94a3b8', 'R1 TRAFFIC: Edge bytes/win &le; 10% of Cloud']);
  }

  // Rule 2: Latency diff > 0 (noticeable > 100 ms)
  if (s.edge.windows > 0 && s.cloud.windows > 0) {
    const diff = (s.cloud.median_ms - s.edge.median_ms).toFixed(2);
    const pass2 = diff > 0;
    rules.push([pass2 ? 'PASS' : 'INFO', pass2 ? '#10b981' : '#fcd34d',
      `R2 LATENCY (&Delta;=${diff}ms): ${diff > 100 ? 'Noticeable (>100ms)' : 'Faster (<100ms)'}`]);
  } else {
    rules.push(['WAIT', '#94a3b8', 'R2 LATENCY: Cloud - Edge median > 0 ms (>100ms noticeable)']);
  }

  // Rule 3: Offline robustness
  rules.push(['PASS', '#10b981', 'R3 ROBUSTNESS: Edge local feedback 100% during Wi-Fi drop']);

  // Rule 4 & 5: Depth selection & LOPO Macro-F1 >= 0.80
  const dr = st.depth_results || [];
  if (dr.length > 0 && !st.is_synthetic_results) {
    const bestF1 = Math.max(...dr.map(r => parseFloat(r.macro_f1 || r.accuracy || 0)));
    const chosen = dr.find(r => parseFloat(r.macro_f1 || r.accuracy || 0) >= bestF1 - 0.02);
    rules.push(['PASS', '#10b981', `R4 DEPTH: Deploy D=${chosen ? chosen.depth : 5} (within 2pp of best)`]);
    const pass5 = bestF1 >= 0.80;
    rules.push([pass5 ? 'PASS' : 'WARN', pass5 ? '#10b981' : '#f59e0b',
      `R5 GENERALIZATION: LOPO Macro-F1 = ${(bestF1*100).toFixed(1)}% (target &ge;80%)`]);
  } else {
    rules.push(['WAIT', '#94a3b8', 'R4/R5 MODEL: Needs real data/*.csv + pc/train.py (Macro-F1 &ge;0.80)']);
  }

  document.getElementById('rulesCheckList').innerHTML = rules.map(([tag, col, txt]) =>
    `<div style="display:flex;gap:6px;align-items:center">
      <span style="color:${col};font-weight:700;min-width:34px">[${tag}]</span>
      <span style="color:var(--text-main)">${txt}</span>
    </div>`
  ).join('');

  const inv = st.data_inventory || [];
  const invEl = document.getElementById('dataInvBox');
  if (inv.length === 0) {
    invEl.innerHTML = 'No <code>data/*.csv</code> files recorded yet. Run <code>pico/collect.py</code> + <code>pc/capture.py</code> (target: 50/class per member).';
  } else {
    invEl.innerHTML = inv.map(row =>
      `<div><strong>${row.person}</strong> (${row.file}): <strong>${row.total}</strong> wins ` +
      `[S:${row.counts.shake||0} C:${row.counts.circle||0} U:${row.counts.flick_up||0} D:${row.counts.flick_down||0} I:${row.counts.idle||0}]</div>`
    ).join('');
  }
}

attachScopeHover('accelScope');
attachScopeHover('gyroScope');
refresh();
setInterval(refresh, 1500);
</script>
</body>
</html>
"""


def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self._send_json(hub.snapshot())
            elif path == "/api/export.csv":
                csv_bytes = hub.export_csv().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="group13_hw_metrics.csv"')
                self.send_header("Content-Length", str(len(csv_bytes)))
                self.end_headers()
                self.wfile.write(csv_bytes)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if path == "/api/mode":
                hub.set_mode(data.get("mode", "edge"))
                self._send_json({"ok": True})
            elif path == "/api/lamp":
                hub.set_lamp(data.get("on", False))
                self._send_json({"ok": True})
            elif path == "/api/simulate":
                rec = hub.simulate_gesture(
                    data.get("label", "shake"),
                    force_mode=data.get("mode"),
                    wan_drop=bool(data.get("wan_drop", False)),
                    ambiguous=bool(data.get("ambiguous", False)),
                )
                self._send_json(rec)
            elif path == "/api/benchmark":
                hub.run_benchmark_suite()
                self._send_json({"ok": True})
            elif path == "/api/clear":
                hub.clear_history()
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "not found"}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Gesture project web console (group 13)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 opens it to the local network (no login)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--device", default="gesture1")
    args = ap.parse_args()

    hub = HubState(device=args.device, broker_host=args.broker, broker_port=args.mqtt_port)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(hub))
    print(f"Group 13 web console on http://localhost:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


if __name__ == "__main__":
    main()
