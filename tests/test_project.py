"""PC tests for the gesture project. No Pico or HAT needed.

Run:  python test_project.py
Needs: pip install numpy scikit-learn matplotlib paho-mqtt pyserial
       (amqtt for the broker test, mpy-cross for the MicroPython syntax check; both optional)

The HAT, Wi-Fi and umqtt are replaced by fakes. Synthetic gestures stand in
for real recordings, so these tests show the code works, not that the real
model will be accurate.
"""

import importlib.util
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PICO = os.path.join(ROOT, "pico")
PC = os.path.join(ROOT, "pc")
sys.path[:0] = [PICO, PC]

# --- MicroPython stand-ins -------------------------------------------------
for name, fn in {"sleep_us": lambda us: None, "sleep_ms": lambda ms: None,
                 "ticks_ms": lambda: int(time.perf_counter() * 1000),
                 "ticks_us": lambda: int(time.perf_counter() * 1e6),
                 "ticks_diff": lambda a, b: a - b, "ticks_add": lambda a, b: a + b}.items():
    setattr(time, name, fn)


class FakeI2C:
    def __init__(self, who=0x67):
        self.mem = {0x00: 0x08, 0x75: who}
        self.writes = []

    def writeto_mem(self, addr, reg, data):
        self.writes.append((addr, reg, data[0]))
        self.mem[reg] = data[0]

    def readfrom_mem(self, addr, reg, n):
        return bytes(self.mem.get(reg + i, 0) for i in range(n))

    def readfrom_mem_into(self, addr, reg, buf):
        buf[:] = self.readfrom_mem(addr, reg, len(buf))


class _Dummy:
    IN = OUT = ONE_SHOT = 0

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: 0


machine = types.ModuleType("machine")
machine.I2C = FakeI2C
machine.Pin = machine.PWM = machine.Timer = _Dummy
sys.modules["machine"] = machine

import features  # noqa: E402
import hat  # noqa: E402
import main as app_mod  # noqa: E402
import collect  # noqa: E402
import capture  # noqa: E402
import train  # noqa: E402
import cloud_classifier  # noqa: E402
import plots  # noqa: E402

RESULTS = []


def test(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, True, ""))
        print("PASS", fn.__name__, flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        RESULTS.append((fn.__name__, False, repr(e)))
        print("FAIL", fn.__name__, e, flush=True)
    return fn


# --- synthetic gestures ----------------------------------------------------
def gauss(t, c, w):
    return math.exp(-((t - c) / w) ** 2)


def synth(label, rng, amp=1.0, speed=1.0):
    n, fs = features.WINDOW, features.SAMPLE_RATE_HZ
    w = [[0.0] * n for _ in range(6)]
    c = rng.uniform(0.5, 1.0)
    phase = rng.uniform(0, 2 * math.pi)
    for i in range(n):
        t = i / fs
        ax, ay, az, gx, gy, gz = 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        if label == "idle":
            # an ordinary slow movement that is not a gesture (still passes the onset threshold)
            ax = 0.2 * math.sin(2 * math.pi * 0.5 * t + phase)
            gy = 90 * amp * math.sin(2 * math.pi * 0.6 * t + phase)
        elif label == "shake":
            env = gauss(t, 0.75, 0.45)
            ax = 1.5 * amp * env * math.sin(2 * math.pi * 5 * speed * t + phase)
            gz = 300 * amp * env * math.cos(2 * math.pi * 5 * speed * t + phase)
        elif label == "circle":
            ax = 0.8 * amp * math.sin(2 * math.pi * speed * t + phase)
            ay = 0.8 * amp * math.cos(2 * math.pi * speed * t + phase)
            gx = 120 * amp * math.cos(2 * math.pi * speed * t + phase)
            gy = -120 * amp * math.sin(2 * math.pi * speed * t + phase)
        else:
            s = 1 if label == "flick_up" else -1
            gx = s * amp * (400 * gauss(t, c, 0.08 / speed) - 150 * gauss(t, c + 0.2, 0.1))
            az = 1 + s * amp * (1.2 * gauss(t, c, 0.08) - 0.6 * gauss(t, c + 0.15, 0.08))
        vals = (ax, ay, az, gx, gy, gz)
        for k in range(6):
            w[k][i] = vals[k] + rng.gauss(0, 0.03 if k < 3 else 3)
    return w


STILL = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def gesture_stream(label, rng, amp=1.0, speed=1.0):
    """Device held still, then the gesture, then still again, as a list of samples."""
    w = synth(label, rng, amp, speed)
    g = [tuple(w[k][i] for k in range(6)) for i in range(features.WINDOW)]
    return [STILL] * 100 + g + [STILL] * 100


def segment(samples):
    seg = features.Segmenter()
    return [w for w in (seg.push(s) for s in samples) if w is not None]


def make_dataset(folder, people=("p1", "p2", "p3"), per_class=40):
    """Recordings made the way collect.py makes them: the Segmenter cuts the window."""
    rng = random.Random(1)
    paths = []
    for p in people:
        amp, speed = rng.uniform(0.8, 1.2), rng.uniform(0.85, 1.15)
        path = os.path.join(folder, p + ".csv")
        with open(path, "w") as f:
            seq = 0
            for label in features.CLASSES:
                for _ in range(per_class):
                    window = segment(gesture_stream(label, rng, amp, speed))[0]
                    line = collect.to_line(label, seq, window)
                    f.write(capture.to_csv_row(line, p)[1])
                    seq += 1
        paths.append(path)
    return paths


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TMP = tempfile.mkdtemp(prefix="gesture-test-")
STATE = {}


# --- tests -----------------------------------------------------------------
@test
def imu_driver_setup_and_read():
    i2c = FakeI2C()
    imu = hat.ICM42670(i2c, odr_hz=50, accel_g=8, gyro_dps=1000)
    regs = [(r, v) for _, r, v in i2c.writes]
    assert regs[0] == (0x02, 0x10), regs           # soft reset
    assert (0x1F, 0x0F) in regs                     # accel + gyro low-noise
    assert (0x21, (0x01 << 5) | 0x0A) in regs       # accel 8 g, 50 Hz
    assert (0x20, (0x01 << 5) | 0x0A) in regs       # gyro 1000 dps, 50 Hz
    # temp, ax=+1 g (4096), ay=-0.5 g, az=0, gx=+32.8 raw -> 1 dps... use 328 -> 10 dps
    import struct
    raw = struct.pack(">7h", 0, 4096, -2048, 0, 328, -328, 3280)
    for i, b in enumerate(raw):
        i2c.mem[0x09 + i] = b
    ax, ay, az, gx, gy, gz = imu.read()
    assert (ax, ay, az) == (1.0, -0.5, 0.0)
    assert abs(gx - 10) < 1e-9 and abs(gy + 10) < 1e-9 and abs(gz - 100) < 1e-9


@test
def imu_driver_rejects_wrong_chip():
    try:
        hat.ICM42670(FakeI2C(who=0x12))
    except OSError as e:
        assert "0x12" in str(e)
    else:
        raise AssertionError("no error for wrong WHO_AM_I")


@test
def features_match_numpy():
    import numpy as np
    rng = random.Random(3)
    w = synth("circle", rng)
    f = features.extract(w)
    assert len(f) == len(features.FEATURE_NAMES) == 40
    for k in range(6):
        a = np.array(w[k])
        mean, std, lo, hi = f[k * 6:k * 6 + 4]
        assert abs(mean - a.mean()) < 1e-9 and abs(std - a.std()) < 1e-6
        assert lo == a.min() and hi == a.max()
    acc = np.sqrt(np.array(w[0]) ** 2 + np.array(w[1]) ** 2 + np.array(w[2]) ** 2)
    assert abs(f[36] - acc.mean()) < 1e-9


@test
def ring_buffer_keeps_time_order():
    r = features.Ring(4)
    for i in range(6):
        r.push([i] * 6)
    assert r.full()
    assert r.window()[0] == [2, 3, 4, 5]


@test
def segmenter_lines_up_windows_on_movement_start():
    onset = [STILL] * 50 + [(0, 0, 1, 100.0, 0, 0)] + [STILL] * 200
    wins = segment(onset)
    assert len(wins) == 1
    w = wins[0]
    assert len(w[3]) == features.WINDOW
    assert w[3][features.PRE_TRIGGER] == 100.0 and max(w[3][:features.PRE_TRIGGER]) == 0
    # nothing happens for a still device, and slow movement below the threshold is ignored
    assert segment([STILL] * 300) == []
    assert segment([(0, 0, 1, 40.0, 0, 0)] * 300) == []


@test
def train_export_and_generated_model_matches_sklearn():
    paths = make_dataset(TMP)
    model_path = os.path.join(TMP, "model.py")
    tree, X, y, rows = train.run(paths, 5, os.path.join(TMP, "results"), model_path)
    by_depth = {r["depth"]: r for r in rows}
    assert by_depth[5]["accuracy"] > 0.9, rows
    for d in (3, 5, 8):
        assert os.path.exists(os.path.join(TMP, "results", "model_d%d.py" % d))
        assert os.path.exists(os.path.join(TMP, "results", "confusion_depth%d.png" % d))
    assert all("macro_f1" in r for r in rows)
    for name in ("per_person.csv", "baselines.csv"):
        assert os.path.exists(os.path.join(TMP, "results", name)), name
    import csv
    with open(os.path.join(TMP, "results", "baselines.csv")) as f:
        base = {r["model"]: float(r["accuracy"]) for r in csv.DictReader(f)}
    assert abs(base["majority class"] - 0.2) < 0.01, base          # 5 balanced classes
    assert base["rotation-energy thresholds"] < by_depth[5]["accuracy"], base
    model = load_module(model_path, "gen_model")
    pred = tree.predict(X)
    for i in range(len(X)):
        idx, conf = model.predict(list(X[i]))
        assert model.CLASSES[idx] == pred[i], i
        assert 0 < conf <= 1
    STATE["model"] = model
    STATE["model_path"] = model_path


class FakeNet:
    def __init__(self):
        self.online = True
        self.sent = []

    def publish(self, topic, msg):
        if not self.online:
            return False
        self.sent.append((topic, msg))
        return True

    def poll(self, now):
        pass


class FakeUI:
    def __init__(self):
        self.gestures = []

    def status(self, mode, online):
        pass

    def gesture(self, name, conf, mode, lat):
        self.gestures.append((name, mode))


class FakeIMU:
    def __init__(self, samples):
        self.samples = iter(samples)

    def read(self):
        return next(self.samples)


class ManualClock:
    def __init__(self):
        self.now = 0

    def ticks_ms(self):
        return self.now

    def ticks_diff(self, a, b):
        return a - b


def make_app(samples, clock=None):
    model = STATE["model"]
    net, ui = FakeNet(), FakeUI()
    app = app_mod.GestureApp(FakeIMU(samples), ui, net, model.predict, model.CLASSES,
                             clock or ManualClock(), device="g1",
                             actions={"flick_up": ("iot/pico2/command", "on")})
    return app, net, ui


@test
def edge_mode_recognises_gestures():
    rng = random.Random(7)
    for label in ("shake", "circle", "flick_up"):
        samples = gesture_stream(label, rng)
        app, net, ui = make_app(samples)
        for _ in samples:
            app.step()
        assert ui.gestures and ui.gestures[0] == (label, "edge"), (label, ui.gestures)
        events = [m for t, m in net.sent if t == "gesture/g1/event"]
        assert events and '"%s"' % label in events[0]
        assert not any(t == "gesture/g1/raw" for t, _ in net.sent)
        if label == "flick_up":
            assert ("iot/pico2/command", "on") in net.sent


@test
def edge_mode_ignores_still_device():
    samples = [(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] * 300
    app, net, ui = make_app(samples)
    for _ in samples:
        app.step()
    assert ui.gestures == [] and net.sent == []


@test
def cloud_mode_round_trip_through_classifier():
    rng = random.Random(8)
    samples = gesture_stream("shake", rng)
    clock = ManualClock()
    app, net, ui = make_app(samples, clock)
    app.set_mode("cloud")
    model = STATE["model"]
    for _ in samples:
        clock.now += 20
        app.step()
        # play the PC classifier: answer every raw window 15 ms later
        for topic, payload in [m for m in net.sent if m[0] == "gesture/g1/raw"]:
            net.sent.remove((topic, payload))
            assert len(payload) == 3 + 6 * features.WINDOW * 2
            reply = cloud_classifier.classify(model, payload)
            clock.now += 15
            app.on_message(b"gesture/g1/label", reply.encode())
    assert ui.gestures and ui.gestures[0] == ("shake", "cloud"), ui.gestures
    import json
    metrics = [json.loads(m) for t, m in net.sent if t == "gesture/g1/metrics"]
    shake = [m for m in metrics if m["g"] == "shake"][0]
    assert shake["lat_ms"] == 15 and shake["bytes"] > 900, shake


@test
def cloud_window_survives_packing():
    rng = random.Random(9)
    w = synth("flick_down", rng)
    seq, back = cloud_classifier.unpack_window(app_mod.pack_window(42, w))
    assert seq == 42
    for k in range(6):
        tol = 0.0011 if k < 3 else 0.11
        assert max(abs(a - b) for a, b in zip(w[k], back[k])) < tol


@test
def cloud_mode_counts_timeouts():
    rng = random.Random(10)
    samples = gesture_stream("circle", rng)
    clock = ManualClock()
    app, net, ui = make_app(samples, clock)
    app.set_mode("cloud")
    for _ in samples:
        clock.now += 20
        app.step()  # nobody answers
    import json
    metrics = [json.loads(m) for t, m in net.sent if t == "gesture/g1/metrics"]
    assert metrics and all(m.get("error") == "timeout" for m in metrics), metrics
    assert ui.gestures == []


@test
def edge_mode_works_offline():
    rng = random.Random(11)
    samples = gesture_stream("shake", rng)
    app, net, ui = make_app(samples)
    net.online = False
    for _ in samples:
        app.step()
    assert ui.gestures and ui.gestures[0][0] == "shake"


@test
def remote_mode_switch():
    app, net, ui = make_app([])
    app.on_message(b"gesture/g1/mode", b"cloud")
    assert app.mode == "cloud"
    app.on_message(b"gesture/g1/mode", b"hybrid")
    assert app.mode == "hybrid"
    app.on_message(b"gesture/g1/mode", b"bogus")
    assert app.mode == "hybrid"


@test
def hybrid_mode_offloads_only_low_confidence():
    rng = random.Random(14)
    samples = gesture_stream("shake", rng)
    clock = ManualClock()
    net, ui = FakeNet(), FakeUI()
    # High confidence (0.95 >= 0.85): resolved locally on the edge, zero raw upload
    app_hi = app_mod.GestureApp(FakeIMU(samples), ui, net, lambda f: (4, 0.95),
                                STATE["model"].CLASSES, clock, device="g1", hybrid_conf=0.85)
    app_hi.set_mode("hybrid")
    for _ in samples:
        app_hi.step()
    assert not any(t == "gesture/g1/raw" for t, _ in net.sent)
    assert any(t == "gesture/g1/event" for t, _ in net.sent)

    # Ambiguous confidence (0.70 < 0.85): offloaded to cloud raw stream
    net2, ui2 = FakeNet(), FakeUI()
    app_lo = app_mod.GestureApp(FakeIMU(samples), ui2, net2, lambda f: (4, 0.70),
                                STATE["model"].CLASSES, clock, device="g1", hybrid_conf=0.85)
    app_lo.set_mode("hybrid")
    for _ in samples:
        app_lo.step()
    assert any(t == "gesture/g1/raw" for t, _ in net2.sent)


@test
def hybrid_mode_uses_own_answer_when_server_is_silent():
    rng = random.Random(15)
    samples = gesture_stream("shake", rng) + [STILL] * 50   # long enough to pass the 2 s timeout
    clock = ManualClock()
    net, ui = FakeNet(), FakeUI()
    app = app_mod.GestureApp(FakeIMU(samples), ui, net, lambda f: (4, 0.70),
                             STATE["model"].CLASSES, clock, device="g1", hybrid_conf=0.85)
    app.set_mode("hybrid")
    for _ in samples:
        clock.now += 20
        app.step()  # the raw window is sent, nobody answers
    assert ui.gestures and ui.gestures[0] == ("shake", "hybrid"), ui.gestures
    import json
    m = [json.loads(x) for t, x in net.sent if t == "gesture/g1/metrics"][0]
    assert m["g"] == "shake" and m["lat_ms"] > app_mod.CLOUD_TIMEOUT_MS and m["bytes"] > 900, m


@test
def bad_label_reply_is_ignored():
    app, net, ui = make_app([])
    app.set_mode("cloud")
    app.pending[5] = (0, 923, None)
    for bad in (b"garbage", b"5,shake", b"x,shake,0.9", b"5,shake,high"):
        app.on_message(b"gesture/g1/label", bad)
    assert 5 in app.pending and ui.gestures == []


@test
def event_bytes_are_counted_in_cloud_mode_too():
    app, net, ui = make_app([])
    app.set_mode("cloud")
    app.pending[6] = (0, 923, None)
    app.on_message(b"gesture/g1/label", b"6,shake,1.0")
    import json
    m = [json.loads(x) for t, x in net.sent if t == "gesture/g1/metrics"][0]
    event = [x for t, x in net.sent if t == "gesture/g1/event"][0]
    assert m["bytes"] == 923 + len(b"6,shake,1.0") + len("gesture/g1/label") + len(event) + len("gesture/g1/event"), m


@test
def web_console_keeps_simulated_rows_out_of_the_evaluation():
    import web_ui
    hub = web_ui.HubState(device="g1", broker_host="127.0.0.1", broker_port=1)  # no broker there
    for mode in ("edge", "hybrid", "cloud"):
        rec = hub.simulate_gesture("shake", force_mode=mode)
        assert rec["source"] == "sim" and rec["g"] == "shake", rec
    assert hub.simulate_gesture("shake", force_mode="cloud", wan_drop=True)["error"] == "timeout"
    assert all(v["windows"] == 0 for v in hub.summary_stats().values())
    assert hub.export_csv().strip().count(chr(10)) == 0          # header only
    hub.history.appendleft({"mode": "cloud", "lat_ms": 41.0, "bytes": 990, "source": "hw", "routed": "cloud-wan"})
    assert hub.summary_stats()["cloud"]["windows"] == 1
    assert "nokia" not in web_ui.HTML_PAGE.lower() and "cpouta" not in web_ui.HTML_PAGE.lower()


@test
def net_reconnects_without_blocking():
    class WLAN:
        connected = False
        connects = 0

        def __init__(self, _):
            pass

        def active(self, _):
            pass

        def connect(self, *a):
            WLAN.connects += 1

        def isconnected(self):
            return WLAN.connected

        def status(self):
            return 0

    network = types.ModuleType("network")
    network.WLAN, network.STA_IF, network.STAT_CONNECTING = WLAN, 0, 1

    class Client:
        fail_publish = False
        instances = []

        def __init__(self, cid, server, **kw):
            self.subs, self.pubs, self.sock = [], [], types.SimpleNamespace(close=lambda: None)
            Client.instances.append(self)

        def set_callback(self, cb):
            pass

        def set_last_will(self, *a, **k):
            pass

        def connect(self):
            pass

        def subscribe(self, t):
            self.subs.append(t)

        def check_msg(self):
            pass

        def ping(self):
            pass

        def publish(self, topic, msg, retain=False, qos=0):
            if Client.fail_publish:
                raise OSError(104)
            self.pubs.append((topic, msg))

    umqtt = types.ModuleType("umqtt")
    simple = types.ModuleType("umqtt.simple")
    simple.MQTTClient = Client
    umqtt.simple = simple
    sys.modules.update({"network": network, "umqtt": umqtt, "umqtt.simple": simple})
    net_mod = load_module(os.path.join(PICO, "net.py"), "net_under_test")

    cfg = types.SimpleNamespace(WIFI_SSID="s", WIFI_PASSWORD="p", DEVICE_NAME="g1", MQTT_BROKER="b",
                                MQTT_PORT=1883, MQTT_USER=None, MQTT_PASSWORD=None)
    n = net_mod.Net(cfg, lambda t, m: None)
    n.subscriptions = ("gesture/g1/label",)
    now = time.ticks_ms()
    n.poll(now)
    assert not n.online and not Client.instances          # no Wi-Fi yet: no MQTT attempt
    WLAN.connected = True
    n.poll(now + 1000)
    assert not n.online                                  # waits for the retry interval
    n.poll(now + net_mod.RETRY_MS + 1)
    assert n.online and Client.instances[-1].subs == ["gesture/g1/label"]
    assert ("gesture/g1/status", b"online") in Client.instances[-1].pubs
    Client.fail_publish = True
    assert n.publish("x", "y") is False and not n.online
    Client.fail_publish = False
    n.poll(now + 3 * net_mod.RETRY_MS)
    assert n.online and len(Client.instances) == 2


@test
def plots_summary_and_files():
    rows = []
    for i in range(20):
        rows.append({"time": str(i), "mode": "edge", "lat_ms": "12", "bytes": "60" if i % 4 == 0 else "0"})
        rows.append({"time": str(i), "mode": "cloud", "lat_ms": "-1" if i == 0 else str(40 + i), "bytes": "990"})
    s = plots.summarise(rows)
    assert s["cloud"]["lost"] == 1 and s["edge"]["bytes_per_window"] == 15
    lo, hi = s["cloud"]["lat_median_ci"]
    assert lo <= s["cloud"]["lat_median_ms"] <= hi
    dlo, dhi = plots.median_diff_ci(s["edge"]["lats"], s["cloud"]["lats"])
    assert dlo > 0 and dhi >= dlo                                   # cloud is slower in this made-up data
    out = os.path.join(TMP, "plots")
    os.makedirs(out)
    plots.plot_modes(s, out)
    assert os.path.exists(os.path.join(out, "bytes_by_mode.png"))
    assert os.path.exists(os.path.join(out, "latency_by_mode.png"))


@test
def cloud_classifier_over_real_broker():
    try:
        import amqtt  # noqa: F401
    except ImportError:
        print("  (amqtt not installed: skipped)")
        return
    import paho.mqtt.client as mqtt
    port = 18831
    code = ("import asyncio\nfrom amqtt.broker import Broker\n"
            "cfg={'listeners':{'default':{'type':'tcp','bind':'127.0.0.1:%d'}},"
            "'plugins':{'amqtt.plugins.authentication.AnonymousAuthPlugin':{'allow_anonymous':True}}}\n"
            "async def m():\n b=Broker(cfg); await b.start(); await asyncio.Event().wait()\n"
            "asyncio.run(m())\n" % port)
    broker = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2.5)
        server = cloud_classifier.make_client(STATE["model"], "cc-test")
        server.connect("127.0.0.1", port)
        server.loop_start()
        got = []
        pico = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fake-pico")
        pico.on_message = lambda c, u, m: got.append(m.payload.decode())
        pico.connect("127.0.0.1", port)
        pico.subscribe("gesture/g9/label")
        pico.loop_start()
        time.sleep(1)
        w = synth("shake", random.Random(12))
        pico.publish("gesture/g9/raw", app_mod.pack_window(7, w))
        deadline = time.time() + 5
        while not got and time.time() < deadline:
            time.sleep(0.05)
        assert got and got[0].startswith("7,shake,"), got
        pico.loop_stop()
        server.loop_stop()
    finally:
        broker.kill()


@test
def micropython_syntax_check():
    try:
        import mpy_cross  # noqa: F401
    except ImportError:
        print("  (mpy-cross not installed: skipped)")
        return
    out = os.path.join(TMP, "mpy")
    os.makedirs(out, exist_ok=True)
    files = ["hat.py", "features.py", "collect.py", "main.py", "net.py", "bench.py", "config.example.py"]
    for f in files:
        r = subprocess.run([sys.executable, "-m", "mpy_cross", "-o", os.path.join(out, f + ".mpy"),
                            os.path.join(PICO, f)], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)
    r = subprocess.run([sys.executable, "-m", "mpy_cross", "-o", os.path.join(out, "model.mpy"),
                        STATE["model_path"]], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


if __name__ == "__main__":
    failed = [r for r in RESULTS if not r[1]]
    print("\n%d/%d passed" % (len(RESULTS) - len(failed), len(RESULTS)))
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
