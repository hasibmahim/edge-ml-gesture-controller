# Edge-ML Gesture Controller: the app that runs on the Pico W + TKJ HAT.
#
# When a movement starts, it takes a 1.5 s window of IMU data (features.Segmenter)
# and classifies it, either here on the Pico ("edge") or by sending the raw
# window to the PC classifier ("cloud"). SW2 switches between those two.
# A third mode, "hybrid" (set over MQTT), classifies on the Pico and asks the
# server only when the tree is unsure. That only helps if the server runs a
# bigger model than the Pico (cloud_classifier.py --model results/model_d8.py).
#
# MQTT topics (DEV = config.DEVICE_NAME):
#   gesture/DEV/event    {"g": "shake", "conf": 0.93, "mode": "edge"}      recognised gestures
#   gesture/DEV/raw      binary window (cloud mode)                         Pico -> PC classifier
#   gesture/DEV/label    "seq,label,conf"                                   PC classifier -> Pico
#   gesture/DEV/metrics  {"seq", "mode", "g", "conf", "lat_ms", "bytes"}    one per classified window
#   gesture/DEV/mode     "edge" or "cloud"                                  switch mode remotely
#   gesture/DEV/status   "online" / "offline" (retained, Last Will)
# Gestures can also send commands to another device, see ACTIONS in config.py.
#
# Files on the Pico: main.py, hat.py, features.py, model.py, net.py, config.py,
# plus umqtt.simple and ssd1306 (install with mip).

import array
import gc
import json
import struct

import features

CLOUD_TIMEOUT_MS = 2000
HEADER = "<HB"  # seq (uint16), samples per axis (uint8); then int16 values, little endian


def pack_window(seq, window):
    """Raw window for cloud mode: acceleration in mg and rotation in 0.1 deg/s, as int16."""
    vals = array.array("h")
    for k, axis in enumerate(window):
        scale = 1000 if k < 3 else 10
        for x in axis:
            v = int(x * scale)
            vals.append(32767 if v > 32767 else -32768 if v < -32768 else v)
    return struct.pack(HEADER, seq & 0xFFFF, len(window[0])) + bytes(vals)


class GestureApp:
    """Hardware-independent logic, so it can be tested on a PC with fakes.

    imu.read() -> 6 floats; ui has status(mode, online) and gesture(name, conf, mode, lat_ms);
    net has publish(topic, msg) -> bool, poll(now) and online; predict(features) -> (index, conf);
    clock provides ticks_ms(), ticks_diff(a, b) (the MicroPython time module does)."""

    def __init__(self, imu, ui, net, predict, classes, clock, device="pico1",
                 conf_min=0.6, hybrid_conf=0.85, actions=None, metrics=True):
        self.imu, self.ui, self.net, self.clock = imu, ui, net, clock
        self.predict, self.classes = predict, classes
        self.conf_min = conf_min
        self.hybrid_conf = hybrid_conf
        self.actions = actions or {}
        self.metrics = metrics
        self.segmenter = features.Segmenter()
        self.mode = "edge"
        self.seq = 0
        self.pending = {}          # seq -> (time sent, bytes sent, local (name, conf) or None)
        base = "gesture/%s/" % device
        self.t_event, self.t_raw, self.t_label = base + "event", base + "raw", base + "label"
        self.t_metrics, self.t_mode = base + "metrics", base + "mode"

    # --- called once per IMU sample ---
    def step(self):
        now = self.clock.ticks_ms()
        window = self.segmenter.push(self.imu.read())
        if self.net:
            self.net.poll(now)
        self._expire(now)
        if window is None:
            return  # no complete movement yet: nothing to classify or send
        self.seq = (self.seq + 1) & 0xFFFF
        if self.mode == "edge":
            f = features.extract(window)
            idx, conf = self.predict(f)
            lat = self.clock.ticks_diff(self.clock.ticks_ms(), now)
            self._result(self.seq, self.classes[idx], conf, lat, 0)
        elif self.mode == "hybrid":
            f = features.extract(window)
            idx, conf = self.predict(f)
            if conf >= self.hybrid_conf or not (self.net and self.net.online):
                lat = self.clock.ticks_diff(self.clock.ticks_ms(), now)
                self._result(self.seq, self.classes[idx], conf, lat, 0)
            else:
                payload = pack_window(self.seq, window)
                if self._publish(self.t_raw, payload):
                    # keep the Pico's own answer: it is used if the server does not reply in time
                    self.pending[self.seq] = (now, len(payload) + len(self.t_raw), (self.classes[idx], conf))
                else:
                    lat = self.clock.ticks_diff(self.clock.ticks_ms(), now)
                    self._result(self.seq, self.classes[idx], conf, lat, 0)
        else:
            payload = pack_window(self.seq, window)
            if self._publish(self.t_raw, payload):
                self.pending[self.seq] = (now, len(payload) + len(self.t_raw), None)
            else:
                self._metric(self.seq, None, 0, -1, 0, "no network")

    # --- incoming MQTT messages ---
    def on_message(self, topic, msg):
        topic = topic.decode() if isinstance(topic, bytes) else topic
        if topic == self.t_label:
            try:
                seq, name, conf = msg.decode().split(",")
                seq, conf = int(seq), float(conf)
            except ValueError:
                return  # not a "seq,label,conf" reply: ignore it, do not crash the loop
            sent = self.pending.pop(seq, None)
            if sent is None:
                return  # too late, already counted as lost
            lat = self.clock.ticks_diff(self.clock.ticks_ms(), sent[0])
            self._result(seq, name, conf, lat, sent[1] + len(msg) + len(topic))
        elif topic == self.t_mode and msg in (b"edge", b"cloud", b"hybrid"):
            self.set_mode(msg.decode())

    def set_mode(self, mode):
        self.mode = mode
        self.pending = {}
        self.ui.status(mode, bool(self.net and self.net.online))

    # --- internals ---
    def _expire(self, now):
        for seq in [s for s, p in self.pending.items() if self.clock.ticks_diff(now, p[0]) > CLOUD_TIMEOUT_MS]:
            t, nbytes, local = self.pending.pop(seq)
            if local:  # hybrid: no reply from the server, so act on the Pico's own answer
                self._result(seq, local[0], local[1], self.clock.ticks_diff(now, t), nbytes)
            else:
                self._metric(seq, None, 0, -1, nbytes, "timeout")

    def _result(self, seq, name, conf, lat, nbytes):
        if name != "idle" and conf >= self.conf_min:
            self.ui.gesture(name, conf, self.mode, lat)
            event = json.dumps({"g": name, "conf": conf, "mode": self.mode})
            if self._publish(self.t_event, event):
                nbytes += len(event) + len(self.t_event)  # counted in every mode
            action = self.actions.get(name)
            if action:
                self._publish(action[0], action[1])
        self._metric(seq, name, conf, lat, nbytes)
        gc.collect()

    def _metric(self, seq, name, conf, lat, nbytes, error=None):
        if not self.metrics:
            return
        m = {"seq": seq, "mode": self.mode, "g": name, "conf": conf, "lat_ms": lat, "bytes": nbytes}
        if error:
            m["error"] = error
        self._publish(self.t_metrics, json.dumps(m))

    def _publish(self, topic, msg):
        return bool(self.net) and self.net.publish(topic, msg)


class HatUI:
    COLOURS = {"shake": (255, 0, 0), "circle": (0, 255, 0),
               "flick_up": (255, 160, 0), "flick_down": (160, 0, 255)}

    def __init__(self, display, rgb, buzzer):
        self.display, self.rgb, self.buzzer = display, rgb, buzzer

    def status(self, mode, online):
        self.display.show("GESTURES", "mode: " + mode, "mqtt: " + ("online" if online else "OFFLINE"),
                          "", "SW2: switch mode")

    def gesture(self, name, conf, mode, lat_ms):
        self.rgb.set(*self.COLOURS.get(name, (255, 255, 255)))
        self.buzzer.beep()
        self.display.show(name.upper(), "conf %d%%" % int(conf * 100), "%s %d ms" % (mode, lat_ms))


def run():
    import time

    import config
    import hat
    import model
    from net import Net

    i2c = hat.make_i2c()
    imu = hat.ICM42670(i2c, odr_hz=features.SAMPLE_RATE_HZ)
    ui = HatUI(hat.Display(i2c), hat.RGBLed(), hat.Buzzer())
    sw2 = hat.Button(hat.SW2_PIN)

    app = None
    net = Net(config, lambda t, m: app.on_message(t, m))
    app = GestureApp(imu, ui, net, model.predict, model.CLASSES, time,
                     device=config.DEVICE_NAME, conf_min=config.CONF_MIN,
                     hybrid_conf=getattr(config, "HYBRID_CONF", 0.85), actions=config.ACTIONS)
    net.subscriptions = (app.t_label, app.t_mode)
    ui.status(app.mode, False)

    period = 1000 // features.SAMPLE_RATE_HZ
    next_t = time.ticks_ms()
    was_online = False
    while True:
        app.step()
        if sw2.pressed():
            app.set_mode("cloud" if app.mode == "edge" else "edge")
        if net.online != was_online:
            was_online = net.online
            ui.status(app.mode, was_online)
        next_t = time.ticks_add(next_t, period)
        if time.ticks_diff(next_t, time.ticks_ms()) <= 0:
            next_t = time.ticks_ms()  # fell behind (classification took long): resync
            continue
        # While a server reply is pending, check for it every 2 ms and not only once
        # per sample (20 ms), so the measured cloud latency is not rounded up.
        while True:
            wait = time.ticks_diff(next_t, time.ticks_ms())
            if wait <= 0:
                break
            if app.pending and wait > 2:
                net.poll(time.ticks_ms())
                time.sleep_ms(2)
            else:
                time.sleep_ms(wait)


if __name__ == "__main__":
    run()
