"""Cloud mode: classify raw windows sent by the Pico and send the label back.

It runs the same features.py and the same generated model.py as the Pico, so
edge and cloud give identical answers and the comparison is only about where
the work happens.

Usage:  python cloud_classifier.py --broker 192.168.1.100
Needs: pip install paho-mqtt
"""

import argparse
import importlib.util
import os
import struct
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
PICO = os.path.join(HERE, "..", "pico")
sys.path.insert(0, PICO)
import features  # noqa: E402

HEADER = "<HB"


def load_model(path):
    spec = importlib.util.spec_from_file_location("gesture_model", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def unpack_window(payload):
    seq, n = struct.unpack_from(HEADER, payload)
    vals = array("h")
    vals.frombytes(payload[struct.calcsize(HEADER):])
    if sys.byteorder != "little":
        vals.byteswap()
    if len(vals) != 6 * n:
        raise ValueError("expected %d values, got %d" % (6 * n, len(vals)))
    window = []
    for k in range(6):
        scale = 1000.0 if k < 3 else 10.0
        window.append([v / scale for v in vals[k * n:(k + 1) * n]])
    return seq, window


def classify(model, payload):
    seq, window = unpack_window(payload)
    idx, conf = model.predict(features.extract(window))
    return "%d,%s,%s" % (seq, model.CLASSES[idx], conf)


def make_client(model, client_id="cloud-classifier"):
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)

    def on_connect(c, userdata, flags, reason, props):
        c.subscribe("gesture/+/raw")
        print("Connected, waiting for raw windows")

    def on_message(c, userdata, msg):
        device = msg.topic.split("/")[1]
        try:
            reply = classify(model, msg.payload)
        except (ValueError, struct.error) as e:
            print("Bad window from", device, e)
            return
        c.publish("gesture/%s/label" % device, reply)

    client.on_connect = on_connect
    client.on_message = on_message
    return client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--model", default=os.path.join(PICO, "model.py"))
    args = ap.parse_args()
    client = make_client(load_model(args.model))
    client.connect(args.broker, args.port)
    client.loop_forever()


if __name__ == "__main__":
    main()
