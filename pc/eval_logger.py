"""Record the Pico's metrics messages to a CSV file during an evaluation run.

Usage:  python eval_logger.py --broker 192.168.1.100 --out ../results/run1.csv
Switch modes from here with --mode edge|cloud|hybrid (sent to gesture/<device>/mode).
Needs: pip install paho-mqtt
"""

import argparse
import csv
import json
import os
import time

import paho.mqtt.client as mqtt

FIELDS = ["time", "device", "seq", "mode", "g", "conf", "lat_ms", "bytes", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="gesture1")
    ap.add_argument("--mode", choices=["edge", "cloud", "hybrid"], help="switch the Pico to this mode first")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if new:
        writer.writeheader()

    def on_connect(c, userdata, flags, reason, props):
        c.subscribe("gesture/+/metrics")
        if args.mode:
            c.publish("gesture/%s/mode" % args.device, args.mode)
        print("Logging to", args.out, "- Ctrl+C to stop")

    def on_message(c, userdata, msg):
        row = json.loads(msg.payload)
        row["time"] = round(time.time(), 3)
        row["device"] = msg.topic.split("/")[1]
        writer.writerow(row)
        f.flush()
        print(row)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="eval-logger")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        f.close()


if __name__ == "__main__":
    main()
