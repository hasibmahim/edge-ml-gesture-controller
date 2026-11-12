"""Save gesture recordings from the Pico (running collect.py) to a CSV file.

Usage:  python capture.py --port COM5 --person m1 --out ../data/m1.csv
Close Thonny first: only one program can use the serial port.
Needs: pip install pyserial
"""

import argparse
import collections
import os


def to_csv_row(line, person):
    """'G,<label>,<seq>,<values>' from the Pico -> '<person>,<label>,<values>' for train.py"""
    _, label, _seq, rest = line.split(",", 3)
    return label, "%s,%s,%s\n" % (person, label, rest)


def main():
    import serial
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--person", required=True, help="short id of the person recording, e.g. m1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    counts = collections.Counter()
    with serial.Serial(args.port, 115200, timeout=1) as port, open(args.out, "a", encoding="ascii") as out:
        print("Recording from", args.port, "- press SW1 on the Pico. Ctrl+C to stop.")
        try:
            while True:
                line = port.readline().decode("ascii", "replace").strip()
                if not line.startswith("G,"):
                    if line:
                        print("pico:", line)
                    continue
                label, row = to_csv_row(line, args.person)
                out.write(row)
                out.flush()
                counts[label] += 1
                print("saved", label, dict(counts))
        except KeyboardInterrupt:
            print("Stopped.", dict(counts))


if __name__ == "__main__":
    main()
