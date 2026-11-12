# On-device benchmark: feature extraction and inference time for each tree depth.
# Copy results/model_d3.py, model_d5.py and model_d8.py from train.py to the Pico,
# plus features.py, then run this file. Paste the printed CSV into results/bench.csv.
# mem_bytes is the RAM the model module takes after import.

import gc
import math
import time

import features

RUNS = 50


def test_window():
    # A fixed, shake-like window: the timing doesn't depend much on the values
    n = features.WINDOW
    return [[math.sin(i * 0.8 + k) * (2 if k < 3 else 200) for i in range(n)] for k in range(6)]


def bench(fn):
    t0 = time.ticks_us()
    for _ in range(RUNS):
        fn()
    return time.ticks_diff(time.ticks_us(), t0) / RUNS / 1000


def main():
    w = test_window()
    f = features.extract(w)
    print("feature extraction: %.2f ms" % bench(lambda: features.extract(w)))
    print("depth,infer_ms,mem_bytes")
    for d in (3, 5, 8):
        gc.collect()
        before = gc.mem_free()
        try:
            m = __import__("model_d%d" % d)
        except ImportError:
            continue
        gc.collect()
        used = before - gc.mem_free()
        print("%d,%.3f,%d" % (d, bench(lambda: m.predict(f)), used))


main()
