# Data collection on the Pico W + TKJ HAT.
#
#   SW2: choose the gesture label (shown on the OLED)
#   SW1: arm the recorder (beep), then do the gesture within 3 s. Recording starts
#        by itself when the movement starts, exactly as in the live app.
#
# For "idle", record ordinary movements that are NOT gestures (reaching for a cup,
# turning the hand slowly, walking). The live app only classifies when something
# moves, so idle teaches the tree what to ignore.
#
# Each recording is printed over USB serial as one line:
#   G,<label>,<seq>,<ax x75>,<ay x75>,<az x75>,<gx x75>,<gy x75>,<gz x75>
# Run pc/capture.py on the PC to save the lines to a CSV file.
# Copy hat.py and features.py to the Pico, then run this file with Thonny or mpremote.

import time

import features
import hat

PERIOD_US = 1_000_000 // features.SAMPLE_RATE_HZ
COLOURS = {"idle": (0, 0, 40), "shake": (255, 0, 0), "circle": (0, 255, 0),
           "flick_up": (255, 160, 0), "flick_down": (160, 0, 255)}


def record(imu, on_start=None, timeout_ms=3000):
    """Sample at 50 Hz until the Segmenter returns a gesture window, or None on timeout."""
    seg = features.Segmenter()
    t = time.ticks_us()
    start = time.ticks_ms()
    started = False
    while True:
        w = seg.push(imu.read())
        if w is not None:
            return w
        if seg.capturing() and not started:
            started = True
            if on_start:
                on_start()
        if not started and time.ticks_diff(time.ticks_ms(), start) > timeout_ms:
            return None
        t = time.ticks_add(t, PERIOD_US)
        wait = time.ticks_diff(t, time.ticks_us())
        if wait > 0:
            time.sleep_us(wait)


def to_line(label, seq, w):
    values = ",".join("%.3f" % x for axis in w for x in axis)
    return "G,%s,%d,%s" % (label, seq, values)


def main():
    i2c = hat.make_i2c()
    imu = hat.ICM42670(i2c, odr_hz=features.SAMPLE_RATE_HZ)
    oled = hat.Display(i2c)
    rgb = hat.RGBLed()
    buzzer = hat.Buzzer()
    sw1 = hat.Button(hat.SW1_PIN)
    sw2 = hat.Button(hat.SW2_PIN)

    label_i = 0
    counts = [0] * len(features.CLASSES)
    seq = 0
    while True:
        label = features.CLASSES[label_i]
        oled.show("COLLECT", "label: " + label, "count: %d" % counts[label_i], "",
                  "SW2 next label", "SW1 record")
        while True:
            if sw2.pressed():
                label_i = (label_i + 1) % len(features.CLASSES)
                break
            if sw1.pressed():
                buzzer.beep(1500, 80)
                oled.show("GO: " + label, "", "move within 3 s")
                w = record(imu, on_start=lambda: rgb.set(*COLOURS[label]))
                rgb.set(0, 0, 0)
                if w is None:
                    oled.show("No movement", "", "try again")
                    time.sleep_ms(1000)
                    break
                buzzer.beep(2500, 40)
                print(to_line(label, seq, w))
                seq += 1
                counts[label_i] += 1
                break
            time.sleep_ms(20)


if __name__ == "__main__":
    main()
