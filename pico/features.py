# Feature extraction shared by the Pico (MicroPython) and the PC (CPython).
# The same file is used for training and on the device, so the model sees
# exactly the same numbers in both places. Pure Python, no numpy.

import math

AXES = ("ax", "ay", "az", "gx", "gy", "gz")
SAMPLE_RATE_HZ = 50
WINDOW = 75        # samples = 1.5 s
PRE_TRIGGER = 10   # samples kept from before the movement starts (0.2 s)
ONSET_DPS = 60.0   # rotation speed that counts as "a movement has started"
COOLDOWN = 15      # samples ignored after a window (0.3 s), so a gesture's tail doesn't retrigger
CLASSES = ("idle", "shake", "circle", "flick_up", "flick_down")

_STATS = ("mean", "std", "min", "max", "cross", "order")
FEATURE_NAMES = tuple("%s_%s" % (a, s) for a in AXES for s in _STATS) + (
    "acc_mag_mean", "acc_mag_std", "gyro_mag_mean", "gyro_mag_std")


def _mean_std(v):
    n = len(v)
    s = 0.0
    s2 = 0.0
    for x in v:
        s += x
        s2 += x * x
    mean = s / n
    var = s2 / n - mean * mean
    return mean, (math.sqrt(var) if var > 0 else 0.0)


def _axis(v):
    n = len(v)
    mean, std = _mean_std(v)
    lo = hi = v[0]
    ilo = ihi = 0
    for i in range(1, n):
        x = v[i]
        if x < lo:
            lo = x
            ilo = i
        elif x > hi:
            hi = x
            ihi = i
    # How often the signal crosses its own mean: high for shaking
    cross = 0
    above = v[0] > mean
    for i in range(1, n):
        a = v[i] > mean
        if a != above:
            cross += 1
            above = a
    # Did the peak come before or after the dip? Separates flick up from flick down
    order = (ihi - ilo) / n
    return (mean, std, lo, hi, cross, order)


def _magnitude(x, y, z):
    return [math.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]) for i in range(len(x))]


def extract(window):
    """window: 6 equal-length sequences (ax, ay, az, gx, gy, gz), oldest sample first.
    Returns a list of floats in FEATURE_NAMES order."""
    f = []
    for v in window:
        f.extend(_axis(v))
    f.extend(_mean_std(_magnitude(window[0], window[1], window[2])))
    f.extend(_mean_std(_magnitude(window[3], window[4], window[5])))
    return f


class Ring:
    """Fixed-size buffer of the last n samples for each of the 6 axes."""

    def __init__(self, n=WINDOW):
        self.n = n
        self._axes = [[0.0] * n for _ in range(6)]
        self._i = 0
        self.count = 0

    def push(self, sample):
        i = self._i
        for k in range(6):
            self._axes[k][i] = sample[k]
        self._i = (i + 1) % self.n
        self.count += 1

    def full(self):
        return self.count >= self.n

    def window(self):
        """Copy of the buffer in time order (oldest first)."""
        i = self._i
        return [a[i:] + a[:i] for a in self._axes]


class Segmenter:
    """Cuts gestures out of the sample stream.

    When the rotation speed passes ONSET_DPS, it waits until the window is
    complete and returns it: PRE_TRIGGER samples from before the movement plus
    the rest after it. Data collection and the live app both use this, so the
    model is trained on windows lined up exactly like the ones it will see."""

    def __init__(self, n=WINDOW, pre=PRE_TRIGGER, onset_dps=ONSET_DPS, cooldown=COOLDOWN):
        self.ring = Ring(n)
        self.pre = pre
        self.onset2 = onset_dps * onset_dps
        self.cooldown = cooldown
        self._remaining = 0
        self._blocked = 0

    def push(self, s):
        """Add one sample. Returns a finished window, or None."""
        self.ring.push(s)
        if self._remaining:
            self._remaining -= 1
            if self._remaining == 0:
                self._blocked = self.cooldown
                return self.ring.window()
            return None
        if self._blocked:
            self._blocked -= 1
            return None
        if self.ring.count > self.pre and s[3] * s[3] + s[4] * s[4] + s[5] * s[5] > self.onset2:
            self._remaining = self.ring.n - self.pre - 1
        return None

    def capturing(self):
        return self._remaining > 0
