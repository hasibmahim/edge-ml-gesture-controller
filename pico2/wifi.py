import network
import time


def connect(ssid, password, timeout_s=20):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return wlan.ifconfig()[0]

    wlan.connect(ssid, password)
    start = time.ticks_ms()
    while not wlan.isconnected():
        if time.ticks_diff(time.ticks_ms(), start) > timeout_s * 1000:
            wlan.active(False)
            raise OSError("Wi-Fi timeout, status=%d" % wlan.status())
        time.sleep_ms(200)
    return wlan.ifconfig()[0]
