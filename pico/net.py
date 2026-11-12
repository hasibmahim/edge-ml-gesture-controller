# Wi-Fi + MQTT that never blocks the gesture loop for long.
# If the network drops, poll() retries in the background every few seconds
# and the app keeps recognising gestures locally (edge mode).

import time

import network
from umqtt.simple import MQTTClient

RETRY_MS = 5000
PING_MS = 30000


class Net:
    def __init__(self, config, on_message):
        self.cfg = config
        self.on_message = on_message
        self.subscriptions = ()
        self.online = False
        self.client = None
        self.wlan = network.WLAN(network.STA_IF)
        self.wlan.active(True)
        self.wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)  # returns at once
        self._next_try = time.ticks_ms()
        self._last_ping = time.ticks_ms()
        self.status_topic = "gesture/%s/status" % config.DEVICE_NAME

    def poll(self, now):
        if self.online:
            try:
                self.client.check_msg()  # non-blocking; calls on_message
                if time.ticks_diff(now, self._last_ping) > PING_MS:
                    self.client.ping()  # umqtt.simple doesn't send keep-alive pings itself
                    self._last_ping = now
            except OSError:
                self._lost()
            return
        if time.ticks_diff(now, self._next_try) < 0:
            return
        self._next_try = time.ticks_add(now, RETRY_MS)
        if not self.wlan.isconnected():
            if self.wlan.status() != network.STAT_CONNECTING:
                self.wlan.connect(self.cfg.WIFI_SSID, self.cfg.WIFI_PASSWORD)
            return
        try:
            self._connect_mqtt()
            print("MQTT connected")
        except OSError as e:
            print("MQTT connect failed:", e)
            self._lost()

    def _connect_mqtt(self):
        c = MQTTClient(self.cfg.DEVICE_NAME, self.cfg.MQTT_BROKER, port=self.cfg.MQTT_PORT,
                       user=self.cfg.MQTT_USER, password=self.cfg.MQTT_PASSWORD, keepalive=60, ssl=self._tls())
        c.set_callback(self.on_message)
        c.set_last_will(self.status_topic, b"offline", retain=True, qos=1)
        c.connect()
        c.publish(self.status_topic, b"online", retain=True, qos=1)
        for t in self.subscriptions:
            c.subscribe(t)
        self.client = c
        self.online = True
        self._last_ping = time.ticks_ms()

    def _tls(self):
        # umqtt.simple (current micropython-lib) takes an SSLContext, not True/False.
        # Not tested on hardware yet.
        if not getattr(self.cfg, "MQTT_SSL", False):
            return None
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ca = getattr(self.cfg, "MQTT_CA_FILE", None)
        if ca:
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_verify_locations(cafile=ca)
        else:
            # Encrypted, but the broker's identity is not checked
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _lost(self):
        self.online = False
        try:
            if self.client:
                self.client.sock.close()
        except Exception:
            pass
        self.client = None

    def publish(self, topic, msg):
        if not self.online:
            return False
        try:
            self.client.publish(topic, msg)
            return True
        except OSError:
            self._lost()
            return False
