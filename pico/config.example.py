# Copy to config.py on the Pico and fill in your values. Do not commit config.py.
WIFI_SSID = "your-network"
WIFI_PASSWORD = "your-password"

# IP of the machine running the broker, not "localhost"
MQTT_BROKER = "192.168.1.100"
MQTT_PORT = 1883
MQTT_USER = None
MQTT_PASSWORD = None
# Stretch goal, not tested on hardware yet: TLS (usually port 8883).
# With MQTT_CA_FILE = None the link is encrypted but the broker is not verified.
MQTT_SSL = False
MQTT_CA_FILE = None

DEVICE_NAME = "gesture1"

# Only act on a gesture if the tree is at least this sure (0-1)
CONF_MIN = 0.6
# Hybrid mode only: below this confidence the Pico asks the server
HYBRID_CONF = 0.85
# The movement threshold that starts a window is features.ONSET_DPS. It is shared
# with collect.py on purpose: change it there and re-record, never in one place only.

# Gesture -> (topic, payload) sent to another device. These drive the LED of a
# second Pico running the code in pico2/.
ACTIONS = {
    "flick_up": ("iot/pico2/command", "on"),
    "flick_down": ("iot/pico2/command", "off"),
}
