# Copy to config.py on the Pico and fill in your values. Do not commit config.py.
WIFI_SSID = "your-network"
WIFI_PASSWORD = "your-password"

HTTP_PORT = 80

# IP of the machine running the broker, not "localhost"
MQTT_BROKER = "192.168.1.100"
MQTT_PORT = 1883
MQTT_USER = None
MQTT_PASSWORD = None

DEVICE_NAME = "pico2"
ROOM = "lab"
PUBLISH_INTERVAL_S = 10

# GPIO number if a DHT22 is wired up, otherwise None to use the onboard sensor
DHT22_PIN = None
