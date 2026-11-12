# Exercises 3 and 4: publish sensor data over MQTT and obey LED commands.
# Needs umqtt.simple: on the Pico, run `import mip; mip.install("umqtt.simple")` once online.
from binascii import hexlify
from machine import Pin, unique_id
from umqtt.simple import MQTTClient
import json
import time
import config
import sensors
import wifi

led = Pin("LED", Pin.OUT)

TELEMETRY = ("iot/%s/telemetry" % config.DEVICE_NAME).encode()
STATUS = ("iot/%s/status" % config.DEVICE_NAME).encode()
COMMAND = ("iot/%s/command" % config.DEVICE_NAME).encode()
MAX_BACKOFF_S = 30


def on_message(topic, msg):
    # Check the topic too: some brokers can deliver messages we did not subscribe to.
    if topic != COMMAND:
        print("Ignored message on", topic)
    elif msg == b"on":
        led.value(1)
    elif msg == b"off":
        led.value(0)
    else:
        print("Unknown command:", msg)


def connect_mqtt():
    client = MQTTClient(
        b"pico-" + hexlify(unique_id()),
        config.MQTT_BROKER,
        port=config.MQTT_PORT,
        user=config.MQTT_USER,
        password=config.MQTT_PASSWORD,
        # A publish counts as activity, so the keep-alive must outlast the interval.
        keepalive=max(60, config.PUBLISH_INTERVAL_S * 2),
    )
    client.set_callback(on_message)
    # Retained, so anyone subscribing later still sees that the device is offline.
    client.set_last_will(STATUS, b"offline", retain=True, qos=1)
    client.connect()
    client.publish(STATUS, b"online", retain=True, qos=1)
    client.subscribe(COMMAND, qos=1)
    return client


def telemetry():
    data = {"device": config.DEVICE_NAME, "room": config.ROOM, "led": led.value()}
    data.update(sensors.read())
    return json.dumps(data).encode()


def close(client):
    try:
        client.sock.close()
    except Exception:
        pass


def main():
    interval_ms = config.PUBLISH_INTERVAL_S * 1000
    client = None
    backoff = 1
    last_publish = 0

    while True:
        try:
            if client is None:
                print("IP:", wifi.connect(config.WIFI_SSID, config.WIFI_PASSWORD))
                client = connect_mqtt()
                print("MQTT connected")
                backoff = 1
                last_publish = time.ticks_add(time.ticks_ms(), -interval_ms)

            client.check_msg()
            if time.ticks_diff(time.ticks_ms(), last_publish) >= interval_ms:
                client.publish(TELEMETRY, telemetry())
                last_publish = time.ticks_ms()
            time.sleep_ms(50)
        except Exception as e:
            # Wi-Fi or broker dropped: back off, then rebuild the connection.
            print("Connection lost:", e)
            if client is not None:
                close(client)
                client = None
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)


if __name__ == "__main__":
    main()
