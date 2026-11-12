# Second Pico W (the lamp)

A plain Pico W that switches its onboard LED when the gesture controller sends
`on` or `off` to `iot/pico2/command`. This is the group's exercise 3 device code.

1. Flash MicroPython for the Pico W.
2. With Wi-Fi connected, run once: `import mip; mip.install("umqtt.simple")`
3. Copy `config.example.py` to `config.py`, fill in the Wi-Fi and broker, and set `DEVICE_NAME = "pico2"`.
4. Copy `config.py`, `wifi.py`, `sensors.py` and `ex3_mqtt_device.py` to the Pico and run `ex3_mqtt_device.py`.

Test without the controller: publish `on` to `iot/pico2/command` from the laptop
(`mosquitto_pub -h <broker ip> -t iot/pico2/command -m on`). The LED should light.
