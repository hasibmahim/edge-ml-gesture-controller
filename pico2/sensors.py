from machine import ADC, Pin
import config

SAMPLES = 8
_temp_channel = ADC(4)
_dht = None

if config.DHT22_PIN is not None:
    import dht

    _dht = dht.DHT22(Pin(config.DHT22_PIN))


def onboard_temp_c():
    # Measures the chip, so it reads above room temperature. Averaging reduces ADC noise.
    total = 0
    for _ in range(SAMPLES):
        total += _temp_channel.read_u16()
    volts = (total / SAMPLES) * 3.3 / 65535
    return 27 - (volts - 0.706) / 0.001721


def read():
    if _dht is not None:
        _dht.measure()
        return {"temperature_c": _dht.temperature(), "humidity_pct": _dht.humidity()}
    return {"temperature_c": round(onboard_temp_c(), 2)}
