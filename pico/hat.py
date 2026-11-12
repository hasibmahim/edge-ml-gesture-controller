# TKJ HAT hardware for MicroPython on a Pico W.
#
# Pin numbers, I2C addresses and the ICM-42670 register setup follow the
# TKJHAT SDK (C), University of Oulu, MIT License, authors Raisul Islam and
# Ivan Sanchez Milara:
# https://github.com/unioulu-ubicomp-programming-courses/JTKJ-PicoRTOS-Project
# This file is our own MicroPython port, not a copy of the C code.

import struct
import time

from machine import I2C, PWM, Pin, Timer

SDA_PIN = 12
SCL_PIN = 13
IMU_ADDR = 0x69
OLED_ADDR = 0x3C
RED_LED_PIN = 14
RGB_PINS = (18, 19, 20)  # common anode: low = on
BUZZER_PIN = 17
SW1_PIN = 2   # active high, pull-down on the board
SW2_PIN = 22


def make_i2c():
    # GPIO 12/13 are I2C0 SDA/SCL on the RP2040
    return I2C(0, sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=400_000)


class ICM42670:
    """Accelerometer + gyroscope. read() returns (ax, ay, az) in g and (gx, gy, gz) in deg/s."""

    REG_MCLK_RDY = 0x00
    REG_SIGNAL_PATH_RESET = 0x02
    REG_DATA = 0x09          # TEMP_DATA1, then ACCEL X/Y/Z, then GYRO X/Y/Z, big endian
    REG_PWR_MGMT0 = 0x1F
    REG_GYRO_CONFIG0 = 0x20
    REG_ACCEL_CONFIG0 = 0x21
    REG_WHO_AM_I = 0x75
    WHO_AM_I = 0x67

    ODR_BITS = {25: 0x0B, 50: 0x0A, 100: 0x09, 200: 0x08, 400: 0x07, 800: 0x06, 1600: 0x05}
    ACCEL_FSR = {2: (0x03, 16384.0), 4: (0x02, 8192.0), 8: (0x01, 4096.0), 16: (0x00, 2048.0)}
    GYRO_FSR = {250: (0x03, 131.0), 500: (0x02, 65.5), 1000: (0x01, 32.8), 2000: (0x00, 16.4)}

    def __init__(self, i2c, addr=IMU_ADDR, odr_hz=50, accel_g=8, gyro_dps=1000):
        # Wider ranges than the SDK default (4 g, 250 dps) because shakes and flicks saturate those.
        self.i2c = i2c
        self.addr = addr
        self._buf = bytearray(14)
        self._reset()
        who = self._read_reg(self.REG_WHO_AM_I)
        if who != self.WHO_AM_I:
            raise OSError("ICM-42670 not found, WHO_AM_I=0x%02x" % who)

        a_bits, self.a_res = self.ACCEL_FSR[accel_g]
        g_bits, self.g_res = self.GYRO_FSR[gyro_dps]
        odr = self.ODR_BITS[odr_hz]
        # Low-noise mode for both sensors, then range and output data rate
        self._write_reg(self.REG_PWR_MGMT0, 0x0F)
        time.sleep_us(400)
        self._write_reg(self.REG_ACCEL_CONFIG0, (a_bits << 5) | odr)
        self._write_reg(self.REG_GYRO_CONFIG0, (g_bits << 5) | odr)
        time.sleep_ms(50)  # the gyro needs about 45 ms to start

    def _write_reg(self, reg, value):
        self.i2c.writeto_mem(self.addr, reg, bytes([value]))

    def _read_reg(self, reg):
        return self.i2c.readfrom_mem(self.addr, reg, 1)[0]

    def _reset(self):
        self._write_reg(self.REG_SIGNAL_PATH_RESET, 0x10)
        time.sleep_ms(1)  # the datasheet asks for 1 ms after a soft reset
        for _ in range(100):  # wait for the internal clock (MCLK_RDY, bit 3)
            if self._read_reg(self.REG_MCLK_RDY) & 0x08:
                break
            time.sleep_us(50)
        time.sleep_us(200)

    def read(self):
        self.i2c.readfrom_mem_into(self.addr, self.REG_DATA, self._buf)
        _, ax, ay, az, gx, gy, gz = struct.unpack(">7h", self._buf)
        a, g = self.a_res, self.g_res
        return (ax / a, ay / a, az / a, gx / g, gy / g, gz / g)


class RGBLed:
    def __init__(self, pins=RGB_PINS):
        self._pwm = []
        for p in pins:
            pwm = PWM(Pin(p))
            pwm.freq(1000)
            self._pwm.append(pwm)
        self.set(0, 0, 0)

    def set(self, r, g, b):
        # 0-255 per channel; inverted because the LED is common anode
        for pwm, v in zip(self._pwm, (r, g, b)):
            pwm.duty_u16(65535 - v * 257)


class Buzzer:
    def __init__(self, pin=BUZZER_PIN):
        self._pwm = PWM(Pin(pin))
        self._pwm.duty_u16(0)
        self._timer = Timer()

    def beep(self, freq=2000, ms=60):
        # Non-blocking: a one-shot timer turns it off, so sampling isn't delayed
        self._pwm.freq(freq)
        self._pwm.duty_u16(32768)
        self._timer.init(mode=Timer.ONE_SHOT, period=ms, callback=lambda t: self._pwm.duty_u16(0))


class Button:
    """Edge-detecting button, polled from the main loop."""

    def __init__(self, pin):
        self._pin = Pin(pin, Pin.IN)  # the HAT has a hardware pull-down
        self._last = 0

    def pressed(self):
        now = self._pin.value()
        edge = now == 1 and self._last == 0
        self._last = now
        return edge


class Display:
    """SSD1306 128x64. Works without the display driver (prints instead)."""

    def __init__(self, i2c):
        try:
            import ssd1306  # install once on the Pico: import mip; mip.install("ssd1306")
            self._oled = ssd1306.SSD1306_I2C(128, 64, i2c, addr=OLED_ADDR)
        except Exception as e:
            print("No OLED:", e)
            self._oled = None

    def show(self, *lines):
        if self._oled is None:
            print(" | ".join(lines))
            return
        self._oled.fill(0)
        for i, line in enumerate(lines[:6]):
            self._oled.text(line[:16], 0, i * 10)
        self._oled.show()
