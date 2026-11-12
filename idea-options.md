# Group 13: Project Ideas

**Deadlines:** register the group and idea by **Tue 10.11, 23:59**. The plan PDF is due **Fri 27.11**, and the final submission **Tue 22.12** (hard deadline).

## What the project must have

**Required (all three):**
- Sensor data collection on the Pico W
- Data processing on the Pico W
- Output control based on that processed data (LED, OLED, app...)

**Also needed (at least two):**
- Wireless (MQTT or HTTP)
- Cloud (InfluxDB, MongoDB, AWS, Google...)
- Mobile app
- Security (data encryption)
- Edge ML

**Evaluation:** two scenarios compared with measured data, such as message count or loss, latency, throughput or payload size, plus plots.

**Grading cautions:**
- "Just setting up an existing system is not accepted." The application logic must be your own.
- The lecturer said the focus is the data flow from sensor to device and onward. He will look at the whole chain: interface, data, storage, processing, security.
- Keep it small. The nominal workload is 55 h per student.

## Chosen (6 Nov): Edge-ML Gesture Controller

**Code:** written and tested on the PC, not yet on hardware. See [README.md](README.md) in this folder. MicroPython was chosen over C. The current team outline is [PROJECT_PLAN.md](PROJECT_PLAN.md); this file is the history of how the idea was chosen.

A Pico W worn on the wrist or held in the hand recognises gestures on the device itself, then uses them to control things over MQTT. The core question for the report: **is it better to classify on the device or in the cloud?**

**Hardware: the TKJ HAT from the Computer Systems course.** It already has everything this project needs, so the "no IMU" risk is gone, provided the group can use the HAT boards themselves and not only the SDK.

| HAT part | Use in this project | Pins / address |
|---|---|---|
| ICM-42670 IMU (accel + gyro) | Gesture sensing. Supports 25–1600 Hz; the SDK defaults to 100 Hz, ±4 g, ±250 dps. | I2C 0x69, SDA GPIO 12, SCL GPIO 13 |
| SSD1306 OLED | Shows the gesture, confidence and edge/cloud mode | I2C 0x3C |
| RGB LED | One colour per gesture | GPIO 18, 19, 20 (PWM) |
| Buzzer | Short beep when a gesture is recognised | GPIO 17 |
| Buttons SW1, SW2 | SW1: record a labelled sample during data collection. SW2: switch edge/cloud mode in the demo. | GPIO 2, GPIO 22 |
| Temperature/humidity, light, microphone | Not needed. Leave them out to keep the scope small. | — |

**Things to check:**
- **Wi-Fi: confirmed from a teammate's photos (11 Nov).** The group's HAT is already on a **Raspberry Pi Pico WH** (Pico W with headers), inside a red hand-held case. The micro-USB port and the debug header are reachable on the back. Still to do: run the I2C scan in MicroPython. The pins should not clash: the HAT uses GPIO 2, 6, 9 and 12–22, and the Pico W's wireless chip uses GPIO 23, 24, 25 and 29. The SDK's "onboard LED" is the HAT's red LED on GPIO 14, so it is safe too. Test that the IMU and OLED respond on the Pico W in the first week.
- **Language:** the TKJHAT SDK is C (Pico SDK + FreeRTOS). The IoT exercises and the code in `exercises/` are MicroPython. See the choice below.
- **Citations:** the SDK is MIT-licensed (University of Oulu; Raisul Islam and Iván Sánchez Milara), so cite it in the report. If you reuse code from your own Computer Systems project, ask the teachers whether that's fine, and say so in the report.
- **Fallback:** if the HAT can't be used, an MPU6050 board costs about €5. Option B (desk lighting) remains the last resort.

**Language choice:**

| | MicroPython (recommended) | C with the TKJHAT SDK |
|---|---|---|
| IMU and OLED | Write a small ICM-42670 driver (about 40 lines of I2C register access), porting the register setup from the SDK's C source with a citation. The OLED uses the standard `ssd1306` driver. | Ready-made: `ICM42670_start_with_default_values()`, `ICM42670_read_sensor_data(...)`, `write_text_xy(...)` |
| MQTT and Wi-Fi | `umqtt.simple`, already tested in `exercises/` | Pico SDK lwIP MQTT client with `pico_cyw43_arch`. More setup, and the course gives no help with it. |
| Fits the course | Same as the IoT exercises and the TAs' help | Different from the course. Builds on what you learned in Computer Systems. |
| Inference speed | Slower, but a decision tree is only a few comparisons | Much faster |
| Risk | Low | Medium (networking in C, FreeRTOS + Wi-Fi) |

The tree exporter can produce either MicroPython or C, so the ML part works with both.

### What it does

| Requirement | How |
|---|---|
| Sensing | The TKJ HAT's ICM-42670 IMU on I2C. 3-axis acceleration and 3-axis rotation sampled at 50 Hz (a supported rate). |
| Processing on the Pico | A 1.5 s window (75 samples × 6 axes) that starts 0.2 s before a movement begins (motion trigger). Features per window: 40 numbers (per axis: mean, standard deviation, min, max, mean crossings and peak order; plus the mean and standard deviation of the acceleration and rotation magnitudes). A decision tree classifies the window. A confidence threshold plus a short cooldown stops one gesture from firing twice. |
| Gestures (5 classes) | Idle, shake, circle, flick up, flick down. These are movements, not static tilts, so simple thresholds can't separate them and the ML is really doing the work. |
| Output control | The HAT's OLED shows the gesture and confidence, the RGB LED shows one colour per gesture, and the buzzer beeps. Each gesture also publishes an MQTT command that controls a second device, for example a lamp LED, a buzzer or a "slide forward/back" action. |
| Wireless | MQTT over Wi-Fi |
| Cloud | Node-RED → InfluxDB → Grafana: gestures per minute, confidence, and the evaluation measurements |
| Edge ML | The decision tree runs on the Pico |
| Security (stretch) | MQTT over TLS with a username and password |

That is three optional features (edge ML, wireless, cloud), with security as a fourth if time allows.

### How the model is built, and what counts as your own work

1. **Data collection script (Pico):** press SW1 on the HAT to record one gesture, and the Pico streams the labelled raw window to the PC over USB serial. Each member records about 50 repetitions of each gesture, so the data covers several people.
2. **Training (PC):** scikit-learn trains a decision tree on the features. Try tree depths of 3, 5 and 8, and use leave-one-person-out testing to check that it works for someone it hasn't seen.
3. **Export:** your own small script turns the trained tree into plain MicroPython `if/else` code. It needs no library on the Pico, uses little memory and is easy to explain at the demo.
4. **On the Pico:** the same feature code as in training, plus the generated tree.

scikit-learn and umqtt are cited libraries. The feature design, the exporter, the Pico application and the evaluation are the group's own work.

### Architecture

```
[Pico W + IMU] --(edge: label only)--MQTT--> [Broker] --> [Node-RED] --> [InfluxDB] --> [Grafana]
   features + tree                              |              |
   LED/OLED shows gesture                       |              +--(cloud mode: classify raw window,
                                                |                  publish label back)
                                                +--> [Second device: lamp LED / buzzer]
```

### Evaluation: two comparisons

1. **Edge vs cloud inference, using the same tree:**
   - *Edge:* the Pico classifies and publishes the label (about 20 bytes).
   - *Cloud:* the Pico publishes the raw window (75 × 6 × 2 bytes = 900 bytes in binary, several kB as JSON). Node-RED or a small Python service classifies it and publishes the label back.
   - Measure:
     - Bytes and messages sent per minute.
     - Time from the end of a gesture to the action, measured on the Pico with `ticks_ms`, so no clock synchronisation is needed.
     - Behaviour when Wi-Fi drops: edge mode still works locally, cloud mode doesn't.
2. **Model size vs accuracy:** tree depth 3, 5 and 8. Compare accuracy (confusion matrix), inference time on the Pico and the size of the generated code.

Plots:
- Bar chart of bytes per minute, edge vs cloud.
- Box plot of latency, edge vs cloud.
- Confusion matrix.
- Accuracy against inference time.

### Risks and fallbacks

| Risk | Plan |
|---|---|
| HAT not usable on the Pico W, or not available | Test it in the first week. Fallback: an MPU6050 board (about €5), then option B. |
| C networking takes too long (C route only) | Use MicroPython for the networked app and keep the C SDK as a reference |
| Low accuracy across people | More data from more people, fewer or more distinct gestures, or a per-user calibration step. Low accuracy that is measured and explained still scores. |
| Windows don't line up with gestures | Solved in the code: collection and the live app both use the same motion trigger (`features.Segmenter`) |
| Pico runs out of memory | A decision tree as code is tiny. Use `array` instead of lists for samples. |
| Live demo fails | Record a backup video. Edge mode works without Wi-Fi, which is itself a demo point. |

### Timeline against the deadlines

| By | Done |
|---|---|
| Tue 10.11 | Group and idea registered. Group agrees on MicroPython or C. HAT boards available? |
| Fri 27.11 | Plan PDF. HAT on a Pico W with the IMU and OLED working, and the data collection script running. |
| Fri 4.12 | Data recorded by every member. First tree trained and running on the Pico. |
| Fri 11.12 | MQTT control of the second device, cloud mode and the Grafana dashboard working |
| Thu 17.12 | Evaluation runs done and plots made |
| Tue 22.12 | Report and GitHub site submitted. Demo rehearsed. |

## Earlier recommendation: Smart Study Room Comfort Monitor

This takes the lecturer's own examples (smart classroom, Oulu SmartCampus) and builds them end to end. It covers **three** of the optional features, so one can fail without putting the pass at risk, and it reuses the exercise 1-5 code already in `exercises/`.

| Requirement | How |
|---|---|
| Sensing | Temperature and humidity (DHT22 or kit sensor), light (LDR on ADC), CO2 if the kit or Fab Lab has one |
| Processing on the Pico | Moving average to smooth noise, a simple comfort score, threshold decisions with hysteresis so the output doesn't flicker |
| Output control | LED or buzzer warning when the air is poor; optional OLED with the current status; a fan or relay as a stretch goal |
| Wireless | MQTT over Wi-Fi |
| Cloud | Node-RED → InfluxDB → Grafana dashboard and email alert |
| Security | MQTT over TLS (encrypted), with broker username and password |
| Stretch (only if time) | A second Pico node in another room, using the extra kits the course offers |

**Architecture:**

```
[Pico W + sensors] --MQTT/TLS--> [Broker] --> [Node-RED] --> [InfluxDB] --> [Grafana + email alert]
      ^  local processing                         |
      |  LED/OLED/buzzer                          +--> command topic back to the Pico (e.g. fan on)
```

**Evaluation: two comparisons, both measurable without synchronising clocks.** For latency, the Pico publishes a message with a sequence number, Node-RED echoes it back, and the Pico measures the round trip with `ticks_ms`.

1. **QoS 0 vs QoS 1**, each at a low and a high message rate (for example 1 and 10 messages per second): message loss (sequence-number gaps), round-trip latency, throughput.
2. **Plain MQTT vs MQTT over TLS:** connection time, round-trip latency and throughput, to show what encryption costs on a microcontroller.

Optionally, compare a small payload (one reading) with a batched large payload (many readings).

**Risks:**
- **Wi-Fi:** campus Wi-Fi (eduroam) generally won't work for the Pico, so ask the TAs which network to use, or use a phone hotspot.
- **TLS** on the Pico W needs some memory tuning, so try it early.
- **CO2 sensor** may not be in the kit, so the plan doesn't depend on one.

## Other options

Each one meets the three required parts and has at least two optional features. "Evaluation" is the two-scenario comparison the template asks for.

### A. Gesture or activity recognition with edge ML

- **Sensing:** accelerometer and gyroscope (an IMU such as the MPU6050) on I2C.
- **Processing on the Pico:** cut the signal into 1–2 s windows and compute features (mean, variance, peak). A small classifier, such as a decision tree or k-nearest neighbours, runs on the Pico. You train it on the PC from data you record yourselves.
- **Output:** an OLED or LED shows the recognised gesture. A gesture can also trigger an action, for example toggling a lamp through an MQTT command.
- **Optional features:** edge ML, wireless (MQTT), cloud (log the predictions).
- **Evaluation:** compare streaming raw IMU data to the server with classifying on the Pico and sending only the label. Measure bytes and messages sent, reaction time, and accuracy (confusion matrix).
- **Pros:** the clearest two-scenario story of any option. It matches the lecturer's interest in edge intelligence and will stand out.
- **Risks:**
  - It needs an IMU in the kit.
  - Recording enough labelled data takes a few evenings.
  - Accuracy may be modest, but that is fine as long as you measure and discuss it.

### B. Smart desk lighting with occupancy

- **Sensing:** PIR motion sensor, LDR light sensor.
- **Processing on the Pico:**
  - An occupancy state machine marks the desk "away" after N minutes without motion.
  - Daylight compensation sets lamp brightness to the target level minus ambient light.
- **Output:** a dimmable LED through PWM, or a relay for a real lamp.
- **Optional features:** wireless, cloud (dashboard of occupancy over time), security.
- **Evaluation:** compare periodic reporting (every 5 s) with event-driven reporting (only when something changes). Measure message count, data volume, and how quickly the dashboard reflects a change.
- **Pros:**
  - The sensors are cheap and common.
  - Dimming is visible in a demo.
  - Occupancy data gives you a privacy discussion that links to week 6.
- **Risks:** a PIR takes a few seconds to reset, so tune the timings early.

### C. Plant care with automatic watering

- **Sensing:** capacitive soil moisture on the ADC, plus temperature and humidity.
- **Processing on the Pico:**
  - Two-point calibration (dry and wet) converts the reading to a percentage, followed by smoothing.
  - The watering decision uses hysteresis and a daily safety limit.
- **Output:** a small pump through a relay or MOSFET, and a status LED.
- **Optional features:** wireless, cloud, and a phone dashboard for manual watering. Ask the TAs whether an existing MQTT phone app counts as the mobile-app feature.
- **Evaluation:** compare an always-on Pico with one that sleeps between readings. Measure current draw and estimated battery life. This needs a USB power meter or multimeter.
- **Risks:**
  - Water near electronics.
  - The pump needs its own power supply.
  - Soil changes slowly, so the demo needs a cup of water.

### D. Library noise-level monitor

- **Sensing:** an analog microphone module on the ADC.
- **Processing on the Pico:** sample a short burst each second and compute an RMS level, a relative dB value and a peak count.
- **Output:** traffic-light LEDs for quiet, moderate and loud.
- **Optional features:** wireless, cloud.
- **Evaluation:** compare sending raw sample blocks with sending only per-second levels. Measure bandwidth, message rate and loop timing on the Pico.
- **Pros:** only levels leave the device, never audio. That is a strong privacy-by-design point for the report.
- **Risks:**
  - MicroPython's ADC sampling is slow and uneven.
  - The dB value is relative, not calibrated.
  - The kit may not include a microphone.

### E. Smart weather station

- **Sensing:** temperature, humidity and pressure (for example a BME280), plus light.
- **Processing on the Pico:** a dew point calculation, and a simple "rain likely" warning from the pressure trend over three hours.
- **Output:** an OLED forecast and an LED.
- **Optional features:** wireless, cloud.
- **Evaluation:** compare MQTT with HTTP POST. Measure payload overhead, latency and connection cost.
- **Pros:** suggested on the slides, and simple to build.
- **Cons:**
  - Many groups will choose it, so it is harder to stand out.
  - The pressure trend needs hours of data.
  - Outdoor Wi-Fi is awkward.

### F. Multi-node campus air monitor (3 Picos)

- **What it is:** the recommended idea running on three nodes in different rooms.
- **Evaluation:** compare 1 node with 3 nodes, plus 10 simulated nodes. Measure message loss, latency, and load on the broker and Node-RED.
- **Cons:** more hardware and debugging. It works better as a stretch goal for the recommended idea than as a project of its own.

### How they compare

| Idea | Difficulty | Hardware risk | Evaluation story | Stands out |
|---|---|---|---|---|
| **Comfort Monitor (recommended)** | Medium | Low | Good | Medium |
| A. Edge-ML gestures | High | Medium (needs an IMU) | Very strong | High |
| B. Desk lighting | Low to medium | Low | Good | Medium |
| C. Plant care | Medium | Medium (pump, water) | Good if you can measure current | Medium |
| D. Noise monitor | Medium | Medium (microphone) | Good | Medium to high |
| E. Weather station | Low | Low | Fair | Low |
| F. Multi-node | Medium to high | Medium | Strong | Medium |

**Which to pick:** A was chosen on 6 Nov. B is the fallback if no IMU can be found.

Check the kit contents before 10.11. Ask at the Friday exercise or on Moodle or Discord, because A, C and D all depend on parts that may not be in it.

## Draft for the 10.11 registration

Fill in the names. The same content grows into the 27.11 plan PDF.

This is for the chosen Edge-ML Gesture Controller.

> **Title:** Edge-ML Gesture Controller
>
> **Group:** 13 — [member 1], [member 2], [member 3], [member 4]
>
> **Outline:** A Raspberry Pi Pico W fitted with the TKJ HAT (from the Computer Systems course) recognises hand gestures (shake, circle, flick up, flick down) on the device using the HAT's ICM-42670 accelerometer and gyroscope. It extracts features from a sliding window of sensor data and classifies them with a decision tree we train on data we record ourselves. The recognised gesture is shown on the HAT's OLED and RGB LED and sent as an MQTT command over Wi-Fi to control a second device. Node-RED stores events in InfluxDB and Grafana shows a dashboard. Optional functionalities: edge ML, wireless connectivity and cloud integration, with MQTT over TLS as a stretch goal.
>
> **Evaluation:** We compare classifying on the Pico (edge) with sending raw sensor windows to the server for classification (cloud), measuring data sent, gesture-to-action latency and behaviour when the network drops. We also compare decision trees of different depths for accuracy, inference time and code size.

**Earlier draft (Comfort Monitor), kept in case the kit has no IMU:**

> **Title:** Smart Study Room Comfort Monitor. A Pico W measures temperature, humidity and light, computes a comfort score on the device and warns with an LED or buzzer. Data goes over MQTT with TLS to Node-RED, InfluxDB and Grafana with email alerts. Evaluation: QoS 0 vs QoS 1, and plain MQTT vs TLS.
