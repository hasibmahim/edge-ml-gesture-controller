# Group 13 Backend Stack (Mosquitto + Node-RED + InfluxDB + Grafana)

This folder sets up Step 7 of `PROJECT_PLAN.md`: the MQTT broker, Node-RED ingestion flow, InfluxDB time-series storage, and Grafana dashboard for the **Edge-ML Gesture Controller**.

## 1. Start the containers

```bash
cd stack
cp .env.example .env
# Edit .env to set your local passwords and token if needed
docker compose up -d
```

Services started:
- **Mosquitto MQTT broker:** `localhost:1883`
- **Node-RED:** `http://localhost:1880`
- **InfluxDB v2:** `http://localhost:8086` (org `iot`, bucket `iot`)
- **Grafana:** `http://localhost:3000` (auto-provisions the `InfluxDB` datasource and the **Group 13 — Edge-ML Gesture Controller** dashboard)

## 2. Import the Node-RED flow (first run only)

1. Open `http://localhost:1880`.
2. Open the menu (top-right) → **Manage palette** → **Install** → search `node-red-contrib-influxdb` → **Install**.
3. Menu → **Import** → select `stack/nodered/flows.json` → **Import**.
4. Double-click the **Write gestures** node → click the pencil next to **InfluxDB v2** → paste your `INFLUX_TOKEN` from `.env` → **Update** → **Deploy**.

## 3. Data model in InfluxDB

Matches Section 2.3 of `plan/plan-group13.tex`:
- **`gestures` measurement** (from `gesture/<dev>/event`):
  - Tags: `device`, `mode`, `gesture`
  - Fields: `conf` (float), `count` (1)
- **`metrics` measurement** (from `gesture/<dev>/metrics`):
  - Tags: `device`, `mode`, `gesture`
  - Fields: `lat_ms` (float, omitted on timeout), `bytes` (int), `conf` (float), `seq` (int), `timeout` (0 or 1)
- **`device_status` measurement** (from `gesture/<dev>/status`):
  - Tags: `device`, `state`
  - Fields: `online` (1 or 0)
