# FL: Notification-Driven oneM2M Federated Learning for Bearing Fault Detection

## 1. Overview

This repository implements a notification-driven federated learning workflow for bearing fault detection using oneM2M resources and the FEMTO-ST PRONOSTIA bearing dataset.

The implementation follows the direction of oneM2M TR-0084, which describes how standard oneM2M resources can be used to support federated learning.

The following oneM2M resources are used:

- `<AE>`
- `<container>`
- `<contentInstance>`
- `<subscription>`
- `<ACP>`

The current prototype provides:

- a single TinyIoT CSE
- one IN-AE coordinator
- three MN-AE local training clients
- notification-driven FL round commands
- node-specific local update containers
- subscription-based update collection
- Conv1D Autoencoder local training
- weighted FedAvg aggregation
- differential privacy configuration
- loss-based anomalous update filtering
- ACP-based access control
- real-time dashboard visualization
- stale process, model, and cache reset
- final model evaluation
- Isaac Sim virtual bearing testbed as the sensor data source
- round-synchronized sensor metadata publication from Isaac Sim
- file-buffer based signal handover between Isaac Sim and MN-AE
- dashboard fault detection fed back into Isaac Sim visualization
- one-command demo runner

The current implementation runs all AEs on one PC.

```text
IN-AE
→ FL coordinator and global model aggregator

MN-AE-1
→ Condition 1 local training client

MN-AE-2
→ Condition 2 local training client

MN-AE-3
→ Condition 3 local training client
```

Because all AEs currently run on the same PC, actual PyTorch model files are stored on the local filesystem.

oneM2M resources exchange model metadata and file paths instead of transferring the complete model binary.

---

## 2. System Architecture

```text
TinyIoT CSE
http://127.0.0.1:3000

TinyIoT
├── IN-AE
│   ├── cnt-fl-control
│   │   ├── FL state CIN
│   │   ├── FL round command CIN
│   │   ├── sub_mn1
│   │   ├── sub_mn2
│   │   └── sub_mn3
│   │
│   ├── cnt-global-model
│   │   └── Global model metadata CIN
│   │
│   └── cnt-local-updates
│       ├── cnt-mn1
│       │   └── sub_in_dropbox_mn1
│       ├── cnt-mn2
│       │   └── sub_in_dropbox_mn2
│       └── cnt-mn3
│           └── sub_in_dropbox_mn3
│
├── MN-AE-1
│   ├── cnt-sensor-data
│   └── cnt-local-model
│
├── MN-AE-2
│   ├── cnt-sensor-data
│   └── cnt-local-model
│
└── MN-AE-3
    ├── cnt-sensor-data
    └── cnt-local-model
```

Isaac Sim runs on the Windows side and acts as both the virtual equipment and the sensor gateway. Preprocessing is not performed inside Isaac Sim.

```text
FEMTO raw CSV
   │  prepare_data_femto_replay.py   (WSL)
   ▼
mn1_stream.npz / mn2_stream.npz / mn3_stream.npz
   │  copied to C:\Projects\bearing_testbed\data\femto_replay
   ▼
Isaac Sim gateway v7            (Windows)
   ├─ replays the stream and visualizes three bearing units
   ├─ writes runtime buffers  → mn1.pkl / mn2.pkl / mn3.pkl
   └─ publishes round metadata → TinyIoT  MN-AE-X/cnt-sensor-data
   ▼
MN-AE waits until metadata round == current round   (WSL)
   └─ reads the buffer slice referenced by data_path
```

Only metadata is published to TinyIoT. The 2,560-sample signal arrays stay in the file buffer and are referenced by path.

### Entity Roles

| Entity | Role |
| --- | --- |
| `IN-AE` | Starts FL rounds, publishes commands, collects local updates, validates updates, and performs global aggregation |
| `MN-AE-1` | Trains a local model using Condition 1 bearing data |
| `MN-AE-2` | Trains a local model using Condition 2 bearing data |
| `MN-AE-3` | Trains a local model using Condition 3 bearing data |
| TinyIoT CSE | Stores oneM2M resources and sends subscription notifications |
| Dashboard | Reads the current FL state and displays training and anomaly graphs |
| oneM2M Design Tool | Displays the TinyIoT resource tree |
| Isaac Sim gateway | Replays FEMTO streams, visualizes the bearing units, writes runtime buffers, and publishes round metadata |
| FL demo runner | Resets demo state and starts Dashboard, MN-AEs, and IN-AE in order |

---

## 3. oneM2M Resource Mapping

| FL Function | oneM2M Resource | Stored Content |
| --- | --- | --- |
| FL round control | `IN-AE/cnt-fl-control` | FL state, current round, maximum rounds, global model URI |
| Global model registry | `IN-AE/cnt-global-model` | global model round, path, and readiness |
| MN1 update drop-box | `IN-AE/cnt-local-updates/cnt-mn1` | MN1 local model metadata and metrics |
| MN2 update drop-box | `IN-AE/cnt-local-updates/cnt-mn2` | MN2 local model metadata and metrics |
| MN3 update drop-box | `IN-AE/cnt-local-updates/cnt-mn3` | MN3 local model metadata and metrics |
| Sensor data metadata | `MN-AE-X/cnt-sensor-data` | dataset path, round, node, and source |
| Local model cache | `MN-AE-X/cnt-local-model` | cached global model metadata |
| Command notification | Subscription under `cnt-fl-control` | notification target for each MN-AE |
| Update notification | Subscription under each `cnt-mnX` | notification target for IN-AE |
| Access control | ACP and `acpi` | IN-AE and MN-AE permissions |

### Model Metadata

The complete PyTorch model is not stored inside a oneM2M content instance.

The actual model file is stored locally.

```text
/tmp/fl_models/global/global_round3.pt
```

The oneM2M content instance stores metadata such as:

```json
{
  "type": "global-model",
  "global_round": 3,
  "model_path": "/tmp/fl_models/global/global_round3.pt",
  "model_ready": true
}
```

A local update contains information such as:

```json
{
  "type": "fl-update",
  "node": "mn1",
  "round": 3,
  "model_path": "/tmp/fl_models/local/mn1/round3.pt",
  "train_loss": 0.001,
  "val_loss": 0.002,
  "val_auroc": 0.95,
  "num_samples": 1000
}
```

Sensor metadata published by the Isaac Sim gateway has the following form:

```json
{
  "type": "isaac-sim-sync-buffer",
  "jobState": "FL_TRAINING",
  "node": "mn3",
  "round": 8,
  "currentRound": 8,
  "data_path": "C:/Projects/bearing_testbed/data/fl_buffer/mn3.pkl",
  "stream_path": "C:/Projects/bearing_testbed/data/femto_replay/mn3_stream.npz",
  "start_idx": 560,
  "end_idx": 640,
  "window_count": 80
}
```

For TinyIoT compatibility, dictionary content is serialized into a JSON string before being stored in `contentInstance.con`.

---

## 4. Notification-Driven FL Workflow

The main FL workflow uses oneM2M subscription notifications.

Continuous polling is not used to control FL execution.

```text
IN-AE
→ creates a round command CIN

TinyIoT
→ detects the new CIN
→ sends NOTIFY to MN-AE-1
→ sends NOTIFY to MN-AE-2
→ sends NOTIFY to MN-AE-3

MN-AEs
→ receive the command
→ retrieve the global model metadata
→ perform local training
→ upload local update CINs

TinyIoT
→ sends update NOTIFY messages to IN-AE

IN-AE
→ receives all local updates
→ validates them
→ performs FedAvg
→ publishes the next global model
```

If a notification is missing, the receiver performs one latest-CIN RETRIEVE request as a recovery path.

The dashboard performs periodic GET requests only for visualization.

Dashboard polling does not control FL training.

---

## 5. Subscription Structure

### 5.1 FL Command Subscriptions

Each MN-AE subscribes to the IN-AE FL control container.

```text
TinyIoT/IN-AE/cnt-fl-control/sub_mn1
└── nu: http://127.0.0.1:5001/notify

TinyIoT/IN-AE/cnt-fl-control/sub_mn2
└── nu: http://127.0.0.1:5002/notify

TinyIoT/IN-AE/cnt-fl-control/sub_mn3
└── nu: http://127.0.0.1:5003/notify
```

The event notification criteria are:

```json
{
  "enc": {
    "net": [3]
  }
}
```

`net = 3` requests a notification when a direct child resource, such as a new content instance, is created.

### 5.2 Local Update Subscriptions

IN-AE subscribes to the three node-specific update containers.

```text
TinyIoT/IN-AE/cnt-local-updates/cnt-mn1/sub_in_dropbox_mn1
TinyIoT/IN-AE/cnt-local-updates/cnt-mn2/sub_in_dropbox_mn2
TinyIoT/IN-AE/cnt-local-updates/cnt-mn3/sub_in_dropbox_mn3
```

All three subscriptions notify:

```text
http://127.0.0.1:6000/notify
```

### 5.3 Isaac Sim Command Subscription

The Isaac Sim gateway also watches the FL control container so that it can publish sensor metadata for the correct round.

```text
TinyIoT/IN-AE/cnt-fl-control
└── Isaac Sim gateway
    → detects the FL_TRAINING round command
    → writes the runtime buffer for that round
    → publishes metadata to MN-AE-X/cnt-sensor-data
```

---

## 6. End-to-End FL Sequence

### 6.1 Resource and Subscription Setup

```text
IN-AE                  TinyIoT CSE                  MN-AE-X
  |                         |                          |
  |---- CREATE AE --------->|                          |
  |---- CREATE CNT -------->|                          |
  |---- CREATE ACP -------->|                          |
  |                         |<------- CREATE AE -------|
  |                         |<------- CREATE CNT ------|
  |                         |<------- CREATE SUB ------|
  |---- CREATE SUB -------->|                          |
  |                         |                          |
```

The setup process creates:

```text
IN-AE
├── cnt-fl-control
├── cnt-global-model
└── cnt-local-updates
    ├── cnt-mn1
    ├── cnt-mn2
    └── cnt-mn3

MN-AE-X
├── cnt-sensor-data
└── cnt-local-model
```

### 6.2 Global Model Distribution

```text
IN-AE                  TinyIoT CSE                  MN-AE-X
  |                         |                          |
  |-- CREATE global CIN --->|                          |
  |-- CREATE command CIN -->|                          |
  |                         |-------- NOTIFY --------->|
  |                         |<------- response --------|
  |                         |<------ RETRIEVE ---------|
  |                         |---- model metadata ----->|
  |                         |<-- CREATE local cache ---|
  |                         |                          |
```

Process:

```text
1. IN-AE creates or loads the global model.
2. IN-AE publishes global model metadata.
3. IN-AE publishes an FL_TRAINING command.
4. TinyIoT sends command notifications.
5. Each MN-AE retrieves the global model metadata.
6. Each MN-AE stores the global model in its local cache.
```

### 6.3 Sensor Data Handover

```text
Isaac Sim              TinyIoT CSE                  MN-AE-X
  |                         |                          |
  |<--- round command ------|                          |
  |                         |                          |
  Write runtime buffer      |                          |
  mnX.pkl                   |                          |
  |                         |                          |
  |-- CREATE sensor CIN --->|                          |
  |                         |-------- NOTIFY --------->|
  |                         |                    Check metadata round
  |                         |                    Load buffer slice
  |                         |                          |
```

Process:

```text
1. Isaac Sim detects the FL_TRAINING round command.
2. Isaac Sim replays the round window range for each node.
3. Isaac Sim writes mn1.pkl, mn2.pkl, and mn3.pkl.
4. Isaac Sim publishes sensor metadata containing the round and the buffer path.
5. MN-AE compares the metadata round with the current FL round.
6. MN-AE proceeds only when the rounds match.
```

### 6.4 Local Training and Update Upload

```text
IN-AE                  TinyIoT CSE                  MN-AE-X
  |                         |                          |
  |                         |                    Load local data
  |                         |                    Train local model
  |                         |                    Validate model
  |                         |                          |
  |                         |<--- CREATE update CIN ---|
  |<------- NOTIFY ---------|                          |
  |-------- response ------>|                          |
  |                         |                          |
```

Process:

```text
1. MN-AE loads its local bearing dataset.
2. MN-AE trains the Conv1D Autoencoder.
3. MN-AE saves the local model.
4. MN-AE creates a local update content instance.
5. TinyIoT sends an update notification to IN-AE.
```

### 6.5 Aggregation and Next Round

```text
IN-AE                  TinyIoT CSE                  MN-AE-X
  |                         |                          |
  |<---- update NOTIFY -----|                          |
  |                         |                          |
  |  Validate local updates                            |
  |  Filter anomalous updates                          |
  |  Perform weighted FedAvg                           |
  |                         |                          |
  |-- CREATE global CIN --->|                          |
  |-- CREATE command CIN -->|                          |
  |                         |-------- NOTIFY --------->|
  |                         |                          |
```

Process:

```text
1. IN-AE waits for MN1, MN2, and MN3 updates.
2. IN-AE validates the node and round.
3. IN-AE filters potentially anomalous updates.
4. IN-AE performs weighted FedAvg.
5. IN-AE publishes the new global model.
6. IN-AE starts the next round.
7. The process repeats until Round 10.
```

---

## 7. FL States

```text
FL_READY
   ↓
FL_TRAINING
   ↓
FL_AGGREGATING
   ↓
FL_TRAINING
   ↓
...
   ↓
FL_COMPLETED
```

Example round command:

```json
{
  "type": "fl-command",
  "jobState": "FL_TRAINING",
  "currentRound": 1,
  "maxRounds": 10,
  "globalModelUri": "TinyIoT/IN-AE/cnt-global-model/la",
  "securityMode": "DP"
}
```

---

## 8. Model and Dataset

| Item | Value |
| --- | --- |
| Dataset | FEMTO-ST PRONOSTIA Bearing Dataset |
| Number of nodes | 3 |
| MN1 condition | 1800 rpm |
| MN2 condition | 1650 rpm |
| MN3 condition | 1500 rpm |
| Input | one-channel vibration signal |
| Sequence length | 2,560 samples |
| Sampling interpretation | 25.6 kHz × 0.1 seconds |
| Model | Conv1D Autoencoder |
| Learning method | unsupervised learning using normal data |
| Loss | MSE reconstruction loss |
| Optimizer | Adam |
| Aggregation | weighted FedAvg |
| Global rounds | 10 |
| Local epochs | 10 |
| Batch size | 32 |
| Anomaly score | reconstruction error |
| Fault decision | threshold exceeded three consecutive times |

### Replay Stream and Runtime Buffer

The Isaac replay demo does not feed raw FEMTO files directly. Windows are classified into normal, degradation, and fault states by RMS, then recomposed into a per-node time-series stream.

| Item | Value |
| --- | --- |
| Stream length | 800 windows per node |
| Windows per FL round | 80 |
| Rounds | 10 |
| MN1 stream | normal 800 |
| MN2 stream | normal 650 + degradation 150 |
| MN3 stream | normal 450 + degradation 170 + fault 180 |
| MN3 fault onset | window 620 |
| Train windows per round | 50 of 80, sampled with `np.linspace` |

MN3 never uses fault data for training. Its fault transition is detected by the global model that is learned together with the other nodes.

```text
prepare_data_femto_replay.py
→ femto_replay/mnX_stream.npz     preprocessing output

Isaac Sim gateway v7
→ fl_buffer/mnX.pkl               runtime buffer

TinyIoT cnt-sensor-data CIN
→ metadata only
  { type, jobState, node, round, currentRound,
    data_path, stream_path, start_idx, end_idx, window_count }
```

The Autoencoder is trained using normal vibration data.

Anomalies are detected using reconstruction error.

```text
Normal input
→ reconstructed accurately
→ low reconstruction error

Anomalous input
→ reconstructed poorly
→ high reconstruction error
```

The anomaly threshold is calculated from normal validation data.

```text
threshold
= normal validation MSE mean
+ 3 × standard deviation
```

A fault is detected when the reconstruction error exceeds the threshold three consecutive times.

---

## 9. Dashboard

The dashboard backend uses Flask and Server-Sent Events.

Start the dashboard:

```bash
python3 -u fl/dashboard_server.py
```

Open:

```text
http://localhost:7000
```

Health endpoint:

```text
http://localhost:7000/health
```

The dashboard displays:

- current FL state
- current FL round
- global model round
- MN-specific train loss
- MN-specific validation loss
- MN-specific validation AUROC
- reconstruction error graph
- node-specific anomaly thresholds
- final detection results

### 9.1 Continuous Graph Stream

The dashboard maintains `score_idx` when a new global model is loaded.

```text
Before modification:

Round 1 model loaded
→ score_idx = 0

Round 2 model loaded
→ score_idx = 0

Round 3 model loaded
→ score_idx = 0
```

This caused the graph to restart from the beginning every round.

The modified behavior is:

```text
Round 1 model loaded
→ score_idx continues

Round 2 model loaded
→ score_idx continues

Round 3 model loaded
→ score_idx continues
```

### 9.2 Stale Global Model Protection

A demo run marker is stored at:

```text
/tmp/fl_models/.demo_run_started
```

The dashboard ignores:

- global model files older than the current run marker
- global models from a future FL round
- files that do not follow the `global_roundN.pt` format

This prevents an old `global_round10.pt` file from being loaded during a new FL run.

A stale `FL_COMPLETED` state left in the CSE from a previous run is also rejected when a new run starts.

### 9.3 Anomaly Demonstration

In the legacy PKL path the demonstration policy is a deterministic input switch.

```text
MN1
→ normal samples during all rounds

MN2
→ normal samples during all rounds

MN3
→ normal samples during Rounds 1–6
→ anomaly samples from Round 7
```

The default start round is configured as:

```text
FL_ANOMALY_START_ROUND=7
```

It can be changed with:

```bash
export FL_ANOMALY_START_ROUND=6
```

It means that MN3 anomaly samples are displayed from Round 7.

It does not mean that the model independently predicts in advance that failure must occur at Round 7.

In the Isaac replay path this switch is not the source of the anomaly. The MN3 stream already contains the degradation and fault segments, and playback advances with the FL round, so the fault onset at window 620 falls inside the Round 8 playback range.

---

## 10. Demo Reset

`clean_fl.sh` prepares a fresh FL demonstration.

It preserves the preprocessed FEMTO PKL files.

`run_femto_isaac_demo.sh` performs its own reset, so `clean_fl.sh` is only required for the manual startup path.

### Reset Target

```text
Existing Dashboard process
Existing IN-AE process
Existing MN-AE processes
TinyIoT oneM2M resources
Global model files
Local model files
MN model caches
Previous demo run marker
```

### Preserved Data

```text
/tmp/fl_data/femto/mn1.pkl
/tmp/fl_data/femto/mn2.pkl
/tmp/fl_data/femto/mn3.pkl
```

### Reset Process

```text
clean_fl.sh
├── checks the preprocessed PKL files
├── terminates the existing Dashboard
├── terminates the existing IN-AE
├── terminates the existing MN-AEs
├── deletes global models
├── deletes local models
├── deletes MN caches
├── resets the TinyIoT database
├── recreates oneM2M resources
├── recreates ACP mappings
├── creates a new demo run marker
└── registers sensor data metadata
```

Give execution permission once:

```bash
chmod +x clean_fl.sh
```

Run:

```bash
sudo -v
./clean_fl.sh
```

The local TinyIoT PostgreSQL configuration is:

```text
Database: tinydb
Port: 5433
Administrative user: postgres
```

Environment variables can override these values:

```bash
export TINYIOT_DB_NAME=tinydb
export TINYIOT_DB_PORT=5433
export TINYIOT_DB_ADMIN_USER=postgres
```

> `clean_fl.sh` resets the configured TinyIoT resource database. Use it only with the dedicated demo database.

After reset, refresh the oneM2M Design Tool page.

---

## 11. Project Structure

```text
FL
├── fl
│   ├── aggregator.py
│   ├── config.py
│   ├── dashboard_server.py
│   ├── data_generator.py
│   ├── edge_node.py
│   ├── in_ae_standard.py
│   ├── mn_ae_standard.py
│   ├── model.py
│   ├── onem2m_utils.py
│   ├── personalize.py
│   ├── prepare_data_femto.py
│   ├── prepare_data_femto_replay.py
│   └── setup_resources_standard.py
│
├── isaac_sim
│   └── isaac_femto_demo_gateway_v7.py
│
├── femto_replay
│   ├── mn1_stream.npz
│   ├── mn2_stream.npz
│   └── mn3_stream.npz
│
├── fl_buffer
│   ├── mn1.pkl
│   ├── mn2.pkl
│   └── mn3.pkl
│
├── archive
│   └── experiments
│
├── clean_fl.sh
├── fl_bearing_dashboard.html
├── fljob.yaml
├── main.py
├── README.md
├── requirements.txt
├── run_fl.sh
└── run_femto_isaac_demo.sh
```

### Main Files

| File | Role |
| --- | --- |
| `fl/in_ae_standard.py` | IN-AE notification receiver, coordinator, validator, and aggregator |
| `fl/mn_ae_standard.py` | MN-AE notification receiver, local trainer, model cache manager, and update uploader |
| `fl/setup_resources_standard.py` | Creates oneM2M resources and ACP mappings and optionally resets TinyIoT DB |
| `fl/data_generator.py` | Registers local dataset metadata under `cnt-sensor-data` |
| `fl/onem2m_utils.py` | Provides oneM2M CREATE, RETRIEVE, UPDATE, DELETE, and subscription functions |
| `fl/aggregator.py` | Performs weighted FedAvg aggregation |
| `fl/model.py` | Defines the Conv1D Autoencoder |
| `fl/prepare_data_femto.py` | Converts PRONOSTIA data into node-specific PKL files |
| `fl/prepare_data_femto_replay.py` | Builds per-node replay streams (`*_stream.npz`) from PRONOSTIA data |
| `fl/dashboard_server.py` | Polls FL state, calculates scores, and sends dashboard SSE events |
| `isaac_sim/isaac_femto_demo_gateway_v7.py` | Isaac Sim gateway: stream replay, visualization, buffer writer, metadata publisher |
| `run_femto_isaac_demo.sh` | Isaac replay demo runner: state reset, Dashboard, MN-AE, and IN-AE startup |
| `clean_fl.sh` | Resets processes, DB resources, model files, and caches while preserving PKL data |
| `fl_bearing_dashboard.html` | Dashboard frontend |

---

## 12. Environment Setup

The demo spans two environments.

| Side | Runs | Notes |
| --- | --- | --- |
| WSL2 Ubuntu | TinyIoT CSE, IN-AE, MN-AE ×3, Dashboard | FL code and oneM2M platform |
| Windows | NVIDIA Isaac Sim gateway | Visualization and sensor metadata publication |

Both sides exchange files through the Windows path `C:\Projects\bearing_testbed`, which is visible from WSL as `/mnt/c/Projects/bearing_testbed`.

```bash
cd ~/projects/federated-learning

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

If required:

```bash
pip install torch numpy pandas pyyaml flask scikit-learn requests
```

TinyIoT must be running at:

```text
http://127.0.0.1:3000
```

Isaac Sim must be able to reach the same endpoint from Windows. If `http://127.0.0.1:3000` is not reachable from the Windows side, enable WSL2 mirrored networking or set `TINYIOT_BASE_URL` in the gateway script to the WSL IP.

The gateway script and the data directories must exist on the Windows side:

```text
C:\Projects\bearing_testbed\scripts\isaac_femto_demo_gateway_v7.py
C:\Projects\bearing_testbed\data\femto_replay
C:\Projects\bearing_testbed\data\fl_buffer
```

---

## 13. Ports

| Component | Port |
| --- | ---: |
| TinyIoT CSE | 3000 |
| MN-AE-1 notification server | 5001 |
| MN-AE-2 notification server | 5002 |
| MN-AE-3 notification server | 5003 |
| IN-AE notification server | 6000 |
| Dashboard | 7000 |

Node IDs and ports must remain consistent.

```text
node_id 0
→ mn1
→ port 5001

node_id 1
→ mn2
→ port 5002

node_id 2
→ mn3
→ port 5003
```

---

## 14. How to Run

### 14.0 Isaac Replay Demo (recommended)

This is the current demo path. The runner performs its own reset and does not call `fl/data_generator.py`, because sensor metadata is published by Isaac Sim.

Prepare the replay streams once:

```bash
cd ~/projects/federated-learning
source .venv/bin/activate

python3 fl/prepare_data_femto_replay.py

cp /tmp/fl_data/femto_replay/*_stream.npz \
   /mnt/c/Projects/bearing_testbed/data/femto_replay/
```

With TinyIoT already running, start the demo:

```bash
cd ~/projects/federated-learning
source .venv/bin/activate

./run_femto_isaac_demo.sh
```

Options:

```text
--skip-clean         skip the demo reset
--no-prompt          start IN-AE without waiting for Enter
--exit-on-complete   exit after FL_COMPLETED
--full-train         use all 80 windows per round
```

When the runner prints the Isaac Sim step, open Isaac Sim on Windows and run this line in `Window > Script Editor`:

```python
exec(open(r"C:\Projects\bearing_testbed\scripts\isaac_femto_demo_gateway_v7.py", encoding="utf-8").read())
```

Wait for the gateway to report that it is waiting:

```text
[start] FEMTO Isaac synced demo started - v7
        Waiting for IN-AE FL_TRAINING round command...
```

Then press Enter in the runner terminal. IN-AE starts and the rounds begin.

```text
TinyIoT
   ↓
./run_femto_isaac_demo.sh          Dashboard, MN-AE ×3
   ↓
Isaac Sim gateway v7               Windows Script Editor
   ↓
Enter in the runner terminal
   ↓
IN-AE  →  Round 1 … Round 10
```

Logs for each run are written to:

```text
logs/femto_isaac_demo/<RUN_ID>
```

The sections below describe the manual startup path without Isaac Sim.

---

### 14.1 Start TinyIoT

```bash
cd ~/projects/tinyIoT/source/server

./server
```

Check the TinyIoT server:

```bash
curl -i http://127.0.0.1:3000
```

### 14.2 Activate the Virtual Environment

Run the following commands in every FL terminal.

```bash
cd ~/projects/federated-learning

source .venv/bin/activate
```

### 14.3 Prepare PRONOSTIA Data

Run this only when:

- the PKL files do not exist
- the source dataset changes
- the preprocessing code changes

```bash
python3 fl/prepare_data_femto.py
```

Expected files:

```text
/tmp/fl_data/femto/mn1.pkl
/tmp/fl_data/femto/mn2.pkl
/tmp/fl_data/femto/mn3.pkl
```

If these files already exist, preprocessing can be skipped.

### 14.4 Reset the Demo

Run this before starting a completely new demonstration.

```bash
sudo -v

./clean_fl.sh
```

`clean_fl.sh` already performs:

```text
TinyIoT DB reset
oneM2M resource recreation
ACP recreation
old model removal
old cache removal
data_generator.py execution
```

Do not immediately run the following commands again after `clean_fl.sh`.

```bash
python3 fl/setup_resources_standard.py

python3 fl/data_generator.py
```

### 14.5 Start the Dashboard

To observe the graph from the initial state, start the dashboard before starting FL.

```bash
python3 -u fl/dashboard_server.py
```

Open:

```text
http://localhost:7000
```

The dashboard can also be started after the MN-AEs, but early round changes may not be visible.

### 14.6 Start IN-AE

Open a new terminal.

```bash
cd ~/projects/federated-learning

source .venv/bin/activate

python3 -u fl/in_ae_standard.py
```

Wait at least three seconds.

IN-AE starts its notification server and prepares the FL coordinator.

### 14.7 Start MN-AE-1

Open a new terminal.

```bash
cd ~/projects/federated-learning

source .venv/bin/activate

python3 -u fl/mn_ae_standard.py 0 5001
```

### 14.8 Start MN-AE-2

Open a new terminal.

```bash
cd ~/projects/federated-learning

source .venv/bin/activate

python3 -u fl/mn_ae_standard.py 1 5002
```

### 14.9 Start MN-AE-3

Open a new terminal.

```bash
cd ~/projects/federated-learning

source .venv/bin/activate

python3 -u fl/mn_ae_standard.py 2 5003
```

### Recommended Startup Order

```text
TinyIoT
   ↓
Optional PRONOSTIA preprocessing
   ↓
clean_fl.sh
   ↓
Dashboard
   ↓
IN-AE
   ↓
Wait at least 3 seconds
   ↓
MN-AE-1
MN-AE-2
MN-AE-3
   ↓
Round 0 and Round 1–10 visualization
```

IN-AE waits until the required MN command subscriptions are available before beginning the FL rounds.

---

## 15. Manual Preparation Without clean_fl.sh

Resources can also be prepared manually.

```bash
python3 fl/setup_resources_standard.py

python3 fl/data_generator.py
```

Manual preparation does not automatically remove:

- old global models
- old local models
- old MN caches
- old Dashboard processes
- old IN-AE processes
- old MN-AE processes

For a completely fresh demonstration, `clean_fl.sh` is recommended.

---

## 16. Expected Runtime Flow

```text
1. Dashboard starts at FL_READY.

2. IN-AE starts the notification server.

3. MN-AEs start their notification servers.

4. MN-AEs create command subscriptions.

5. IN-AE confirms the MN subscriptions.

6. IN-AE publishes the Round 0 global model.

7. IN-AE publishes FL_TRAINING for Round 1.

8. TinyIoT sends command NOTIFY messages.

9. MN-AEs retrieve the global model.

10. Isaac Sim writes the round buffer and publishes sensor metadata.

11. MN-AEs confirm the metadata round and perform local training.

12. MN-AEs upload local update CINs.

13. TinyIoT sends update NOTIFY messages to IN-AE.

14. IN-AE collects three local updates.

15. IN-AE publishes FL_AGGREGATING.

16. IN-AE filters anomalous updates.

17. IN-AE performs weighted FedAvg.

18. IN-AE publishes the next global model.

19. The process repeats until Round 10.

20. IN-AE publishes FL_COMPLETED.

21. The dashboard displays the final result.
```

Steps 10 and 11 apply to the Isaac replay path. In the manual path the sensor metadata is registered once by `data_generator.py`.

Example MN-AE log:

```text
[NOTIFY] mn1 command received: FL_TRAINING, round=1

Round 1 command received via NOTIFY

Local training started

Local update uploaded

Waiting for Round 2
```

Example IN-AE log:

```text
[COLLECT/NOTIFY] mn1 Round 1 (1/3)

[COLLECT/NOTIFY] mn2 Round 1 (2/3)

[COLLECT/NOTIFY] mn3 Round 1 (3/3)

All results received by NOTIFY

FedAvg Round 1

Global model published Round 1
```

---

## 17. Output Paths

### Preprocessed Data

```text
/tmp/fl_data/femto                 legacy PKL path
├── mn1.pkl
├── mn2.pkl
└── mn3.pkl

/tmp/fl_data/femto_replay          replay stream output
├── mn1_stream.npz
├── mn2_stream.npz
└── mn3_stream.npz
```

### Isaac Sim Shared Paths

```text
C:\Projects\bearing_testbed\data\femto_replay     stream input
C:\Projects\bearing_testbed\data\fl_buffer        runtime buffer
C:\Projects\bearing_testbed\scripts               gateway script

/mnt/c/Projects/bearing_testbed/data/fl_buffer    same buffer, seen from WSL
```

### Global Models

```text
/tmp/fl_models/global
├── global_round0.pt
├── global_round1.pt
├── global_round2.pt
├── ...
└── global_round10.pt
```

### Local Models

```text
/tmp/fl_models/local
├── mn1
├── mn2
└── mn3
```

### Edge Cache

```text
/tmp/fl_models/cache
├── mn1
├── mn2
└── mn3
```

### Demo Run Marker

```text
/tmp/fl_models/.demo_run_started
```

### Demo Logs

```text
logs/femto_isaac_demo/<RUN_ID>
```

---

## 18. Configuration

Main configuration files:

```text
fl/config.py

fljob.yaml
```

Useful environment variables:

```text
FL_PKL_DIR
FL_MODEL_BASE_DIR
FL_DEMO_RUN_MARKER
FL_ANOMALY_DEMO_NODE
FL_ANOMALY_START_ROUND

FL_DASHBOARD_PORT
FL_DASHBOARD_POLL_INTERVAL
FL_DASHBOARD_SCORE_INTERVAL

FL_COMMAND_NOTIFY_TIMEOUT
FL_COLLECTION_NOTIFY_TIMEOUT

TINYIOT_BASE_URL
TINYIOT_CSE_NAME

TINYIOT_DB_NAME
TINYIOT_DB_PORT
TINYIOT_DB_ADMIN_USER
```

Isaac replay demo variables:

```text
FEMTO_REPLAY_DIR            replay stream directory
FL_PKL_DIR                  runtime buffer directory
FL_SENSOR_ROUND_WAIT_SEC    MN-AE wait limit for matching sensor metadata
FL_SENSOR_ROUND_POLL_SEC    MN-AE poll interval while waiting
FL_DEMO_ROUND_TRAIN_N       training windows sampled per round
ISAAC_SCRIPT_WIN            gateway script path shown in the runner instruction
LOG_DIR                     demo log directory
```

Example:

```bash
export FL_ANOMALY_START_ROUND=7

export FL_DASHBOARD_PORT=7000

export TINYIOT_DB_NAME=tinydb

export TINYIOT_DB_PORT=5433

export FEMTO_REPLAY_DIR=/mnt/c/Projects/bearing_testbed/data/femto_replay

export FL_PKL_DIR=/mnt/c/Projects/bearing_testbed/data/fl_buffer
```

---

## 19. Troubleshooting

### 19.1 Dashboard Starts at Round 10

Cause:

```text
An old Dashboard process is still running

or

Old TinyIoT resources and model files remain
```

Reset the demo:

```bash
sudo -v

./clean_fl.sh
```

Check port 7000:

```bash
ss -ltnp | grep :7000
```

No output means the port is free.

### 19.2 Port 7000 Is Already in Use

```text
Address already in use

Port 7000 is in use by another program
```

Stop the previous Dashboard:

```bash
pkill -f "fl/dashboard_server.py" || true
```

Then restart:

```bash
python3 -u fl/dashboard_server.py
```

### 19.3 Old Global Model Is Loaded

Check:

```bash
ls -l /tmp/fl_models/global
```

Reset:

```bash
./clean_fl.sh
```

The dashboard also ignores model files older than:

```text
/tmp/fl_models/.demo_run_started
```

### 19.4 MN Notification Port Is Already in Use

```bash
pkill -f "fl/mn_ae_standard.py" || true
```

Then restart the MN-AEs.

### 19.5 IN-AE Notification Port Is Already in Use

```bash
pkill -f "fl/in_ae_standard.py" || true
```

Then restart IN-AE.

### 19.6 TinyIoT Database Error

Current local database configuration:

```text
Database: tinydb
Port: 5433
Administrator: postgres
```

Check the database:

```bash
sudo -u postgres psql \
  -p 5433 \
  -d tinydb \
  -c "SELECT current_database(), current_user;"
```

### 19.7 Missing PKL Files

```text
/tmp/fl_data/femto/mn1.pkl
/tmp/fl_data/femto/mn2.pkl
/tmp/fl_data/femto/mn3.pkl
```

Generate them:

```bash
python3 fl/prepare_data_femto.py
```

### 19.8 FL Stops at Round 1 in the Isaac Demo

Cause:

```text
The virtual environment was not activated

or

Isaac Sim was started after IN-AE

or

Isaac Sim cannot reach TinyIoT from Windows
```

Activate the virtual environment before running the demo:

```bash
source .venv/bin/activate
```

Start Isaac Sim first and wait for the gateway to report that it is waiting for the round command, then press Enter in the runner terminal.

Check reachability from Windows:

```text
http://127.0.0.1:3000
```

If this fails, enable WSL2 mirrored networking or set the WSL IP in the gateway script.

### 19.9 MN-AE Waits for Sensor Metadata

```text
Waiting for sensor metadata of round N
```

Cause:

```text
The Isaac Sim gateway is not running

or

The gateway published metadata for a different round
```

Restart the gateway:

```python
stop_femto_demo()
```

Then run the `exec(...)` line again.

---

## 20. Isaac Sim Integration

Isaac Sim is integrated as the sensor data source for the demo. It does not replace the oneM2M FL workflow; it supplies the data that the workflow carries.

### 20.1 Role

```text
1. reads the FEMTO-based replay stream
2. visualizes the MN1 / MN2 / MN3 bearing units
3. writes the FL runtime buffer (pkl)
4. watches cnt-fl-control for the FL round command
5. publishes round metadata to MN-AE-X/cnt-sensor-data
6. reflects the dashboard fault decision in the visualization
```

Preprocessing and postprocessing are performed on the WSL side. Isaac Sim replays prepared signals and acts as the gateway between the virtual equipment and the CSE.

### 20.2 Node Conditions

| Node | Stream Composition | Purpose |
| --- | --- | --- |
| MN1 | normal only | normal reference |
| MN2 | normal and degradation | non-IID normal condition |
| MN3 | normal, degradation, and fault | anomaly demonstration |

### 20.3 Round Synchronization

Isaac Sim must be started before IN-AE. IN-AE publishes the round command, Isaac Sim detects it and publishes the matching sensor metadata, and MN-AE waits until the metadata round equals the current round before training.

```text
R1  : windows   0– 79
R2  : windows  80–159
...
R8  : windows 560–639      MN3 fault onset at 620
R9  : windows 640–719
R10 : windows 720–799
```

### 20.4 Fault Indicator

The bearing indicator stays green regardless of the internal state code. It turns red only after the dashboard reports three consecutive score exceedances for that node, so the on-screen fault is the federated learning result rather than simulator foreknowledge.

### 20.5 Idle Behavior

Between rounds the gateway loops the last 24 windows at 6 windows per second so the equipment keeps moving. After the Round 10 visualization completes, the controller stops automatically.

### 20.6 Stop and Restart

```python
stop_femto_demo()
```

Re-running the `exec(...)` line restarts the gateway and clears the previous subscription.

### 20.7 Current Limitations

The current Isaac Sim integration does not provide:

- true 25.6 kHz PhysX simulation
- physics-generated vibration signals
- exact crack propagation
- exact BPFO or BPFI physical reproduction
- bearing temperature physics
- remaining useful life prediction

The signals are replayed from the PRONOSTIA dataset. Isaac Sim provides the equipment view, the round-synchronized data handover, and the fault visualization.

### 20.8 Next Phase: ROS 2 Bridge and IPE

```text
Isaac Sim
   ↓
ROS 2 Topic
   ↓
ROS 2 IPE
   ↓
oneM2M contentInstance
   ↓
MN-AE
```

The next phase requires:

- ROS 2 topic definitions
- vibration window buffering
- ROS 2 message conversion
- oneM2M IPE implementation
- physics-based signal generation instead of file replay
- network-accessible model storage for multi-PC execution

---

## 21. Implementation Status

### Implemented

- oneM2M AE, container, CIN, subscription, and ACP setup
- three MN-AE command subscriptions
- three IN-AE local update subscriptions
- notification verification
- command notification delivery
- local update notification delivery
- notification body parsing
- latest-CIN recovery after notification timeout
- ten-round FL execution
- Conv1D Autoencoder local training
- differential privacy configuration
- local data retention
- anomalous update filtering
- weighted FedAvg aggregation
- global model redistribution
- SSE dashboard visualization
- continuous dashboard score index
- stale global model protection
- configurable MN3 anomaly start round
- TinyIoT DB reset
- global and local model reset
- MN cache reset
- old FL process termination
- final evaluation
- Isaac Sim replay-based virtual bearing testbed
- Isaac Sim runtime buffer generation
- round-synchronized sensor metadata publication
- MN-AE round-identifier matching before training
- stale FL_COMPLETED rejection on a new run
- dashboard-driven fault indicator in Isaac Sim
- one-command Isaac replay demo runner

### Planned

- ROS 2 Bridge
- ROS 2 IPE
- physics-based vibration signal generation in Isaac Sim
- multi-CSE deployment
- multi-PC deployment
- HTTP, S3, or MinIO model artifact transfer
- native `<flJob>` resource type in TinyIoT

---

## 22. Notes

- Raw training data remains on the MN-AE side.
- IN-AE receives local model references and training metadata.
- Actual model files are stored in local filesystem paths.
- The current path-based model transfer is suitable for a single-PC demonstration.
- Multi-PC deployment requires HTTP, S3, MinIO, or another network-accessible model store.
- A missing notification is recovered with one latest-CIN RETRIEVE.
- The FL coordinator does not continuously poll for local updates.
- Dashboard polling is used only for visualization.
- Only one Dashboard process can use port 7000.
- Starting a new demo without clearing old processes, models, caches, and CSE resources may expose stale Round 10 data.
- The Round 7 anomaly switch controls which MN3 test samples are displayed in the legacy PKL path.
- In the Isaac replay path the anomaly comes from the MN3 stream itself, and playback advances with the FL round.
- Isaac Sim must be started before IN-AE so that the first round command is not missed.
- Sensor signals are handed over through the file buffer; only metadata is published to the CSE.
