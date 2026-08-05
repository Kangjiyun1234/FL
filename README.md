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
- a planned Isaac Sim file-based virtual bearing data pipeline

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

### 6.3 Local Training and Update Upload

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

### 6.4 Aggregation and Next Round

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

### 9.3 Round 7 Anomaly Demonstration

The default dashboard demonstration policy is:

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

This is a deterministic demo-input switch.

It means that MN3 anomaly samples are displayed from Round 7.

It does not mean that the model independently predicts in advance that failure must occur at Round 7.

---

## 10. Demo Reset

`clean_fl.sh` prepares a fresh FL demonstration.

It preserves the preprocessed FEMTO PKL files.

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
│   └── setup_resources_standard.py
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
└── run_fl.sh
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
| `fl/dashboard_server.py` | Polls FL state, calculates scores, and sends dashboard SSE events |
| `clean_fl.sh` | Resets processes, DB resources, model files, and caches while preserving PKL data |
| `fl_bearing_dashboard.html` | Dashboard frontend |

---

## 12. Environment Setup

The current prototype is designed for Linux or WSL2.

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

The recommended procedure uses separate terminals.

This makes it easier to inspect each AE log independently.

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

10. MN-AEs perform local training.

11. MN-AEs upload local update CINs.

12. TinyIoT sends update NOTIFY messages to IN-AE.

13. IN-AE collects three local updates.

14. IN-AE publishes FL_AGGREGATING.

15. IN-AE filters anomalous updates.

16. IN-AE performs weighted FedAvg.

17. IN-AE publishes the next global model.

18. The process repeats until Round 10.

19. IN-AE publishes FL_COMPLETED.

20. The dashboard displays the final result.
```

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
/tmp/fl_data/femto
├── mn1.pkl
├── mn2.pkl
└── mn3.pkl
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

Example:

```bash
export FL_ANOMALY_START_ROUND=7

export FL_DASHBOARD_PORT=7000

export TINYIOT_DB_NAME=tinydb

export TINYIOT_DB_PORT=5433
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

---

## 20. Isaac Sim Integration Roadmap

Isaac Sim integration is planned as an additional data source.

It does not replace the current oneM2M FL workflow.

### 20.1 Phase 1: File-Based Data Dumper

```text
Isaac Sim
   ↓
Simplified virtual bearing testbed
   ↓
Shaft RPM and housing load
   ↓
IMU or physics-state collection
   ↓
Python-generated impact and noise
   ↓
CSV or NumPy output
   ↓
Node-specific PKL conversion
   ↓
cnt-sensor-data path registration
   ↓
Existing MN-AE local training
   ↓
Existing oneM2M notification workflow
   ↓
IN-AE FedAvg aggregation
```

### Initial Virtual Testbed

```text
Isaac Sim World
└── PRONOSTIA_Testbed
    ├── Base
    ├── Housing
    │   ├── Accel_Sensor_X
    │   └── Accel_Sensor_Y
    │
    └── Shaft
        └── EccentricMass
```

Initial implementation scope:

- Cube-based housing
- Cylinder-based shaft
- simplified bearing geometry
- approximately 1,800 rpm shaft operation
- approximately 4,000 N housing load
- X-axis and Y-axis acceleration positions
- node-specific RPM and load conditions
- normal, degradation, and fault stages

### Proposed Node Conditions

| Node | Example Condition | Purpose |
| --- | --- | --- |
| MN1 | stable RPM and stable load | normal reference |
| MN2 | different RPM and load | non-IID normal condition |
| MN3 | normal-to-fault transition | anomaly demonstration |

### Hybrid Signal Generation

A simplified Isaac Sim model cannot reproduce every physical characteristic of a real damaged bearing.

The initial signal generation method is:

```text
Final vibration signal
=
Isaac Sim IMU or motion component
+
RPM-related periodic component
+
Synthetic fault impulse
+
Measurement noise
```

The final input window remains:

```text
Sampling rate interpretation: 25,600 Hz

Window duration: 0.1 seconds

Sequence length: 2,560 samples
```

### Proposed Output Files

Raw Isaac Sim output:

```text
/tmp/fl_data/isaac/raw
├── mn1_raw.npz
├── mn2_raw.npz
└── mn3_raw.npz
```

Processed FL input:

```text
/tmp/fl_data/isaac
├── mn1.pkl
├── mn2.pkl
└── mn3.pkl
```

The processed PKL files should use the current data contract.

```python
{
    "train_signals": ...,
    "val_signals": ...,
    "val_labels": ...,
    "test_stream_signals": ...,
    "test_stream_labels": ...,
    "norm_mean": ...,
    "norm_std": ...,
    "seq_len": 2560,
    "n_channels": 1,
    "node": "mn1",
    "source": "isaac-sim"
}
```

### 20.2 Phase 2: ROS 2 Bridge and IPE

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

The second phase requires:

- ROS 2 topic definitions
- vibration window buffering
- ROS 2 message conversion
- oneM2M IPE implementation
- FL round and data-buffer synchronization
- network-accessible model storage for multi-PC execution

### Isaac Sim Limitations

The first Isaac Sim demo does not aim to provide:

- true 25.6 kHz PhysX simulation
- exact PRONOSTIA signal reproduction
- exact crack propagation
- exact BPFO or BPFI physical reproduction
- complete bearing temperature physics
- complete run-to-failure reproduction
- accurate remaining useful life prediction

The first completion criterion is:

```text
Isaac Sim generates node-specific vibration files

→ the files are converted into the current PKL format

→ data paths are registered in cnt-sensor-data

→ the existing oneM2M-FL pipeline completes ten rounds

→ the dashboard displays the generated signals
```

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

### Planned

- Isaac Sim basic-geometry bearing testbed
- virtual bearing Data Dumper
- Isaac Sim NPZ-to-PKL converter
- Isaac Sim source metadata registration
- ROS 2 Bridge
- ROS 2 IPE
- multi-CSE deployment
- multi-PC deployment
- HTTP, S3, or MinIO model artifact transfer

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
- The Round 7 anomaly switch controls which MN3 test samples are displayed.
