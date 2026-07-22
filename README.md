# FL: Notification-Driven oneM2M Federated Learning Prototype

## 1. Overview

This repository implements a oneM2M resource-based federated learning workflow for bearing fault detection using the FEMTO-ST PRONOSTIA dataset.

The implementation follows the direction of **oneM2M TR-0084: Developer guide; Use of oneM2M resources to support Federated Learning**. Standard oneM2M resources such as `<AE>`, `<container>`, `<contentInstance>`, `<subscription>`, and ACP-linked resources are used to coordinate the complete FL lifecycle.

The current prototype provides:

- notification-driven FL round control
- global model metadata distribution
- local training at three MN-AEs
- per-node local update upload
- subscription-based local update collection
- FedAvg aggregation
- loss Z-score-based anomalous update filtering
- ACP-based access control verification
- dashboard visualization
- final cold-start evaluation

The current implementation runs on a **single TinyIoT CSE**. `IN-AE` acts as the coordinator, while `MN-AE-1`, `MN-AE-2`, and `MN-AE-3` act as local training clients.

---

## 2. System Architecture

```text
TinyIoT CSE (http://127.0.0.1:3000)
├── IN-AE
│   ├── cnt-fl-control
│   │   ├── FL state and round command CINs
│   │   ├── sub_mn1
│   │   ├── sub_mn2
│   │   └── sub_mn3
│   │
│   ├── cnt-global-model
│   │   └── Global model metadata CINs
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
| `IN-AE` | Starts FL rounds, publishes commands, receives local update notifications, validates updates, and performs aggregation |
| `MN-AE-1` | Trains on Condition 1 bearing data and uploads a local update |
| `MN-AE-2` | Trains on Condition 2 bearing data and uploads a local update |
| `MN-AE-3` | Trains on Condition 3 bearing data and uploads a local update |
| `TinyIoT CSE` | Stores oneM2M resources and delivers subscription notifications |
| Dashboard | Retrieves current resource state for visualization only |

The sequence diagrams use one logical **MN-side** participant to represent the repeated `MN-CSE + MN-AE[i]` behavior. The same sequence is executed for `i = 1..3`.

---

## 3. oneM2M Resource Mapping

| FL Function | oneM2M Resource | Stored Content |
| --- | --- | --- |
| Round control | `IN-AE/cnt-fl-control` | `jobState`, `currentRound`, `maxRounds`, `globalModelUri`, privacy configuration |
| Global model registry | `IN-AE/cnt-global-model` | global round, model path, model-ready state |
| Local update drop-box | `IN-AE/cnt-local-updates/cnt-mnX` | node, round, model path, losses, AUROC, sample count |
| Local sensor metadata | `MN-AE-X/cnt-sensor-data` | local FEMTO data path and metadata |
| Edge model cache | `MN-AE-X/cnt-local-model` | cached global model metadata |
| Round command delivery | `<subscription>` under `cnt-fl-control` | notification target for each MN-AE |
| Local update delivery | `<subscription>` under each `cnt-mnX` | notification target for IN-AE |
| Access control | ACP + `acpi` | resource access permissions for IN-AE and each MN-AE |

### TinyIoT Label Compatibility

The label-based filtering fields used in the initial design were removed because the current TinyIoT implementation does not provide the required label-based FL retrieval path. Instead, `type`, `round`, and `node` are stored and validated inside `contentInstance.con`.

For TinyIoT compatibility, dictionary content is serialized as a JSON string before being stored in `contentInstance.con`.

---

## 4. Notification-Driven FL Flow

The main FL workflow does **not** continuously poll for round commands or local updates.

- Round commands are delivered from TinyIoT to each MN-AE through `<subscription>` notifications.
- Local update arrivals are delivered from TinyIoT to IN-AE through `<subscription>` notifications.
- If a notification is missing, the receiver performs a **single latest-CIN RETRIEVE** as a recovery path.
- The dashboard uses periodic GET requests only to visualize the current state. Dashboard polling does not control FL execution.

### Command Subscriptions

```text
TinyIoT/IN-AE/cnt-fl-control/sub_mn1
  nu = http://127.0.0.1:5001/notify

TinyIoT/IN-AE/cnt-fl-control/sub_mn2
  nu = http://127.0.0.1:5002/notify

TinyIoT/IN-AE/cnt-fl-control/sub_mn3
  nu = http://127.0.0.1:5003/notify
```

Each subscription uses:

```json
{
  "enc": {
    "net": [3]
  }
}
```

`net = 3` triggers a notification when a direct child resource, such as a new `<contentInstance>`, is created.

### Local Update Subscriptions

```text
TinyIoT/IN-AE/cnt-local-updates/cnt-mn1/sub_in_dropbox_mn1
TinyIoT/IN-AE/cnt-local-updates/cnt-mn2/sub_in_dropbox_mn2
TinyIoT/IN-AE/cnt-local-updates/cnt-mn3/sub_in_dropbox_mn3
```

All three notify:

```text
http://127.0.0.1:6000/notify
```

---

## 5. Final Sequence

### A. Initial Resource and Subscription Setup

1. Create IN-AE and MN-AE resources.
2. Create ACP resources and connect them using `acpi`.
3. Create `cnt-fl-control`, `cnt-global-model`, and node-specific local update containers.
4. Create `cnt-sensor-data` and `cnt-local-model` for each MN-AE.
5. Start the MN-AE notification servers.
6. Each MN-AE subscribes to `cnt-fl-control`.
7. IN-AE starts its notification server and subscribes to each `cnt-mnX` container.
8. IN-AE confirms that `sub_mn1`, `sub_mn2`, and `sub_mn3` exist before starting FL.

```text
IN-AE                 TinyIoT CSE                 MN-side[i]
  |                         |                          |
  |                         |<-- CREATE AE/CNT/ACP ----|
  |-- CREATE AE/CNT/ACP --->|                          |
  |                         |<-- CREATE subscription --|
  |                         |    to cnt-fl-control     |
  |-- CREATE subscription ->|                          |
  |   to cnt-mnX            |                          |
  |<---- VERIFY NOTIFY -----|----- VERIFY NOTIFY ----->|
  |                         |                          |
```

`MN-side[i]` combines the logical behavior of `MN-CSE[i]` and `MN-AE[i]`. The same setup is repeated for `i = 1..3`.

### B. Global Model Distribution and Round Command

1. IN-AE creates the initial or newly aggregated global model file.
2. IN-AE creates a global model metadata CIN under `cnt-global-model`.
3. IN-AE creates an `FL_TRAINING` command CIN under `cnt-fl-control`.
4. TinyIoT sends the command NOTIFY to every MN-AE notification endpoint.
5. Each MN-AE reads the command content and retrieves the global model metadata using `globalModelUri`.
6. Each MN-AE stores the retrieved model metadata in `cnt-local-model` and updates its local model cache.

```text
IN-AE                 TinyIoT CSE                 MN-side[i]
  |                         |                          |
  |-- CREATE global CIN --->|                          |
  |-- CREATE command CIN -->|                          |
  |                         |------ NOTIFY ----------->|
  |                         |<----- response ----------|
  |                         |<----- RETRIEVE ----------|
  |                         |------ model metadata --->|
  |                         |<-- CREATE local cache ---|
  |                         |                          |
```

The command notification and global model retrieval sequence is repeated for each MN side.

### C. Local Training and Update Upload

1. Each MN-AE loads its local FEMTO bearing data.
2. Each MN-AE trains a Conv1D Autoencoder using local normal data.
3. Differential privacy noise is applied during local training according to the configured privacy parameters.
4. Each MN-AE stores its local model artifact locally.
5. Each MN-AE creates a local update CIN in its node-specific drop-box container.
6. TinyIoT sends a local update NOTIFY to the IN-AE notification endpoint.
7. IN-AE reads `rep.m2m:cin.con` and records the update by node and round.

```text
IN-AE                 TinyIoT CSE                 MN-side[i]
  |                         |                          |
  |                         |                   Local training
  |                         |                          |
  |                         |<-- CREATE update CIN ----|
  |<-------- NOTIFY --------|                          |
  |-------- response ------>|                          |
  |                         |                          |
```

Each MN side uploads its local update to its own `cnt-mnX` drop-box container.

### D. Global Aggregation and Next Round

1. IN-AE waits until all three local update notifications are received.
2. If a notification is missing at timeout, IN-AE retrieves the latest CIN once from only the missing node container.
3. IN-AE validates `type`, `round`, and `node` from each update payload.
4. IN-AE publishes `FL_AGGREGATING` under `cnt-fl-control`.
5. Loss Z-score-based anomaly detection excludes potentially poisoned updates when applicable.
6. IN-AE performs weighted FedAvg using valid local model states and sample counts.
7. IN-AE stores the new global model and publishes its metadata under `cnt-global-model`.
8. IN-AE publishes the next `FL_TRAINING` command, and the notification-driven cycle repeats.
9. After the final round, IN-AE publishes `FL_COMPLETED` and runs the cold-start hidden test evaluation.

```text
IN-AE                 TinyIoT CSE                 MN-side[i]
  |                         |                          |
  |<---- NOTIFY updates ----|                          |
  |                         |                          |
  |   Validate and aggregate local model updates      |
  |                         |                          |
  |-- CREATE global CIN --->|                          |
  |-- CREATE next command ->|                          |
  |                         |------ NOTIFY ----------->|
  |                         |                          |
```

The next-round command starts the same distribution, local training, upload, and aggregation cycle again.

The text sequences intentionally omit conditional branches and show `MN-CSE[i]` and `MN-AE[i]` as one compressed logical `MN-side[i]` participant. They describe the successful end-to-end execution sequence implemented by the prototype.

---

## 6. FL State and Payloads

### Round Command

```json
{
  "type": "fl-command",
  "jobState": "FL_TRAINING",
  "currentRound": 1,
  "maxRounds": 10,
  "globalModelUri": "TinyIoT/IN-AE/cnt-global-model/la",
  "securityMode": "DP",
  "privacyParams": {
    "epsilon": 12.0,
    "delta": 0.00001,
    "max_grad_norm": 1.0
  },
  "timestamp": 0.0
}
```

### Local Update

```json
{
  "type": "fl-update",
  "node": "mn1",
  "round": 1,
  "model_path": "/tmp/fl_models/cache/mn1/local_round1.pt",
  "train_loss": 0.0,
  "val_loss": 0.0,
  "val_auroc": 0.0,
  "num_samples": 0,
  "timestamp": 0.0
}
```

### Global Model Metadata

```json
{
  "type": "global-model",
  "global_round": 1,
  "model_path": "/tmp/fl_models/global/global_round1.pt",
  "model_ready": true,
  "timestamp": 0.0
}
```

### FL States

```text
FL_READY
→ FL_TRAINING
→ FL_AGGREGATING
→ FL_TRAINING for the next round
→ FL_COMPLETED
```

---

## 7. Model and Dataset

| Item | Value |
| --- | --- |
| Dataset | FEMTO-ST PRONOSTIA Bearing Dataset |
| Input | one-channel vibration signal, 2,560 samples |
| Model | Conv1D Autoencoder |
| Learning method | unsupervised learning using normal data |
| Loss | MSE reconstruction loss |
| Local optimizer | Adam |
| Aggregation | weighted FedAvg |
| Global rounds | 10 by default |
| Local epochs | 10 by default |
| Batch size | 32 by default |
| Anomaly score | reconstruction error |
| Security | local-data retention, DP training, ACP isolation, anomalous update filtering |

Raw bearing data remains at each MN-AE. The coordinator receives only model artifact references and training metadata through oneM2M resources.

---

## 8. Project Structure

```text
FL
├── FL-diagrams/
│   ├── fig_initial_setup.puml
│   ├── fig_global_distribution.puml
│   ├── fig_local_training.puml
│   └── fig_global_aggregation.puml
│
├── fl/
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
├── out/FL-diagrams/
├── archive/experiments/
├── fl_bearing_dashboard.html
├── fljob.yaml
├── requirements.txt
├── run_fl.sh
└── clean_fl.sh
```

### Main Files

| File | Role |
| --- | --- |
| `fl/in_ae_standard.py` | IN-AE notification receiver, coordinator, validator, and aggregator |
| `fl/mn_ae_standard.py` | MN-AE notification receiver, local trainer, model cacher, and update uploader |
| `fl/setup_resources_standard.py` | Creates oneM2M resources and ACP mappings |
| `fl/onem2m_utils.py` | Performs oneM2M CREATE, RETRIEVE, UPDATE, DELETE, and subscription operations |
| `fl/aggregator.py` | Performs FedAvg aggregation |
| `fl/model.py` | Defines the Conv1D Autoencoder |
| `fl/dashboard_server.py` | Visualizes FL state by periodically retrieving TinyIoT resources |
| `run_fl.sh` | Runs data preparation, MN-AEs, IN-AE, and dashboard in the correct order |

---

## 9. Environment

The prototype is tested in WSL2 with Python 3.10.

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
http://127.0.0.1:3000/TinyIoT
```

---

## 10. Ports

| Component | Port |
| --- | ---: |
| TinyIoT CSE | 3000 |
| MN-AE-1 notification server | 5001 |
| MN-AE-2 notification server | 5002 |
| MN-AE-3 notification server | 5003 |
| IN-AE notification server | 6000 |
| Dashboard | 7000 |

The node and notification-port pairs must remain consistent:

```text
node_id 0 → mn1 → 5001
node_id 1 → mn2 → 5002
node_id 2 → mn3 → 5003
```

---

## 11. How to Run

### 11.1 Start TinyIoT

```bash
cd ~/projects/tinyIoT/source/server
./server
```

Check the server:

```bash
curl -i http://127.0.0.1:3000
```

### 11.2 Automatic Run

`run_fl.sh` starts the MN-AEs before IN-AE so the command subscriptions exist before the first FL command is created.

```bash
cd ~/projects/federated-learning
source .venv/bin/activate
chmod +x run_fl.sh
./run_fl.sh
```

### 11.3 Manual Run

#### Step 1: Prepare Data and oneM2M Resources

```bash
cd ~/projects/federated-learning
source .venv/bin/activate

python3 fl/prepare_data_femto.py   # may be skipped when prepared data already exists
python3 fl/setup_resources_standard.py
python3 fl/data_generator.py
```

#### Step 2: Start MN-AEs First

Open three separate terminals.

```bash
python3 fl/mn_ae_standard.py 0 5001
```

```bash
python3 fl/mn_ae_standard.py 1 5002
```

```bash
python3 fl/mn_ae_standard.py 2 5003
```

Wait until all terminals report subscription verification and `cnt-fl-control` subscription creation.

#### Step 3: Start IN-AE

```bash
python3 fl/in_ae_standard.py
```

IN-AE waits until `sub_mn1`, `sub_mn2`, and `sub_mn3` are ready before publishing the first training command.

#### Step 4: Start the Dashboard

```bash
python3 fl/dashboard_server.py
```

```text
http://localhost:7000
```

Correct startup order:

```text
TinyIoT
→ data and resource preparation
→ MN-AE-1, MN-AE-2, MN-AE-3
→ command subscriptions ready
→ IN-AE
→ dashboard
```

---

## 12. Expected Runtime Flow

```text
MN-AEs subscribe to cnt-fl-control
IN-AE subscribes to cnt-mn1, cnt-mn2, and cnt-mn3
IN-AE publishes Round 0 global model
IN-AE publishes FL_TRAINING for Round 1
TinyIoT sends command NOTIFY to MN-AEs
MN-AEs retrieve and cache the global model
MN-AEs perform local training
MN-AEs upload local update CINs
TinyIoT sends update NOTIFY to IN-AE
IN-AE collects 3/3 updates
IN-AE publishes FL_AGGREGATING
IN-AE performs anomaly filtering and FedAvg
IN-AE publishes the new global model
IN-AE publishes the next FL_TRAINING command
...
IN-AE publishes FL_COMPLETED after Round 10
```

Example MN-AE log:

```text
[NOTIFY] mn1 command received: FL_TRAINING, round=1
Round 1 command received via NOTIFY
local update uploaded
wait Round 2
```

Example IN-AE log:

```text
[COLLECT/NOTIFY] mn1 Round 1 (1/3)
[COLLECT/NOTIFY] mn2 Round 1 (2/3)
[COLLECT/NOTIFY] mn3 Round 1 (3/3)
All results received by NOTIFY
FedAvg (Round 1)
Global model published Round 1
```

---

## 13. Dashboard Polling

The dashboard periodically performs GET/RETRIEVE requests to show the current FL state.

Example dashboard log:

```text
[Poll] R9/10 FL_TRAINING nodes=['mn1', 'mn2', 'mn3']
[Poll] R9/10 FL_AGGREGATING nodes=['mn1', 'mn2', 'mn3']
[Poll] R10/10 FL_COMPLETED nodes=['mn1', 'mn2', 'mn3']
```

This polling is only for visualization. The IN-AE and MN-AE round workflow remains subscription-notification driven.

---

## 14. Output Paths

```text
/tmp/fl_data/femto
/tmp/fl_models/global
/tmp/fl_models/cache/mn1
/tmp/fl_models/cache/mn2
/tmp/fl_models/cache/mn3
```

Generated global models:

```text
global_round0.pt
global_round1.pt
...
global_round10.pt
```

---

## 15. Configuration

Main FL configuration:

```text
fl/config.py
fljob.yaml
```

Useful environment variables:

```text
FL_COMMAND_NOTIFY_TIMEOUT
FL_COLLECTION_NOTIFY_TIMEOUT
FL_GLOBAL_MODEL_DIR
FL_HIDDEN_TEST_PATH
```

---

## 16. Implementation Status

Implemented and verified:

- oneM2M resource and ACP setup
- MN-AE command subscriptions
- IN-AE local update subscriptions
- subscription verification requests
- command NOTIFY delivery
- local update NOTIFY delivery
- notification body parsing through `rep.m2m:cin.con`
- one-time latest-CIN recovery after notification timeout
- multi-round FL execution through Round 10
- Conv1D Autoencoder local training
- DP configuration and local data retention
- anomalous update filtering
- weighted FedAvg aggregation
- global model redistribution
- dashboard visualization
- final cold-start evaluation

---

## 17. Notes

- Raw training data does not leave the MN-AE side.
- Actual model files are stored in local filesystem paths; oneM2M CINs exchange their metadata and paths.
- The diagram sequence is written without conditional branches and compresses the repeated MN-side AE/CSE flow into one logical participant.
- A single missing notification is recovered with one latest-CIN RETRIEVE; the FL coordinator does not perform continuous polling.
- The TinyIoT notification path must use the path contained in `notificationURI`, such as `/notify`.
