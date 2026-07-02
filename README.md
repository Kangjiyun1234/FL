# FL: oneM2M Resource-based Federated Learning Prototype

## 1. Overview

This repository contains a prototype implementation of a federated learning workflow using oneM2M resources.

The project is aligned with the ongoing oneM2M work on **TR-0084: Developer guide; Use of oneM2M resources to support Federated Learning**.

The goal of this prototype is to show how native oneM2M resources such as **Application Entity (AE)**, **`<container>`**, **`<contentInstance>`**, and **`<subscription>`** can be used to support a complete federated learning lifecycle.

Instead of relying on a separate machine learning middleware for coordination, this prototype uses oneM2M-style resource operations to manage:

```text
- federated learning round control
- global model metadata distribution
- local training result upload
- local update collection
- global aggregation
- dashboard visualization
- subscription notification-based update detection
```

The current implementation uses **TinyIoT** as the oneM2M CSE layer.

---

## 2. Related oneM2M Work Item

### Main Reference

| Item               | Description                                                                     |
| ------------------ | ------------------------------------------------------------------------------- |
| **oneM2M TR-0084** | Developer guide; Use of oneM2M resources to support Federated Learning          |
| **Contribution**   | TDE-2026-0022 Federated Learning Resources and Procedures                       |
| **Purpose**        | Define how oneM2M resources can be used to support federated learning workflows |

This repository is a prototype implementation related to the TR-0084 direction.

The prototype focuses on mapping the federated learning lifecycle onto oneM2M resource operations and resource structures.

---

## 3. Concept

Federated learning allows multiple distributed nodes to train models locally without sending raw data to a central server.

In this prototype:

```text
- IN-AE acts as the global FL coordinator and aggregator.
- MN-AEs act as local training clients.
- TinyIoT CSE stores oneM2M resources and metadata.
- Raw local data stays at each MN-AE side.
- Only local update metadata and model artifact paths are exchanged.
```

The basic FL lifecycle is:

```text
1. IN-AE initializes the global model.
2. IN-AE publishes global model metadata to a oneM2M resource.
3. Each MN-AE retrieves the global model metadata.
4. Each MN-AE performs local training.
5. Each MN-AE uploads local update metadata.
6. IN-AE collects local updates.
7. IN-AE aggregates local updates into a new global model.
8. IN-AE publishes the updated global model.
9. The next FL round starts.
```

---

## 4. oneM2M Resource Mapping

The prototype uses the following oneM2M-style resource structure.

```text
TinyIoT
└── IN-AE
    ├── cnt-fl-control
    │   └── FL command and round state metadata
    │
    ├── cnt-global-model
    │   └── Global model metadata
    │
    └── cnt-local-updates
        ├── cnt-mn1
        │   └── Local update metadata from MN-AE-1
        ├── cnt-mn2
        │   └── Local update metadata from MN-AE-2
        └── cnt-mn3
            └── Local update metadata from MN-AE-3
```

### Resource Roles

| Resource                        | Role                                                             |
| ------------------------------- | ---------------------------------------------------------------- |
| `IN-AE`                         | Application Entity for FL coordination                           |
| `cnt-fl-control`                | Stores FL round state and control messages                       |
| `cnt-global-model`              | Stores global model metadata                                     |
| `cnt-local-updates`             | Parent container for local update results                        |
| `cnt-mn1`, `cnt-mn2`, `cnt-mn3` | Per-node local update containers                                 |
| `contentInstance`               | Stores model metadata, local update metadata, and training state |

---

## 5. FL Entity Mapping

| FL Concept                    | oneM2M Mapping                                        | Implementation               |
| ----------------------------- | ----------------------------------------------------- | ---------------------------- |
| FL Server / Aggregator        | IN-AE                                                 | `fl/in_ae_standard.py`       |
| FL Client                     | MN-AE                                                 | `fl/mn_ae_standard.py`       |
| Global model metadata         | `<contentInstance>` under `cnt-global-model`          | JSON metadata                |
| Local update metadata         | `<contentInstance>` under `cnt-local-updates/cnt-mnX` | JSON metadata                |
| Round state                   | `<contentInstance>` under `cnt-fl-control`            | FL status message            |
| Event-driven update detection | `<subscription>` + notification                       | Planned / partially verified |

---

## 6. Current Implementation Status

Implemented:

```text
- oneM2M resource setup for FL
- IN-AE global model initialization
- global model metadata publishing
- MN-AE local training
- local update metadata upload
- IN-AE local update collection
- FedAvg-style aggregation
- dashboard visualization
- TinyIoT subscription notification path verification
```

Current focus:

```text
- Replace or supplement polling-based local update collection with oneM2M subscription notification.
```

---

## 7. Subscription Notification Support

TR-0084 focuses on using oneM2M resources to support federated learning procedures.
For FL update collection, subscription notification is useful because IN-AE does not need to continuously poll local update containers.

The intended notification-based flow is:

```text
1. IN-AE creates subscriptions under local update containers.
2. Each subscription uses enc.net = [3].
3. MN-AE uploads a local update as a contentInstance.
4. TinyIoT detects the contentInstance creation event.
5. TinyIoT sends a notification to IN-AE.
6. IN-AE parses the notification and stores the local update by round and node.
7. When all expected local updates arrive, IN-AE performs aggregation.
```

Target subscription locations:

```text
TinyIoT/IN-AE/cnt-local-updates/cnt-mn1
TinyIoT/IN-AE/cnt-local-updates/cnt-mn2
TinyIoT/IN-AE/cnt-local-updates/cnt-mn3
```

Example subscription:

```json
{
  "m2m:sub": {
    "rn": "sub-fl-mn1",
    "nu": ["http://127.0.0.1:9100/fl/notify"],
    "enc": {
      "net": [3]
    }
  }
}
```

Here, `enc.net = [3]` means that a notification is triggered when a direct child resource is created under the subscribed resource. In this prototype, that direct child resource is a local update `contentInstance`.

---

## 8. TinyIoT Notification Verification

Before applying notification to the FL workflow, TinyIoT subscription notification behavior was tested separately.

### Test Receiver

```text
http://127.0.0.1:9000/notify
```

### Test Subscription

```json
{
  "m2m:sub": {
    "rn": "sub-check-001",
    "nu": ["http://127.0.0.1:9000/notify"],
    "enc": {
      "net": [3]
    }
  }
}
```

### Verification Request

When a subscription is created, TinyIoT sends a verification request to confirm that the notification URI is reachable.

Expected receiver output:

```text
path: /notify
>>> SUBSCRIPTION VERIFICATION REQUEST
```

Expected body:

```json
{
  "m2m:sgn": {
    "vrq": true
  }
}
```

### Normal Notification

After the subscription is created, a normal notification is sent when a `contentInstance` is created under the subscribed container.

Test contentInstance:

```json
{
  "m2m:cin": {
    "con": "hello fixed notification"
  }
}
```

Expected receiver output:

```text
path: /notify
>>> NORMAL NOTIFICATION
```

Expected body contains:

```json
{
  "m2m:sgn": {
    "nev": {
      "net": 3,
      "rep": {
        "m2m:cin": {
          "con": "hello fixed notification"
        }
      }
    }
  }
}
```

---

## 9. TinyIoT Notification Path Fix

During testing, TinyIoT initially sent the subscription verification request to the wrong HTTP path.

Expected path:

```text
POST /notify
```

Actual path before fix:

```text
POST /TinyIoT/IN-AE/cnt-noti-check
```

The issue occurred because TinyIoT parsed the notification URI correctly but used `o2pt->to` as the HTTP request URI inside `http_notify()`.

For subscription verification:

```text
o2pt->to   = TinyIoT/IN-AE/cnt-noti-check
nt->target = /notify
```

The fix was to use `nt->target` first when setting the HTTP request URI.

```c
if (nt && strlen(nt->target) > 0)
{
    req->uri = strdup(nt->target);
}
else
{
    req->uri = strdup(o2pt->to);
}
```

After the fix:

```text
Subscription verification request → POST /notify
Normal notification              → POST /notify
```

This confirms that TinyIoT can now send HTTP notifications to the path specified by `notificationURI`.

---

## 10. Project Structure

```text
FL
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
├── out/
├── fl_bearing_dashboard.html
├── fljob.yaml
├── main.py
├── requirements.txt
├── run_fl.sh
├── clean_fl.sh
├── fig_initial_setup.puml
├── fig_global_distribution.puml
├── fig_local_training.puml
└── fig_global_aggregation.puml
```

---

## 11. Environment

This project is tested in a WSL environment.

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If dependency installation fails, install the main packages manually:

```bash
pip install torch numpy pandas pyyaml flask scikit-learn requests
```

---

## 12. TinyIoT Requirement

TinyIoT must be running before executing the FL prototype.

Example:

```bash
cd ~/projects/tinyIoT/source/server
./server
```

Expected TinyIoT base URL:

```text
http://127.0.0.1:3000/TinyIoT
```

---

## 13. How to Run

### 1. Start TinyIoT

```bash
cd ~/projects/tinyIoT/source/server
./server
```

### 2. Prepare Python Environment

```bash
cd ~/projects/FL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Full FL Pipeline

```bash
chmod +x run_fl.sh
./run_fl.sh
```

The script performs the full FL workflow, including resource setup, local training, global aggregation, and dashboard execution.

---

## 14. Manual Run

If the shell script is not used, run each step manually.

### Setup resources and data

```bash
python3 fl/setup_resources_standard.py
python3 fl/data_generator.py
```

### Start IN-AE

```bash
python3 fl/in_ae_standard.py
```

### Start MN-AEs

Open separate terminals.

```bash
python3 fl/mn_ae_standard.py 0 5001
python3 fl/mn_ae_standard.py 1 5002
python3 fl/mn_ae_standard.py 2 5003
```

### Start Dashboard

```bash
python3 fl/dashboard_server.py
```

Dashboard URL:

```text
http://localhost:7000
```

---

## 15. Output Paths

The prototype stores model and data artifacts in local temporary paths.

```text
/tmp/fl_models/global
/tmp/fl_data/femto
```

Example generated model files:

```text
global_round0.pt
global_round1.pt
global_round2.pt
...
```

---

## 16. Configuration

Main configuration file:

```text
fljob.yaml
```

Example oneM2M configuration:

```yaml
onem2m:
  enabled: true
  base_url: "http://127.0.0.1:3000/TinyIoT"
  origin: "CAdmin"
  rvi: "2a"
  in_ae: "IN-AE"
```

---

## 17. Planned Work

Next steps:

```text
- Add IN-AE notification receiver endpoint.
- Create subscriptions under cnt-mn1, cnt-mn2, and cnt-mn3.
- Receive local update notifications instead of polling.
- Parse rep.m2m:cin.con from notification body.
- Store received updates by round and node ID.
- Trigger aggregation when all expected node updates arrive.
- Keep polling as a fallback mode.
```

Proposed IN-AE notification endpoint:

```text
http://127.0.0.1:9100/fl/notify
```

---

## 18. Notes

```text
- Raw training data remains local to each MN-AE.
- oneM2M resources are used for metadata exchange and FL procedure coordination.
- contentInstance.con should be sent as a string for TinyIoT compatibility.
- Subscription notification has been verified with both verification request and normal notification.
- The current implementation can be extended to follow the TR-0084 developer guide direction more closely.
```

---

## 19. Commit Example

```bash
git add README.md
git commit -m "docs: add README for TR-0084 FL prototype"
git push
```
