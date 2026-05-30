"""
표준 oneM2M FL 아키텍처 - MN-AE Trainer (Raw Signal AE 버전)
FEMTO PRONOSTIA Bearing 데이터셋용

변경 사항 (이전 통계피처 버전 대비):
- EdgeNode(MLP) → AEEdgeNode(Conv1DAE)
- 입력: raw 시계열 신호 (N, seq_len)
- 손실: MSE reconstruction loss (정상만)
- val_acc → val_auroc (재구성 오차 기반 AUROC)
"""
import sys
sys.path.append('/home/eunjin/federated-learning/fl')

import os
import time
import json
import pickle
import threading
import hashlib
import shutil

import numpy as np
import torch
from flask import Flask, request, make_response

import config
import onem2m_utils as om2m
from edge_node import AEEdgeNode


class MNAETrainer:
    def __init__(self, node_id: int, notification_port: int, inject_anomaly: bool = False):
        self.node_id          = node_id
        self.node_name        = f"mn{node_id + 1}"
        self.notification_port = notification_port
        self.inject_anomaly   = inject_anomaly

        self.ae_name = f"MN-AE-{node_id + 1}"

        # Edge resources
        self.sensor_data_path  = f"{config.CSE_NAME}/{self.ae_name}/cnt-sensor-data"
        self.local_model_cnt   = "cnt-local-model"
        self.local_model_path  = f"{config.CSE_NAME}/{self.ae_name}/{self.local_model_cnt}"

        # Cloud resources
        self.fl_control_path   = f"{config.CSE_NAME}/{config.IN_AE_NAME}/cnt-fl-control"
        self.global_model_path = f"{config.CSE_NAME}/{config.IN_AE_NAME}/cnt-global-model"

        # Drop-box
        self.dropbox_cnt_path  = (
            f"{config.CSE_NAME}/{config.IN_AE_NAME}"
            f"/cnt-local-updates/cnt-{self.node_name}"
        )

        self.edge_node = None

        self.app = Flask(f"MN-AE-{self.node_name}")
        self._setup_flask()

        self.edge_cache_dir = f"/tmp/fl_models/cache/{self.node_name}"
        os.makedirs(self.edge_cache_dir, exist_ok=True)

        self.local_cache_meta_file = os.path.join(
            self.edge_cache_dir, "cache_meta.json"
        )

        self.round_start_stagger_sec = float(self.node_id) * 2.0
        self.upload_stagger_sec      = float(self.node_id) * 3.0

    # ---------------------------
    # Hash
    # ---------------------------
    @staticmethod
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ---------------------------
    # Flask / Notify
    # ---------------------------
    def _setup_flask(self):
        @self.app.route("/notify", methods=["POST"])
        def handle_notification():
            try:
                body = request.get_json(silent=True) or {}
                sgn  = body.get("m2m:sgn", {})
                if sgn.get("vrq") is True:
                    print(f"  [VERIFY] {self.node_name} subscription verification OK")
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp
                resp = make_response("", 200)
                resp.headers["X-M2M-RSC"] = "2000"
                return resp
            except Exception as e:
                print(f"  ✗ notify error: {e}")
                return make_response("", 500)

    def start_server(self):
        def run():
            self.app.run(
                host="0.0.0.0", port=self.notification_port,
                threaded=True, use_reloader=False
            )
        th = threading.Thread(target=run, daemon=True)
        th.start()
        time.sleep(1)
        print(f"  ✓ {self.node_name} Notification server on port {self.notification_port}")

    # ---------------------------
    # oneM2M ops
    # ---------------------------
    def _subscribe_to_fl_control(self):
        sub_name = f"sub_{self.node_name}"
        om2m.delete_subscription(self.fl_control_path, sub_name)
        time.sleep(0.2)
        res = om2m.create_subscription(
            parent_path=self.fl_control_path,
            subscription_name=sub_name,
            notification_uri=(
                f"http://{config.NOTIFY_HOST}:{self.notification_port}/notify"
            ),
            event_types=[3],
            use_nct=None,
        )
        if res:
            print(f"  ✓ {self.node_name} subscribed to cnt-fl-control (net=3)")
        return res

    def _ensure_local_model_container(self):
        parent = f"{config.CSE_NAME}/{self.ae_name}"
        res = om2m.create_container(
            parent_path=parent, container_name=self.local_model_cnt, mni=20
        )
        if res:
            print(f"  ✓ local cache container ready: {self.local_model_path}")

    def _get_current_job_state(self):
        cin = om2m.get_latest_content_instance(self.fl_control_path)
        if cin and "m2m:cin" in cin:
            con = cin["m2m:cin"]["con"]
            return json.loads(con) if isinstance(con, str) else con
        return None

    # ---------------------------
    # Sensor data load (AE 버전)
    # ---------------------------
    def _load_sensor_data_for_round(self, round_num: int, max_attempts: int = 3):
        print(f"\n  [{self.node_name}] load sensor data for round_{round_num} ...")

        # TinyIoT discovery가 라벨 기반 검색을 지원하지 않으므로 /la (최신 CIN) 사용.
        # data_generator.py가 모든 라운드에 동일한 pkl 경로를 publish하므로 문제없음.
        cin = None
        for attempt in range(1, max_attempts + 1):
            cin = om2m.get_latest_content_instance(self.sensor_data_path)
            if cin and "m2m:cin" in cin:
                break
            if attempt < max_attempts:
                wait_sec = 1.5 * attempt
                print(f"    ⚠ no data yet (attempt {attempt}/{max_attempts})"
                      f" -> retry in {wait_sec:.1f}s")
                time.sleep(wait_sec)

        if not cin or "m2m:cin" not in cin:
            print(f"    ✗ no sensor data in {self.sensor_data_path}")
            return None

        con       = cin["m2m:cin"]["con"]
        meta      = json.loads(con) if isinstance(con, str) else con
        data_path = meta.get("data_path")

        if not data_path or not os.path.exists(data_path):
            print(f"    ✗ data_path not found: {data_path}")
            return None

        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)

        # 시나리오: 하루 1라운드, 매 라운드 새로운 데이터 수집
        # train_signals 전체를 GLOBAL_ROUNDS 등분하여 해당 라운드 슬라이스만 사용
        train_sigs = data_dict.get("train_signals", np.array([]))
        total = len(train_sigs)
        if total > 0 and config.GLOBAL_ROUNDS > 1:
            n_per_round = max(1, total // config.GLOBAL_ROUNDS)
            start = (round_num - 1) * n_per_round
            end   = total if round_num >= config.GLOBAL_ROUNDS else start + n_per_round
            data_dict = dict(data_dict)
            data_dict["train_signals"] = train_sigs[start:end]
            print(f"    ✓ round slice [{start}:{end}] ({end - start}/{total} samples)")

        node   = data_dict.get("node", "unknown")
        motors = data_dict.get("motors", [])
        print(f"    ✓ node={node}  motors={motors}  train_n={len(data_dict['train_signals'])}")
        return data_dict

    def _build_ae_node_from_data_dict(self, data_dict: dict) -> AEEdgeNode | None:
        """
        PKL dict → AEEdgeNode
        PKL 형식 (prepare_data_femto.py 출력):
          train_signals, val_signals, val_labels, norm_mean, norm_std, ...
        """
        train_sigs = data_dict.get("train_signals")
        val_sigs   = data_dict.get("val_signals")
        val_labels = data_dict.get("val_labels")

        if train_sigs is None or len(train_sigs) == 0:
            print(f"    ✗ train_signals 없음")
            return None

        if val_sigs is None:
            val_sigs   = np.empty((0, train_sigs.shape[1]), dtype=np.float32)
            val_labels = np.empty((0,), dtype=np.int64)

        # 이상 노드 시뮬레이션 (테스트용)
        if self.inject_anomaly:
            noise_scale = 2.0  # 정규화된 신호 기준이므로 작게
            print(f"  ⚠ [ANOMALY] 센서 데이터 노이즈 주입 (scale={noise_scale})")
            train_sigs = train_sigs + np.random.randn(*train_sigs.shape).astype(
                np.float32
            ) * noise_scale

        train_cfg = getattr(config, "TRAIN_CFG", config.TrainConfig())
        ae_cfg    = getattr(config, "AE_CFG",    config.AEConfig())

        edge_node = AEEdgeNode(
            node_id       = self.node_id,
            train_signals = train_sigs,
            val_signals   = val_sigs,
            val_labels    = val_labels,
            ae_cfg        = ae_cfg,
            train_cfg     = train_cfg,
            device        = "cpu",
        )
        return edge_node

    # ---------------------------
    # Local cache metadata
    # ---------------------------
    def _read_local_cache_meta(self):
        if not os.path.exists(self.local_cache_meta_file):
            return None
        try:
            with open(self.local_cache_meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_local_cache_meta(self, cached_round: int, local_path: str, src_uri: str):
        payload = {
            "type":       "GLOBAL_MODEL_CACHE",
            "round":      cached_round,
            "model_path": local_path,
            "sha256":     (
                self._sha256_file(local_path) if os.path.exists(local_path) else ""
            ),
            "src_uri":    src_uri,
            "timestamp":  time.time(),
        }
        try:
            with open(self.local_cache_meta_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return payload
        except Exception as e:
            print(f"    ⚠ local cache meta write failed: {e}")
            return None

    # ---------------------------
    # Edge cache
    # ---------------------------
    def _read_cache_meta_la(self):
        local = self._read_local_cache_meta()
        if local:
            return local
        cin = om2m.get_latest_content_instance(self.local_model_path)
        if not cin or "m2m:cin" not in cin:
            return None
        con = cin["m2m:cin"]["con"]
        try:
            obj = json.loads(con) if isinstance(con, str) else con
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _write_cache_meta(self, cached_round: int, local_path: str, src_uri: str):
        payload = self._write_local_cache_meta(cached_round, local_path, src_uri)
        if not payload:
            return None
        labels = [self.node_name, f"round_{cached_round}", "type:global-cache"]
        r = om2m.create_content_instance(self.local_model_path, payload, labels=labels)
        if r:
            print(f"    ✓ cached meta -> {self.local_model_path} (round {cached_round})")
        else:
            print(f"    ⚠ cache meta mirror failed (local OK, round {cached_round})")
        return payload

    def _try_load_global_from_cache(self, expected_round: int):
        meta = self._read_cache_meta_la()
        if not meta:
            return None
        try:
            cached_round = int(meta.get("round", -1))
        except Exception:
            return None
        if cached_round != expected_round:
            return None
        path = meta.get("model_path")
        if not path or not os.path.exists(path):
            return None
        sha = meta.get("sha256", "")
        if sha:
            try:
                if sha != self._sha256_file(path):
                    print("    ⚠ cache sha256 mismatch -> ignore cache")
                    return None
            except Exception:
                return None
        return path

    def _get_global_model_path_for_expected_round(self, expected_round: int):
        cin = om2m.get_latest_content_instance(self.global_model_path)
        if not cin or "m2m:cin" not in cin:
            return None
        con        = cin["m2m:cin"]["con"]
        model_data = json.loads(con) if isinstance(con, str) else con
        if int(model_data.get("global_round", 0)) != expected_round:
            return None
        return model_data.get("model_path")

    def _pull_and_cache_global_model(self, expected_round: int):
        src_path = self._get_global_model_path_for_expected_round(expected_round)
        if not src_path or not os.path.exists(src_path):
            return None
        dst_path = os.path.join(
            self.edge_cache_dir, f"global_round{expected_round}.pt"
        )
        try:
            shutil.copyfile(src_path, dst_path)
        except Exception as e:
            print(f"    ⚠ copy failed: {e}")
            return None
        self._write_cache_meta(
            cached_round=expected_round,
            local_path=dst_path,
            src_uri=f"{self.global_model_path}/la",
        )
        return dst_path

    # ---------------------------
    # Drop-box upload
    # ---------------------------
    def _upload_to_dropbox(
        self, round_num, train_loss, val_loss, val_auroc, num_samples, model_path
    ):
        payload = {
            "node":        self.node_name,
            "round":       round_num,
            "model_path":  model_path,
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "val_auroc":   val_auroc,   # AUROC (이전 val_acc 대체)
            "num_samples": num_samples,
            "timestamp":   time.time(),
        }
        labels = [self.node_name, f"round_{round_num}", "type:fl-update"]
        r = om2m.create_content_instance(self.dropbox_cnt_path, payload, labels=labels)
        if r:
            print(f"  ✓ uploaded -> {self.dropbox_cnt_path} (round {round_num})")
        return r

    # ---------------------------
    # Main loop
    # ---------------------------
    def run(self):
        print(f"\n=== {self.node_name.upper()} Trainer start (AE 버전) ===")
        print(f"  AE:           {self.ae_name}")
        print(f"  sensor-data:  {self.sensor_data_path}")
        print(f"  local-model:  {self.local_model_path} (Edge cache)")
        print(f"  fl-control:   {self.fl_control_path}")
        print(f"  global-model: {self.global_model_path}")
        print(f"  drop-box:     {self.dropbox_cnt_path}")
        if self.inject_anomaly:
            print("  ⚠ ANOMALY MODE: 센서 데이터에 노이즈 주입 예정")

        self.start_server()
        self._subscribe_to_fl_control()
        self._ensure_local_model_container()

        for round_num in range(1, config.GLOBAL_ROUNDS + 1):
            print("\n" + "─" * 50)
            print(f"[{self.node_name}] wait Round {round_num}")
            print("─" * 50)

            while True:
                job_state = self._get_current_job_state()
                if job_state is None:
                    time.sleep(2)
                    continue
                state         = job_state.get("jobState", "")
                current_round = int(job_state.get("currentRound", 0))
                if state == "FL_COMPLETED":
                    print(f"  ✓ {self.node_name} done: FL_COMPLETED")
                    return
                if state == "FL_TRAINING" and current_round == round_num:
                    print(f"  ✓ Round {round_num} start! (jobState={state})")
                    break
                time.sleep(2)

            if self.round_start_stagger_sec > 0:
                time.sleep(self.round_start_stagger_sec)

            # ── 데이터 로드 + AEEdgeNode 생성 ──
            data_dict = self._load_sensor_data_for_round(round_num)
            if data_dict is None:
                print(f"  ✗ failed to load data for round {round_num}")
                continue

            self.edge_node = self._build_ae_node_from_data_dict(data_dict)
            if self.edge_node is None:
                print(f"  ✗ failed to build AEEdgeNode for round {round_num}")
                continue

            # ── 이전 글로벌 모델 로드 (round > 1) ──
            if round_num > 1:
                expected = round_num - 1
                cached   = self._try_load_global_from_cache(expected)
                if cached:
                    sd = torch.load(cached, map_location="cpu")
                    self.edge_node.set_state_dict(sd)
                    print(f"    ✓ global model from EDGE cache (round {expected})")
                else:
                    pulled = self._pull_and_cache_global_model(expected)
                    if pulled:
                        sd = torch.load(pulled, map_location="cpu")
                        self.edge_node.set_state_dict(sd)
                        print(f"    ✓ global model from IN-AE (round {expected})")
                    else:
                        print("    ⚠ global model not available (skip load)")

            # ── DP 설정 ──
            security_mode  = job_state.get("securityMode", "DP")
            privacy_params = job_state.get("privacyParams", {})
            dp_epsilon       = float(privacy_params.get(
                "epsilon",       getattr(config, "DP_EPSILON", 8.0)))
            dp_max_grad_norm = float(privacy_params.get(
                "max_grad_norm", getattr(config, "DP_MAX_GRAD_NORM", 1.5)))
            dp_delta         = float(privacy_params.get(
                "delta",         getattr(config, "DP_DELTA", 1e-5)))

            if security_mode == "DP":
                print(f"  [{self.node_name}] Round {round_num} training..."
                      f" DP (ε={dp_epsilon}, C={dp_max_grad_norm})")
            else:
                print(f"  [{self.node_name}] Round {round_num} training..."
                      f" {security_mode} (DP 비활성)")
                dp_epsilon = None

            train_loss, val_loss, val_auroc = self.edge_node.train_local(
                dp_epsilon       = dp_epsilon,
                dp_delta         = dp_delta,
                dp_max_grad_norm = dp_max_grad_norm,
            )

            # ── 모델 저장 ──
            os.makedirs(
                f"/tmp/fl_models/local/{self.node_name}", exist_ok=True
            )
            model_path = (
                f"/tmp/fl_models/local/{self.node_name}/round{round_num}.pt"
            )
            torch.save(self.edge_node.get_state_dict(), model_path)

            print(f"  ✓ train_loss={train_loss:.6f}"
                  f"  val_loss={val_loss:.6f}"
                  f"  val_auroc={val_auroc:.4f}")

            if self.upload_stagger_sec > 0:
                time.sleep(self.upload_stagger_sec)

            self._upload_to_dropbox(
                round_num  = round_num,
                train_loss = train_loss,
                val_loss   = val_loss,
                val_auroc  = val_auroc,
                num_samples = self.edge_node.num_train_samples,
                model_path = model_path,
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 mn_ae_standard.py <node_id> <port> [--anomaly]")
        print("Example: python3 mn_ae_standard.py 0 5001")
        print("Example: python3 mn_ae_standard.py 2 5003 --anomaly")
        raise SystemExit(1)

    node_id        = int(sys.argv[1])
    port           = int(sys.argv[2])
    inject_anomaly = "--anomaly" in sys.argv

    if inject_anomaly:
        print(f"⚠ [ANOMALY MODE] node_id={node_id}")

    MNAETrainer(node_id, port, inject_anomaly=inject_anomaly).run()
