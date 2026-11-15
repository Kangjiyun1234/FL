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
import model_codec
import onem2m_utils as om2m
from edge_node import AEEdgeNode


class MNAETrainer:
    def __init__(self, node_id: int, notification_port: int, inject_anomaly: bool = False):
        self.node_id          = node_id
        self.node_name        = f"mn{node_id + 1}"

        # 이 노드가 '학습에 참여하지 않는 신규 설비'인지.
        # config.COLD_START_NODES 에 이름이 있으면 train_local을 돌리지 않고
        # 글로벌 모델로 추론만 한다.
        self.is_cold_start = self.node_name in getattr(
            config, "COLD_START_NODES", []
        )
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
        self.dropbox_cnt_path = (
            f"{config.CSE_NAME}/{config.IN_AE_NAME}"
            f"/cnt-local-updates/cnt-{self.node_name}"
        )

        # NOTIFY 기반 round 시작
        self.job_event = threading.Event()
        self.job_lock = threading.Lock()
        self.latest_job_command = None
        self.last_command_source = None

        # NOTIFY가 누락됐을 때만 /la를 한 번 확인하기 위한 timeout
        self.command_wait_timeout_sec = float(
            os.getenv("FL_COMMAND_NOTIFY_TIMEOUT", "30")
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
                sgn = body.get("m2m:sgn", {})

                # Subscription verification
                if sgn.get("vrq") is True:
                    print(
                        f"  [VERIFY] {self.node_name} "
                        f"subscription verification OK"
                    )

                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                nev = sgn.get("nev", {}) or {}
                net = nev.get("net", 0)
                net_values = net if isinstance(net, list) else [net]

                try:
                    is_cin_created = 3 in {
                        int(value) for value in net_values
                    }
                except (TypeError, ValueError):
                    is_cin_created = False

                if not is_cin_created:
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                # 정상 경로:
                # round command CIN의 con을 NOTIFY body에서 읽음
                command = self._parse_cin_con_from_notification(sgn)

                # TinyIoT가 rep를 보내지 않은 경우에만
                # NOTIFY를 계기로 최신 CIN을 한 번 조회
                if command is None:
                    command = self._get_current_job_state()

                if command:
                    state = str(command.get("jobState", ""))

                    try:
                        current_round = int(
                            command.get("currentRound", 0)
                        )
                    except (TypeError, ValueError):
                        current_round = 0

                    command_type = (
                        str(command.get("type", ""))
                        .lower()
                        .replace("_", "-")
                    )

                    if command_type and command_type != "fl-command":
                        print(
                            f"  ⚠ {self.node_name}: "
                            f"ignore command type={command_type!r}"
                        )

                    elif state in ("FL_TRAINING", "FL_COMPLETED"):
                        with self.job_lock:
                            self.latest_job_command = command
                            self.last_command_source = "NOTIFY"

                        print(
                            f"  [NOTIFY] {self.node_name} "
                            f"command received: "
                            f"{state}, round={current_round}"
                        )

                        self.job_event.set()

                resp = make_response("", 200)
                resp.headers["X-M2M-RSC"] = "2000"
                return resp

            except Exception as e:
                print(f"  ✗ notify error: {e}")
                return make_response("", 500)

    def _parse_cin_con_from_notification(self, sgn: dict):
        nev = sgn.get("nev", {}) or {}
        rep = nev.get("rep", {}) or {}

        cin = rep.get("m2m:cin")
        if not isinstance(cin, dict):
            return None

        con = cin.get("con")
        if con is None:
            return None

        try:
            data = json.loads(con) if isinstance(con, str) else con
        except (TypeError, json.JSONDecodeError):
            return None

        return data if isinstance(data, dict) else None

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

    def _is_expected_command(self, command: dict | None, round_num: int) -> bool:
        if not isinstance(command, dict):
            return False

        state = str(command.get("jobState", ""))

        try:
            current_round = int(command.get("currentRound", command.get("round", 0)))
        except (TypeError, ValueError):
            current_round = 0

        # 핵심 수정:
        # 새 실행에서 MN-AE가 Round 1을 기다리는 중인데,
        # TinyIoT cnt-fl-control/la에 이전 실행의 FL_COMPLETED가 남아 있으면
        # 그걸 새 실행 완료로 착각하고 종료하는 문제가 생김.
        #
        # 따라서 Round 1 대기 중에는 FL_COMPLETED를 expected command로 인정하지 않음.
        # Round 1은 반드시 FL_TRAINING/currentRound=1로 시작해야 함.
        if state == "FL_COMPLETED":
            if round_num == 1:
                print(
                    f"  ⚠ {self.node_name}: ignore stale FL_COMPLETED "
                    f"while waiting Round 1"
                )
                return False

            # Round 2 이후에는 실제 완료 신호일 수 있으므로 허용.
            # 단, round 정보가 이상하면 무시함.
            return current_round >= round_num

        return (
            state == "FL_TRAINING"
            and current_round == round_num
        )

    def _wait_for_round_command(self, round_num: int):
        """
        NOTIFY를 주 경로로 사용함.
        최초 timeout 때 cnt-fl-control/la를 한 번만 조회하고,
        그 이후에는 다시 polling하지 않고 NOTIFY만 기다림.
        """
        fallback_checked = False

        while True:
            with self.job_lock:
                command = self.latest_job_command
                source = self.last_command_source

            if self._is_expected_command(command, round_num):
                return command, source or "NOTIFY"

            wait_timeout = (
                None
                if fallback_checked
                else self.command_wait_timeout_sec
            )

            notified = self.job_event.wait(timeout=wait_timeout)
            self.job_event.clear()

            if notified:
                continue

            # 최초 timeout에서만 단발성 fallback RETRIEVE 수행
            fallback_checked = True
            print(
                f"  ⚠ {self.node_name}: command NOTIFY timeout; "
                "check cnt-fl-control/la once"
            )

            fallback = self._get_current_job_state()
            if self._is_expected_command(fallback, round_num):
                with self.job_lock:
                    self.latest_job_command = fallback
                    self.last_command_source = "fallback RETRIEVE"
                return fallback, "fallback RETRIEVE"

            print(
                f"  ⚠ {self.node_name}: no matching command in fallback; "
                "continue waiting for NOTIFY"
            )

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

        # Isaac Sim sync demo에서는 race condition을 막기 위해
        # cnt-sensor-data/la가 현재 round와 일치할 때까지 기다림.
        #
        # 기존 문제:
        # IN-AE가 Round N 명령을 publish하는 순간 MN-AE와 Isaac Sim이 동시에 반응함.
        # MN-AE가 Isaac Sim의 sensor-data publish보다 먼저 /la를 읽으면
        # 이전 round metadata를 읽을 수 있음.
        #
        # 수정:
        # meta.round == round_num 또는 meta.currentRound == round_num 일 때만 사용.
        wait_total_sec = float(os.getenv("FL_SENSOR_ROUND_WAIT_SEC", "90"))
        poll_sec = float(os.getenv("FL_SENSOR_ROUND_POLL_SEC", "0.5"))
        allow_legacy_without_round = (
            os.getenv("FL_ALLOW_SENSOR_META_WITHOUT_ROUND", "0") == "1"
        )

        deadline = time.time() + wait_total_sec
        next_log_time = 0.0
        last_seen_round = None
        cin = None
        meta = None

        while True:
            cin = om2m.get_latest_content_instance(self.sensor_data_path)

            if cin and "m2m:cin" in cin:
                con = cin["m2m:cin"].get("con")
                try:
                    candidate = json.loads(con) if isinstance(con, str) else con
                except Exception:
                    candidate = None

                if isinstance(candidate, dict):
                    meta_round_raw = candidate.get(
                        "round",
                        candidate.get("currentRound", None)
                    )

                    if meta_round_raw is None:
                        last_seen_round = "missing"
                        if allow_legacy_without_round:
                            meta = candidate
                            print(
                                "    ⚠ sensor-data has no round field; "
                                "use legacy metadata because "
                                "FL_ALLOW_SENSOR_META_WITHOUT_ROUND=1"
                            )
                            break
                    else:
                        try:
                            meta_round = int(meta_round_raw)
                        except (TypeError, ValueError):
                            meta_round = None

                        last_seen_round = meta_round

                        if meta_round == round_num:
                            meta = candidate
                            print(f"    ✓ matched sensor-data round={meta_round}")
                            break

            now = time.time()
            if now >= deadline:
                print(
                    f"    ✗ sensor-data round mismatch timeout: "
                    f"expected={round_num}, last_seen={last_seen_round}, "
                    f"waited={wait_total_sec:.1f}s"
                )
                return None

            if now >= next_log_time:
                remain = max(0.0, deadline - now)
                print(
                    f"    ⏳ wait Isaac sensor-data round={round_num} "
                    f"(last_seen={last_seen_round}, remain={remain:.1f}s)"
                )
                next_log_time = now + 5.0

            time.sleep(poll_sec)

        data_path = meta.get("data_path")
        if not data_path or not os.path.exists(data_path):
            print(f"    ✗ data_path not found: {data_path}")
            return None

        print(f"    ✓ data_path={data_path}")

        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)

        train_sigs = data_dict.get("train_signals", np.array([]))
        total = len(train_sigs)

        if total > 0:
            # Isaac Sim이 publish한 metadata의 start_idx/end_idx를 우선 사용함.
            # 없으면 기존 방식처럼 GLOBAL_ROUNDS 기준으로 등분함.
            start = None
            end = None

            try:
                meta_start = meta.get("start_idx", None)
                meta_end = meta.get("end_idx", None)
                if meta_start is not None and meta_end is not None:
                    meta_start = int(meta_start)
                    meta_end = int(meta_end)
                    if 0 <= meta_start < meta_end <= total:
                        start = meta_start
                        end = meta_end
            except Exception:
                start = None
                end = None

            if start is None or end is None:
                if config.GLOBAL_ROUNDS > 1:
                    n_per_round = max(1, total // config.GLOBAL_ROUNDS)
                    start = (round_num - 1) * n_per_round
                    end = total if round_num >= config.GLOBAL_ROUNDS else start + n_per_round
                else:
                    start = 0
                    end = total

            data_dict = dict(data_dict)

            round_train_signals = train_sigs[start:end]
            original_round_train_n = len(round_train_signals)

            # Demo fast mode:
            # 각 round의 train window를 최대 FL_DEMO_ROUND_TRAIN_N개만 사용함.
            # 기본값은 50. 빈 값/0/음수면 비활성화.
            #
            # 단순히 앞 50개만 자르면 round 내부의 앞쪽 분포만 보게 되므로,
            # np.linspace로 round 전체 구간에서 균등 샘플링함.
            fast_n_raw = os.getenv("FL_DEMO_ROUND_TRAIN_N", "50").strip()
            fast_n = 0

            if fast_n_raw:
                try:
                    fast_n = int(fast_n_raw)
                except ValueError:
                    fast_n = 0

            if fast_n > 0 and original_round_train_n > fast_n:
                pick = np.linspace(
                    0,
                    original_round_train_n - 1,
                    fast_n,
                    dtype=np.int64,
                )
                round_train_signals = round_train_signals[pick]
                data_dict["round_train_sample_indices"] = pick.astype(np.int64)
                print(
                    f"    ✓ demo fast train_n "
                    f"{original_round_train_n} -> {len(round_train_signals)}"
                )

            data_dict["train_signals"] = round_train_signals
            data_dict["active_round"] = int(round_num)
            data_dict["round_start_idx"] = int(start)
            data_dict["round_end_idx"] = int(end)
            data_dict["round_train_n_original"] = int(original_round_train_n)
            data_dict["round_train_n_used"] = int(len(round_train_signals))
            data_dict["sensor_meta"] = meta

            print(f"    ✓ round slice [{start}:{end}] ({end - start}/{total} samples)")

        node = data_dict.get("node", "unknown")
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

    def _write_cache_meta(
        self,
        cached_round: int,
        local_path: str,
        src_uri: str
    ):
        payload = self._write_local_cache_meta(
            cached_round,
            local_path,
            src_uri
        )

        if not payload:
            return None

        payload["type"] = "global-cache"

        result = om2m.create_content_instance(
            self.local_model_path,
            payload
        )

        if result:
            print(
                f"    ✓ cached meta -> {self.local_model_path} "
                f"(round {cached_round})"
            )
        else:
            print(
                f"    ⚠ cache meta mirror failed "
                f"(local OK, round {cached_round})"
            )
        return payload

    def _try_load_global_from_cache(
        self,
        expected_round: int,
        expected_src_uri: str | None = None
    ):
        meta = self._read_cache_meta_la()
        if not meta:
            return None
        try:
            cached_round = int(meta.get("round", -1))
        except Exception:
            return None
        if cached_round != expected_round:
            return None

        cached_src_uri = str(meta.get("src_uri", ""))
        if expected_src_uri and cached_src_uri != str(expected_src_uri):
            print(
                "    ⚠ cache source URI differs from current command "
                "-> refresh cache"
            )
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

    def _remove_bad_cache(self, model_path: str):
        try:
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
        except OSError:
            pass

        try:
            if os.path.exists(self.local_cache_meta_file):
                os.remove(self.local_cache_meta_file)
        except OSError:
            pass

    def _load_global_state_safely(self, model_path: str) -> bool:
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            self.edge_node.set_state_dict(state_dict)
            return True
        except RuntimeError as exc:
            print(f"    ⚠ incompatible model checkpoint: {model_path}")
            print(f"    ⚠ {exc}")
            self._remove_bad_cache(model_path)
            return False
        except Exception as exc:
            print(f"    ⚠ failed to load model {model_path}: {exc}")
            return False

    def _get_global_model_meta_for_expected_round(
        self,
        expected_round: int,
        global_model_uri: str | None = None
    ):
        """
        글로벌 모델 contentInstance의 con 전체를 돌려준다.
        예전에는 con 안의 model_path만 꺼내 돌려줬지만, 이제는 con에 가중치
        자체(weights_b64)가 실려 오므로 호출측이 둘 중 하나를 고를 수 있어야 한다.
        """
        cin = None

        if global_model_uri:
            resource_path = str(global_model_uri)
            if resource_path.startswith(config.BASE_URL):
                resource_path = resource_path[len(config.BASE_URL):]
            resource_path = resource_path.lstrip("/")
            cin = om2m.get_resource(resource_path)

        # exact URI 조회가 실패한 경우에만 /la를 한 번 확인
        if not cin:
            cin = om2m.get_latest_content_instance(
                self.global_model_path
            )

        if not cin or "m2m:cin" not in cin:
            return None

        con = cin["m2m:cin"].get("con")
        try:
            model_data = (
                json.loads(con)
                if isinstance(con, str)
                else con
            )
        except (TypeError, json.JSONDecodeError):
            return None

        if not isinstance(model_data, dict):
            return None

        try:
            global_round = int(model_data.get("global_round", -1))
        except (TypeError, ValueError):
            return None

        if global_round != expected_round:
            print(
                f"    ⚠ global model round mismatch: "
                f"expected={expected_round}, got={global_round}"
            )
            return None

        return model_data

    def _pull_and_cache_global_model(
        self,
        expected_round: int,
        global_model_uri: str | None = None
    ):
        meta = self._get_global_model_meta_for_expected_round(
            expected_round,
            global_model_uri=global_model_uri
        )

        if not meta:
            return None

        dst_path = os.path.join(
            self.edge_cache_dir,
            f"global_round{expected_round}.pt"
        )

        # con에 가중치가 실려 왔으면 복원해서 캐시에 저장한다.
        # 이 경로만으로 동작하면 MN이 다른 머신에 있어도 FL이 성립한다.
        state_dict = None
        try:
            state_dict = model_codec.decode_state_dict(
                meta,
                fetch_chunk=self._make_chunk_fetcher(meta),
            )
        except Exception as e:
            print(
                f"    ⚠ con 가중치 복원 실패 "
                f"({type(e).__name__}: {e})"
            )

        if state_dict is not None:
            try:
                torch.save(state_dict, dst_path)
                print("    ↓ global weights from con(base64)")
            except Exception as e:
                print(f"    ⚠ 캐시 저장 실패: {e}")
                return None
        else:
            # 호환 모드: 같은 PC에 있을 때만 통하는 예전 방식
            src_path = meta.get("model_path")
            if not src_path or not os.path.exists(src_path):
                print(
                    "    ✗ con에 가중치가 없고 model_path도 접근 불가 "
                    "(MN이 IN과 다른 머신이면 FL_EMBED_WEIGHTS=1 필요)"
                )
                return None
            try:
                shutil.copyfile(src_path, dst_path)
                print("    ↓ global weights via model_path (호환 모드)")
            except Exception as e:
                print(f"    ⚠ copy failed: {e}")
                return None

        self._write_cache_meta(
            cached_round=expected_round,
            local_path=dst_path,
            src_uri=(
                global_model_uri
                or f"{self.global_model_path}/la"
            ),
        )

        return dst_path

    # ---------------------------
    # 가중치 조각 (CSE 페이로드 상한 대응)
    # ---------------------------
    def _ensure_weights_container(self, parent_path: str) -> str:
        """
        조각을 쌓을 자식 컨테이너를 보장한다.
        부모가 아니라 자식에 쌓는 이유: 부모 컨테이너에는 구독이 걸려 있어서
        조각 수십 개가 전부 알림을 발생시키면 수신측이 폭주한다.
        자식에 쌓으면 CREATE_OF_DIRECT_CHILD 조건에 걸리지 않는다.
        """
        om2m.create_container(
            parent_path,
            "cnt-weights",
            mni=1000,
            mbs=60_000_000,
        )
        return f"{parent_path}/cnt-weights"

    def _upload_weight_chunks(self, round_num: int, enc: dict):
        """
        가중치 base64를 CIN 상한에 맞게 조각내어 올리고, 매니페스트를 돌려준다.
        조각 하나라도 실패하면 None을 돌려 호출측이 폴백하도록 한다.
        """
        chunks, manifest = model_codec.split_for_cin(
            enc,
            getattr(config, "FL_MAX_CIN_BYTES", 48 * 1024),
        )

        cnt_path = self._ensure_weights_container(self.dropbox_cnt_path)

        for i, part in enumerate(chunks):
            res = om2m.create_content_instance(
                cnt_path,
                {"i": i, "d": part},
                resource_name=model_codec.chunk_resource_name(round_num, i),
            )
            if not res:
                print(
                    f"    ✗ 가중치 조각 {i + 1}/{len(chunks)} 업로드 실패"
                )
                return None

        manifest["weights_chunk_container"] = cnt_path
        manifest["weights_chunk_round"] = round_num
        return manifest

    def _make_chunk_fetcher(self, meta: dict):
        """
        매니페스트를 보고 조각을 이름으로 직접 RETRIEVE 하는 함수를 만든다.
        discovery를 쓰지 않으므로 조각 순서가 보장된다.
        """
        cnt_path = meta.get("weights_chunk_container")
        if not cnt_path:
            return None

        chunk_round = meta.get(
            "weights_chunk_round",
            meta.get("global_round", meta.get("round")),
        )

        def fetch(index: int):
            res = om2m.get_resource(
                f"{cnt_path}/"
                f"{model_codec.chunk_resource_name(chunk_round, index)}"
            )
            if not res:
                return None
            con = (res.get("m2m:cin") or {}).get("con")
            try:
                obj = json.loads(con) if isinstance(con, str) else con
            except (TypeError, ValueError):
                return None
            return (obj or {}).get("d")

        return fetch

    # ---------------------------
    # Drop-box upload
    # ---------------------------
    def _upload_inference_result(
        self,
        round_num,
        val_loss,
        val_auroc,
    ):
        """
        cold start 노드 전용 업로드.

        학습을 하지 않았으므로 가중치도, model_path도 싣지 않는다.
        type을 fl-inference로 두어 IN-AE가 FedAvg 대상에서 제외하고
        대시보드 표시용으로만 받도록 한다.
        """
        payload = {
            "type": "fl-inference",
            "node": self.node_name,
            "round": round_num,
            "cold_start": True,

            # 학습 미참여를 드러내는 값들
            "train_loss": 0.0,
            "num_samples": 0,

            "accuracy": val_auroc,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "timestamp": time.time(),
        }

        result = om2m.create_content_instance(
            self.dropbox_cnt_path,
            payload
        )

        if result:
            print(
                f"  ✓ inference result -> {self.dropbox_cnt_path} "
                f"(round {round_num}, 가중치 없음)"
            )

        return result

    def _upload_to_dropbox(
        self,
        round_num,
        train_loss,
        val_loss,
        val_auroc,
        num_samples,
        model_path
    ):
        payload = {
            "type": "fl-update",
            "node": self.node_name,
            "round": round_num,

            # 호환용 필드. 같은 PC에서 돌던 기존 방식을 위해 남겨두지만,
            # 집계측은 아래에서 동봉하는 weights_b64를 우선 사용한다.
            "model_path": model_path,
            "weights_uri": model_path,

            "accuracy": val_auroc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "num_samples": num_samples,
            "timestamp": time.time(),
        }

        # 실제 가중치를 con 안에 실어 보낸다.
        # 이것이 있어야 MN을 다른 머신에 올려도 FL이 성립한다.
        if getattr(config, "FL_EMBED_WEIGHTS", False):
            try:
                enc = model_codec.encode_state_dict(
                    torch.load(model_path, map_location="cpu"),
                    fp16=getattr(config, "FL_WEIGHTS_FP16", True),
                )
                limit = getattr(config, "FL_MAX_CIN_BYTES", 48 * 1024)

                if enc["weights_bytes"] <= limit:
                    payload.update(enc)
                    print(
                        f"    ↑ weights in con: {model_codec.describe(enc)}"
                    )
                elif getattr(config, "FL_CHUNK_WEIGHTS", True):
                    manifest = self._upload_weight_chunks(round_num, enc)
                    if manifest:
                        payload.update(manifest)
                        print(
                            f"    ↑ weights in con: "
                            f"{model_codec.describe(manifest)} "
                            f"-> {manifest['weights_chunk_container']}"
                        )
                    else:
                        print(
                            "    ⚠ 조각 업로드 실패 "
                            "-> model_path 방식으로 대체"
                        )
                else:
                    print(
                        f"    ⚠ weights {enc['weights_bytes'] / 1024:.0f}KB "
                        f"> limit {limit / 1024:.0f}KB, 조각 전송 꺼짐 "
                        f"-> model_path 방식으로 대체"
                    )
            except Exception as exc:
                print(
                    f"    ⚠ weight embedding 실패 ({type(exc).__name__}: {exc}) "
                    f"-> model_path 방식으로 대체"
                )

        result = om2m.create_content_instance(
            self.dropbox_cnt_path,
            payload
        )

        if result:
            print(
                f"  ✓ uploaded -> {self.dropbox_cnt_path} "
                f"(round {round_num})"
            )

        return result

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
        if self.is_cold_start:
            print(
                "  ❄ COLD START MODE: 학습에 참여하지 않음 "
                "(글로벌 모델 수신 → 추론만 → 가중치 미전송)"
            )
        if self.inject_anomaly:
            print("  ⚠ ANOMALY MODE: 센서 데이터에 노이즈 주입 예정")

        self.start_server()

        if not self._subscribe_to_fl_control():
            print(f"  ✗ {self.node_name}: failed to subscribe to fl-control")
            return

        self._ensure_local_model_container()

        for round_num in range(1, config.GLOBAL_ROUNDS + 1):
            print("\n" + "─" * 50)
            print(f"[{self.node_name}] wait Round {round_num}")
            print("─" * 50)

            job_state, command_source = self._wait_for_round_command(
                round_num
            )

            state = str(job_state.get("jobState", ""))
            if state == "FL_COMPLETED":
                print(f"  ✓ {self.node_name} done: FL_COMPLETED")
                return

            print(
                f"  ✓ Round {round_num} command received "
                f"via {command_source} (jobState={state})"
            )

            if self.round_start_stagger_sec > 0:
                time.sleep(self.round_start_stagger_sec)

            # 데이터 로드 + AEEdgeNode 생성
            data_dict = self._load_sensor_data_for_round(round_num)
            if data_dict is None:
                print(
                    f"  ✗ failed to load data for round {round_num}; "
                    "stop MN-AE"
                )
                return

            self.edge_node = self._build_ae_node_from_data_dict(data_dict)
            if self.edge_node is None:
                print(
                    f"  ✗ failed to build AEEdgeNode for round {round_num}; "
                    "stop MN-AE"
                )
                return

            # round N은 global model round N-1을 사용
            expected_round = round_num - 1
            global_model_uri = job_state.get("globalModelUri")

            model_loaded = False
            cached_path = self._try_load_global_from_cache(
                expected_round,
                expected_src_uri=global_model_uri
            )

            if cached_path:
                model_loaded = self._load_global_state_safely(
                    cached_path
                )
                if model_loaded:
                    print(
                        f"    ✓ global model from EDGE cache "
                        f"(round {expected_round})"
                    )

            if not model_loaded:
                pulled_path = self._pull_and_cache_global_model(
                    expected_round,
                    global_model_uri=global_model_uri,
                )

                if not pulled_path:
                    print(
                        f"  ✗ global model unavailable "
                        f"for round {expected_round}; stop MN-AE"
                    )
                    return

                model_loaded = self._load_global_state_safely(
                    pulled_path
                )
                if not model_loaded:
                    print(
                        f"  ✗ incompatible global model "
                        f"for round {expected_round}; stop MN-AE"
                    )
                    return

                print(
                    f"    ✓ global model from IN-AE "
                    f"(round {expected_round})"
                )

            # DP 설정
            security_mode = job_state.get("securityMode", "DP")
            privacy_params = job_state.get("privacyParams", {})
            dp_epsilon = float(
                privacy_params.get(
                    "epsilon",
                    getattr(config, "DP_EPSILON", 8.0)
                )
            )
            dp_max_grad_norm = float(
                privacy_params.get(
                    "max_grad_norm",
                    getattr(config, "DP_MAX_GRAD_NORM", 1.5)
                )
            )
            dp_delta = float(
                privacy_params.get(
                    "delta",
                    getattr(config, "DP_DELTA", 1e-5)
                )
            )

            if self.is_cold_start:
                # 학습하지 않는다. 받은 글로벌 모델 그대로 자기 데이터를
                # 평가해서 임계값/AUROC만 구하고, 가중치 없이 올린다.
                print(
                    f"  [{self.node_name}] Round {round_num} "
                    f"COLD START — 학습 없음, 글로벌 모델로 추론만"
                )

                val_loss, val_auroc = self.edge_node._evaluate_val()

                print(
                    f"  ✓ inference only"
                    f"  val_loss={val_loss:.6f}"
                    f"  val_auroc={val_auroc:.4f}"
                )

                if self.upload_stagger_sec > 0:
                    time.sleep(self.upload_stagger_sec)

                if not self._upload_inference_result(
                    round_num=round_num,
                    val_loss=val_loss,
                    val_auroc=val_auroc,
                ):
                    print(
                        f"  ✗ inference result upload failed "
                        f"in round {round_num}; stop MN-AE"
                    )
                    return

                continue

            if security_mode == "DP":
                print(
                    f"  [{self.node_name}] Round {round_num} training..."
                    f" DP (ε={dp_epsilon}, C={dp_max_grad_norm})"
                )
            else:
                print(
                    f"  [{self.node_name}] Round {round_num} training..."
                    f" {security_mode} (DP 비활성)"
                )
                dp_epsilon = None

            train_loss, val_loss, val_auroc = self.edge_node.train_local(
                dp_epsilon=dp_epsilon,
                dp_delta=dp_delta,
                dp_max_grad_norm=dp_max_grad_norm,
            )

            os.makedirs(
                f"/tmp/fl_models/local/{self.node_name}",
                exist_ok=True
            )
            model_path = (
                f"/tmp/fl_models/local/{self.node_name}/"
                f"round{round_num}.pt"
            )
            torch.save(self.edge_node.get_state_dict(), model_path)

            print(
                f"  ✓ train_loss={train_loss:.6f}"
                f"  val_loss={val_loss:.6f}"
                f"  val_auroc={val_auroc:.4f}"
            )

            if self.upload_stagger_sec > 0:
                time.sleep(self.upload_stagger_sec)

            upload_result = self._upload_to_dropbox(
                round_num=round_num,
                train_loss=train_loss,
                val_loss=val_loss,
                val_auroc=val_auroc,
                num_samples=self.edge_node.num_train_samples,
                model_path=model_path,
            )

            if not upload_result:
                print(
                    f"  ✗ local update upload failed in round {round_num}; "
                    "stop MN-AE"
                )
                return


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