"""
표준 oneM2M FL 아키텍처 - IN-AE Coordinator (Drop-box aligned)
- IN-AE/cnt-fl-control : jobState + securityMode + privacyParams publish
- IN-AE/cnt-global-model: global model metadata publish
- IN-AE/cnt-local-updates : Drop-box root
- Loss Z-score 기반 이상치 탐지 (Poisoning Attack 방어)
- ACP 접근 제어 검증 로그
- [추가] FL 완료 후 cold-start hidden test 자동 평가 호출
"""

import sys
sys.path.append('/home/eunjin/federated-learning/fl')

import time
import os
import json
import threading
import subprocess
import requests as _req
import torch
from flask import Flask, request, make_response

import config
import onem2m_utils as om2m
from aggregator import Aggregator


class INAECoordinator:
    def __init__(self, max_rounds=config.GLOBAL_ROUNDS):
        self.max_rounds = max_rounds
        self.current_round = 0

        self.fl_control_path   = f"{config.CSE_NAME}/{config.IN_AE_NAME}/cnt-fl-control"
        self.global_model_path = f"{config.CSE_NAME}/{config.IN_AE_NAME}/cnt-global-model"
        self.dropbox_root      = f"{config.CSE_NAME}/{config.IN_AE_NAME}/cnt-local-updates"

        self.expected_nodes = config.NUM_CLIENTS
        self.node_names = [f"mn{i}" for i in range(1, self.expected_nodes + 1)]
        self.dropbox_by_node = {
            node: f"{self.dropbox_root}/cnt-{node}" for node in self.node_names
        }

        self.aggregator = Aggregator()

        self.app = Flask("IN-AE")
        self.notification_port = 6000
        self._setup_flask()

        self.collected_results = {}
        self.collection_lock = threading.Lock()
        self.collection_event = threading.Event()

        # -----------------------
        # 추가: 글로벌 모델/평가 경로
        # -----------------------
        self.global_model_dir = "/tmp/fl_models/global"
        os.makedirs(self.global_model_dir, exist_ok=True)

        self.latest_global_model_path = None

        # cold-start hidden test 평가용
        self.hidden_test_path = "/tmp/fl_data/femto/mn1.pkl"
        self.eval_script_path = "/home/eunjin/federated-learning/fl/evaluate_mn1_hidden_test.py"

    # -----------------------
    # Flask notify handler
    # -----------------------
    def _setup_flask(self):
        @self.app.route("/notify", methods=["POST"])
        def handle_notification():
            try:
                body = request.get_json(silent=True) or {}
                sgn = body.get("m2m:sgn", {})

                if sgn.get("vrq") is True:
                    print("  [VERIFY] IN-AE subscription verification OK")
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                nev = sgn.get("nev", {})
                sur = sgn.get("sur", "")
                net = nev.get("net", 0)

                print(f"\n  [NOTIFY] sur={sur}, net={net}")

                if net != 3:
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                fired_node = None
                for node, cnt_path in self.dropbox_by_node.items():
                    if cnt_path in sur:
                        fired_node = node
                        break

                if not fired_node:
                    print("  ⚠ cannot map sur -> node (ignore)")
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                round_label = f"round_{self.current_round}"
                cin_obj = om2m.retrieve_first_cin_by_label(self.dropbox_by_node[fired_node], round_label)

                if not cin_obj or "m2m:cin" not in cin_obj:
                    print(f"  ⚠ {fired_node}: CIN not found by label={round_label}")
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                con = cin_obj["m2m:cin"].get("con")
                result_data = json.loads(con) if isinstance(con, str) else con

                cin_round = result_data.get("round")
                if cin_round != self.current_round:
                    print(f"  ⚠ {fired_node}: round mismatch (cin={cin_round}, expected={self.current_round})")
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                with self.collection_lock:
                    self.collected_results[fired_node] = result_data
                    print(f"  [COLLECT] {fired_node} Round {cin_round} ({len(self.collected_results)}/{self.expected_nodes})")

                    if len(self.collected_results) >= self.expected_nodes:
                        print("  ✓ all nodes collected -> set event")
                        self.collection_event.set()

                resp = make_response("", 200)
                resp.headers["X-M2M-RSC"] = "2000"
                return resp

            except Exception as e:
                print(f"  ✗ Notification error: {e}")
                import traceback
                traceback.print_exc()
                return make_response("", 500)

    def start_server(self):
        def run():
            self.app.run(host="0.0.0.0", port=self.notification_port, threaded=True, use_reloader=False)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        time.sleep(1)
        print(f"  ✓ IN-AE Notification server on port {self.notification_port}")

    # -----------------------
    # ACP 검증
    # -----------------------
    def _verify_acp(self):
        """시작 시 ACP 접근 제어 검증 로그 출력"""
        print("\n=== ACP 접근 제어 검증 ===")
        cse = config.CSE_NAME
        cin = {"m2m:cin": {"con": "acp-verify-test"}}

        def check(method, path, origin, data=None, expect_ok=True):
            h = {
                "X-M2M-RI":     "acp-verify",
                "X-M2M-RVI":    "2a",
                "X-M2M-Origin": origin,
                "Content-Type": "application/json;ty=4",
            }
            url = f"{config.BASE_URL}/{path}"
            try:
                r = getattr(_req, method)(url, headers=h, json=data, timeout=5)
                blocked = "no privilege" in r.text or r.status_code == 403
                allowed = r.status_code in [200, 201]
                short_path = path.replace(f"{cse}/", "")

                if expect_ok:
                    ok = allowed
                    label = "✓ 허용" if ok else "✗ 차단됨(버그)"
                else:
                    ok = blocked
                    label = "✓ 차단" if ok else "✗ 허용됨(버그)"

                print(f"    {label}  [{method.upper():4s}] {short_path}  (Origin={origin})")
            except Exception as e:
                print(f"    ✗ 오류: {e}")

        # 1. acpi 필드 확인
        print("\n  [1] acpi 연결 확인")
        check_paths = [
            f"{cse}/{config.IN_AE_NAME}/cnt-fl-control",
            f"{cse}/{config.IN_AE_NAME}/cnt-global-model",
            f"{cse}/{config.IN_AE_NAME}/cnt-local-updates",
        ]
        for i in range(1, config.NUM_CLIENTS + 1):
            check_paths += [
                f"{cse}/MN-AE-{i}/cnt-sensor-data",
                f"{cse}/MN-AE-{i}/cnt-local-model",
                f"{cse}/{config.IN_AE_NAME}/cnt-local-updates/cnt-mn{i}",
            ]
        h_admin = config.HEADERS.copy()
        for path in check_paths:
            r = _req.get(f"{config.BASE_URL}/{path}", headers=h_admin, timeout=5)
            if r.status_code == 200:
                acpi = r.json().get("m2m:cnt", {}).get("acpi", [])
                short = path.replace(f"{cse}/", "")
                status = "✓" if acpi else "⚠ acpi 없음"
                print(f"    {status}  {short}  acpi={acpi}")

        # 2. 차단 케이스
        print("\n  [2] 차단돼야 하는 접근")
        check("get",  f"{cse}/MN-AE-2/cnt-sensor-data",                "CMN-AE-1", expect_ok=False)
        check("get",  f"{cse}/{config.IN_AE_NAME}/cnt-local-updates/cnt-mn2", "CMN-AE-1", data=cin, expect_ok=False)
        check("post", f"{cse}/{config.IN_AE_NAME}/cnt-fl-control",     "CMN-AE-1", data=cin, expect_ok=False)
        check("get",  f"{cse}/MN-AE-1/cnt-local-model",                "CMN-AE-2", expect_ok=False)
        check("get",  f"{cse}/{config.IN_AE_NAME}/cnt-local-updates/cnt-mn1", "CMN-AE-3", data=cin, expect_ok=False)
        check("get",  f"{cse}/{config.IN_AE_NAME}/cnt-local-updates/cnt-mn1", "CMN-AE-1", data=cin, expect_ok=False)

        # 3. 허용 케이스
        print("\n  [3] 허용돼야 하는 접근")
        check("get",  f"{cse}/{config.IN_AE_NAME}/cnt-fl-control",     "CMN-AE-1", expect_ok=True)
        check("get",  f"{cse}/{config.IN_AE_NAME}/cnt-global-model",   "CMN-AE-2", expect_ok=True)
        check("get",  f"{cse}/MN-AE-3/cnt-sensor-data",                "CMN-AE-3", expect_ok=True)
        check("get",  f"{cse}/MN-AE-3/cnt-local-model",                "CMN-AE-3", data=cin, expect_ok=True)
        check("post", f"{cse}/{config.IN_AE_NAME}/cnt-fl-control",     "CIN-AE",   data=cin, expect_ok=True)
        check("get",  f"{cse}/{config.IN_AE_NAME}/cnt-local-updates/cnt-mn1", "CIN-AE", expect_ok=True)
        print()

    # -----------------------
    # oneM2M resource prep
    # -----------------------
    @staticmethod
    def _read_acpi(path: str) -> list:
        res = om2m.get_resource(path)
        if res and "m2m:cnt" in res:
            return res["m2m:cnt"].get("acpi", [])
        return []

    def _prepare_dropbox(self):
        print("\n  Prepare drop-box containers (delete & recreate to clear old CINs)...")
        dropbox_path = f"{config.CSE_NAME}/{config.IN_AE_NAME}/cnt-local-updates"

        # 삭제 전 ACP 정보 저장
        root_acpi = self._read_acpi(dropbox_path)
        node_acpis = {
            node: self._read_acpi(f"{self.dropbox_root}/cnt-{node}")
            for node in self.node_names
        }

        om2m.delete_resource(dropbox_path)
        om2m.create_container(f"{config.CSE_NAME}/{config.IN_AE_NAME}", "cnt-local-updates",
                              mni=5000, mbs=50_000_000)
        if root_acpi:
            om2m.update_acpi(dropbox_path, root_acpi)

        for node in self.node_names:
            parent = self.dropbox_root
            om2m.delete_resource(f"{parent}/cnt-{node}")
            time.sleep(0.5)
            om2m.create_container(parent, f"cnt-{node}", mni=2000, mbs=20_000_000)
            if node_acpis[node]:
                om2m.update_acpi(f"{parent}/cnt-{node}", node_acpis[node])
            print(f"    ✓ {parent}/cnt-{node}")

        # TinyIoT DB가 컨테이너 생성을 완전히 커밋할 시간 확보
        time.sleep(2.0)

    def _subscribe_dropbox(self):
        print("\n  Subscribe drop-box per-node containers (net=3 CIN create)...")
        for node, cnt_path in self.dropbox_by_node.items():
            sub_name = f"sub_in_dropbox_{node}"
            om2m.delete_subscription(cnt_path, sub_name)
            time.sleep(0.2)

            res = om2m.create_subscription(
                parent_path=cnt_path,
                subscription_name=sub_name,
                notification_uri=f"http://{config.NOTIFY_HOST}:{self.notification_port}/notify",
                event_types=[3],
                use_nct=None,
            )
            if res:
                print(f"    ✓ {node} subscribed -> {cnt_path}")

    # -----------------------
    # FL publish
    # -----------------------
    def _publish_job_state(self, state: str, round_num: int):
        data = {
            "type": "FL_COMMAND",
            "jobState": state,
            "currentRound": round_num,
            "maxRounds": self.max_rounds,
            "globalModelUri": f"{self.global_model_path}/la",
            "securityMode": "DP",
            "privacyParams": {
                "epsilon": config.DP_EPSILON,
                "delta": config.DP_DELTA,
                "max_grad_norm": config.DP_MAX_GRAD_NORM,
            },
            "timestamp": time.time(),
        }
        r = om2m.create_content_instance(self.fl_control_path, data, labels=[f"round_{round_num}", state])
        if r:
            print(f"  ✓ jobState: {state} (Round {round_num})")
        return r

    def _publish_global_model(self, round_num: int):
        model_path = f"{self.global_model_dir}/global_round{round_num}.pt"
        data = {
            "global_round": round_num,
            "model_path": model_path,
            "model_ready": True,
            "timestamp": time.time(),
        }
        r = om2m.create_content_instance(self.global_model_path, data, labels=[f"round_{round_num}"])
        if r:
            print(f"  ✓ Global model published Round {round_num}")
        return r

    def _collect_results_polling(self, round_num: int) -> dict:
        print(f"\n  Round {round_num} collect results (Polling fallback)...")
        results = {}
        round_label = f"round_{round_num}"

        for node, cnt_path in self.dropbox_by_node.items():
            cin_obj = om2m.retrieve_first_cin_by_label(cnt_path, round_label)
            if cin_obj and "m2m:cin" in cin_obj:
                con = cin_obj["m2m:cin"]["con"]
                data = json.loads(con) if isinstance(con, str) else con
                if data.get("round") == round_num:
                    results[node] = data
                    print(f"    ✓ {node}: train={data['train_loss']:.4f}, n={data['num_samples']}")
        return results

    # -----------------------
    # Anomaly Detection
    # -----------------------
    def _detect_anomaly(self, results: dict, round_num: int,
                        z_threshold: float = 2.0,
                        abs_threshold: float = 1.5,
                        bypass_rounds: int = 3) -> list:
        import math

        nodes = list(results.keys())
        if len(nodes) < 2:
            print("  [Anomaly Detection] 노드 수 < 2, 스킵")
            return []

        if round_num <= bypass_rounds:
            print(f"  [Anomaly Detection] Round {round_num} <= {bypass_rounds} (early bypass), 스킵")
            return []

        train_losses = {n: results[n]["train_loss"] for n in nodes}
        val_losses   = {n: results[n]["val_loss"]   for n in nodes}

        def z_scores(loss_dict):
            vals = list(loss_dict.values())
            mean = sum(vals) / len(vals)
            std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            if std < 1e-8:
                return {n: 0.0 for n in loss_dict}
            return {n: abs(v - mean) / std for n, v in loss_dict.items()}

        train_z = z_scores(train_losses)
        val_z   = z_scores(val_losses)

        print(f"\n  [Anomaly Detection] Loss Z-score (z>{z_threshold} AND loss>{abs_threshold}):")
        anomalies = []
        for n in nodes:
            tz = train_z[n]
            vz = val_z[n]
            tl = train_losses[n]
            vl = val_losses[n]
            is_anomaly = ((tz > z_threshold and tl > abs_threshold) or
                          (vz > z_threshold and vl > abs_threshold))
            status = "✗ ANOMALY" if is_anomaly else "✓ normal"
            print(f"    {n}: train_loss={tl:.4f} (z={tz:.2f}), "
                  f"val_loss={vl:.4f} (z={vz:.2f})  {status}")
            if is_anomaly:
                anomalies.append(n)

        return anomalies

    # -----------------------
    # Aggregation
    # -----------------------
    def _aggregate(self, results: dict, round_num: int):
        print(f"\n  FedAvg (Round {round_num})...")

        state_dicts = {}
        num_samples = {}
        for node, data in results.items():
            sd = torch.load(data["model_path"], map_location="cpu")
            state_dicts[node] = sd
            num_samples[node] = data["num_samples"]
            print(f"    ← {node} (n={data['num_samples']})")

        anomalies = self._detect_anomaly(results, round_num)

        if anomalies:
            print(f"  ⚠ 이상 노드 제외: {anomalies}")
            for n in anomalies:
                state_dicts.pop(n)
                num_samples.pop(n)

        if not state_dicts:
            print("  ✗ 정상 노드 없음, aggregation 스킵")
            return 0.0, None

        client_states = [(sd, num_samples[n]) for n, sd in state_dicts.items()]
        global_state = self.aggregator.aggregate(client_states)

        out_path = f"{self.global_model_dir}/global_round{round_num}.pt"
        torch.save(global_state, out_path)
        self.latest_global_model_path = out_path

        normal_nodes = [n for n in results if n not in anomalies]
        total = sum(results[n]["num_samples"] for n in normal_nodes)
        avg_loss = sum(results[n]["train_loss"] * results[n]["num_samples"]
                       for n in normal_nodes) / max(total, 1)

        print(f"  ✓ Aggregation done! Global Train Loss: {avg_loss:.4f}, Total: {total}")
        print(f"  ✓ Global state saved: {out_path}")
        if anomalies:
            print(f"  ⚠ (이상 노드 {anomalies} 제외됨)")
        return avg_loss, out_path

    # -----------------------
    # Cold-start evaluation
    # -----------------------
    def _run_cold_start_eval(self):
        print("\n=== Cold-start hidden test evaluation ===")

        if not self.latest_global_model_path or not os.path.exists(self.latest_global_model_path):
            print("  ⚠ final global model not found -> skip evaluation")
            return

        if not os.path.exists(self.hidden_test_path):
            print(f"  ⚠ hidden test file not found -> {self.hidden_test_path}")
            print("  ⚠ skip cold-start evaluation")
            return

        if not os.path.exists(self.eval_script_path):
            print(f"  ⚠ evaluation script not found -> {self.eval_script_path}")
            print("  ⚠ skip cold-start evaluation")
            return

        cmd = [
            sys.executable,
            self.eval_script_path,
            "--model", self.latest_global_model_path,
            "--hidden-test", self.hidden_test_path,
        ]

        print("  ▶ Run:", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)

            if proc.stdout:
                print(proc.stdout.strip())

            if proc.stderr:
                print(proc.stderr.strip())

            if proc.returncode == 0:
                print("  ✓ cold-start hidden test evaluation done")
            else:
                print(f"  ✗ cold-start evaluation failed (returncode={proc.returncode})")

        except Exception as e:
            print(f"  ✗ evaluation invoke error: {e}")

    # -----------------------
    # Main loop
    # -----------------------
    def run(self):
        print("\n=== IN-AE Coordinator start ===")
        print(f"  fl-control:   {self.fl_control_path}")
        print(f"  global-model: {self.global_model_path}")
        print(f"  drop-box:     {self.dropbox_root}")

        self.start_server()
        self._prepare_dropbox()
        self._subscribe_dropbox()

        # ACP 검증
        self._verify_acp()

        self._publish_job_state("FL_READY", 0)
        self._publish_global_model(0)
        print("  ✓ Initial global model ready (Round 0)")

        for r in range(1, self.max_rounds + 1):
            self.current_round = r

            print("\n" + "=" * 60)
            print(f"GLOBAL ROUND {r}/{self.max_rounds}")
            print("=" * 60)

            self.collection_event.clear()
            with self.collection_lock:
                self.collected_results = {}

            self._publish_job_state("FL_TRAINING", r)

            print("  ⏳ Wait local updates (timeout: 600s)...")
            ok = self.collection_event.wait(timeout=600)

            if ok:
                print("  ✓ All results received via Notification -> aggregate now")
                with self.collection_lock:
                    results = dict(self.collected_results)
            else:
                print("  ⚠ Timeout -> polling fallback")
                results = self._collect_results_polling(r)

            if not results:
                print("  ✗ No results, skip round")
                continue

            self._publish_job_state("FL_AGGREGATING", r)
            _, final_path = self._aggregate(results, r)
            self._publish_global_model(r)

        print("\nFederated learning completed!")
        self._publish_job_state("FL_COMPLETED", self.max_rounds)

        # -----------------------
        # 추가: FL 완료 후 cold-start hidden test 자동 평가
        # -----------------------
        self._run_cold_start_eval()


if __name__ == "__main__":
    coordinator = INAECoordinator(max_rounds=config.GLOBAL_ROUNDS)
    coordinator.run()