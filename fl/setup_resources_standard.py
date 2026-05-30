"""
표준 oneM2M FL 리소스 구조 생성 (AE 방식) + ACP
ACP 정책:
  acp-fl-control    : IN-AE 전체(63), MN-AE-* 읽기(2=RETRIEVE)
  acp-global-model  : IN-AE 전체(63), MN-AE-* 읽기(2=RETRIEVE)
  acp-local-updates : IN-AE 전체(63), MN-AE-* 생성(1=CREATE)
  acp-dropbox-mnX   : IN-AE 전체(63), MN-AE-X 생성(1=CREATE), 나머지 차단
  acp-sensor-mnX    : MN-AE-X 전체(63), 나머지 차단
  acp-cache-mnX     : MN-AE-X 전체(63), IN-AE 읽기(2=RETRIEVE)

TinyIoT ACOP 비트마스크:
  CREATE=1, RETRIEVE=2, UPDATE=4, DELETE=8, NOTIFY=16, DISCOVERY=32, ALL=63

Usage:
  python3 setup_resources_standard.py           # 리소스 생성만
  python3 setup_resources_standard.py --clean   # DB 초기화 후 리소스 생성
"""
import sys
sys.path.append('/home/eunjin/federated-learning/fl')

import time
import subprocess
import requests as _req
import config
import onem2m_utils as om2m

ORIGIN_ADMIN = "CAdmin"
ORIGIN_IN_AE = "CIN-AE"
ORIGIN_MN_AE = [f"CMN-AE-{i}" for i in range(1, config.NUM_CLIENTS + 1)]

ACOP_ALL       = 63
ACOP_CREATE    = 1
ACOP_RETRIEVE  = 2
ACOP_UPDATE    = 4
ACOP_DELETE    = 8
ACOP_NOTIFY    = 16
ACOP_DISCOVERY = 32


def clean_db():
    print("\n=== DB 초기화 ===")
    # general + ae 테이블 모두 초기화 (ae는 별도 테이블로 관리됨)
    tables = ["general", "ae", "aea", "cnt", "cnta", "cin", "cina", "acp", "sub", "grp"]
    for table in tables:
        sql = f"DELETE FROM {table};"
        result = subprocess.run(
            ["sudo", "-u", "postgres", "psql", "tinyiotdb", "-c", sql],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            rows = result.stdout.strip()
            print(f"  ✓ {table}: {rows}")
        else:
            print(f"  ⚠ {table}: {result.stderr.strip()}")
    time.sleep(1)
    return True


def setup_acp_resources():
    print("\n=== ACP 생성 ===")
    cse = config.CSE_NAME
    acps = {}
    pvs = [{"acor": [ORIGIN_ADMIN], "acop": ACOP_ALL}]

    # acp-fl-control: IN-AE 전체, MN-AE-* 읽기(RETRIEVE=2)
    ri = om2m.create_acp(cse, "acp-fl-control", pv_rules=[
        {"acor": [ORIGIN_ADMIN, ORIGIN_IN_AE], "acop": ACOP_ALL},
        {"acor": ORIGIN_MN_AE,                 "acop": ACOP_RETRIEVE},
    ], pvs_rules=pvs)
    acps["fl-control"] = ri
    print(f"  ✓ acp-fl-control  ri={ri}  (MN: RETRIEVE={ACOP_RETRIEVE})")
    time.sleep(0.2)

    # acp-global-model: IN-AE 전체, MN-AE-* 읽기(RETRIEVE=2)
    ri = om2m.create_acp(cse, "acp-global-model", pv_rules=[
        {"acor": [ORIGIN_ADMIN, ORIGIN_IN_AE], "acop": ACOP_ALL},
        {"acor": ORIGIN_MN_AE,                 "acop": ACOP_RETRIEVE},
    ], pvs_rules=pvs)
    acps["global-model"] = ri
    print(f"  ✓ acp-global-model  ri={ri}  (MN: RETRIEVE={ACOP_RETRIEVE})")
    time.sleep(0.2)

    # acp-local-updates: IN-AE 전체, MN-AE-* 생성(CREATE=1)
    ri = om2m.create_acp(cse, "acp-local-updates", pv_rules=[
        {"acor": [ORIGIN_ADMIN, ORIGIN_IN_AE], "acop": ACOP_ALL},
        {"acor": ORIGIN_MN_AE,                 "acop": ACOP_CREATE},
    ], pvs_rules=pvs)
    acps["local-updates"] = ri
    print(f"  ✓ acp-local-updates  ri={ri}  (MN: CREATE={ACOP_CREATE})")
    time.sleep(0.2)

    for i in range(1, config.NUM_CLIENTS + 1):
        mn_origin = f"CMN-AE-{i}"

        # acp-dropbox-mnX: IN-AE 전체, 해당 MN만 생성(CREATE=1)
        ri = om2m.create_acp(cse, f"acp-dropbox-mn{i}", pv_rules=[
            {"acor": [ORIGIN_ADMIN, ORIGIN_IN_AE], "acop": ACOP_ALL},
            {"acor": [mn_origin],                   "acop": ACOP_CREATE},
        ], pvs_rules=pvs)
        acps[f"dropbox-mn{i}"] = ri
        print(f"  ✓ acp-dropbox-mn{i}  ri={ri}  (MN-{i}: CREATE={ACOP_CREATE})")
        time.sleep(0.2)

        # acp-sensor-mnX: 해당 MN만 전체
        ri = om2m.create_acp(cse, f"acp-sensor-mn{i}", pv_rules=[
            {"acor": [ORIGIN_ADMIN, mn_origin], "acop": ACOP_ALL},
        ], pvs_rules=pvs)
        acps[f"sensor-mn{i}"] = ri
        print(f"  ✓ acp-sensor-mn{i}  ri={ri}  (MN-{i}: ALL={ACOP_ALL})")
        time.sleep(0.2)

        # acp-cache-mnX: 해당 MN 전체, IN-AE 읽기(RETRIEVE=2)
        ri = om2m.create_acp(cse, f"acp-cache-mn{i}", pv_rules=[
            {"acor": [ORIGIN_ADMIN, ORIGIN_IN_AE], "acop": ACOP_RETRIEVE},
            {"acor": [mn_origin],                   "acop": ACOP_ALL},
        ], pvs_rules=pvs)
        acps[f"cache-mn{i}"] = ri
        print(f"  ✓ acp-cache-mn{i}  ri={ri}  (IN-AE: RETRIEVE={ACOP_RETRIEVE}, MN-{i}: ALL={ACOP_ALL})")
        time.sleep(0.2)

    return acps


def setup_in_cse_resources(acps: dict):
    print("\n=== IN-CSE 리소스 생성 ===")
    cse = config.CSE_NAME

    h = config.HEADERS.copy()
    h["Content-Type"] = "application/json;ty=2"
    h["X-M2M-Origin"] = ORIGIN_IN_AE
    payload = {"m2m:ae": {"rn": config.IN_AE_NAME, "api": f"N{config.IN_AE_NAME}", "rr": True, "srv": ["2a"]}}
    r = _req.post(f"{config.BASE_URL}/{cse}", json=payload, headers=h)
    print(f"  ✓ AE: {config.IN_AE_NAME} ({r.status_code})")
    time.sleep(0.3)

    in_ae_path = f"{cse}/{config.IN_AE_NAME}"
    om2m.create_container(in_ae_path, "cnt-fl-control",    mni=50, acpi=[acps["fl-control"]]    if acps.get("fl-control")    else None)
    time.sleep(0.3)
    om2m.create_container(in_ae_path, "cnt-global-model",  mni=50, acpi=[acps["global-model"]]  if acps.get("global-model")  else None)
    time.sleep(0.3)
    om2m.create_container(in_ae_path, "cnt-local-updates", mni=50, acpi=[acps["local-updates"]] if acps.get("local-updates") else None)
    time.sleep(0.3)
    print("  ✓ IN-CSE 구조 완료")


def setup_mn_cse_resources(acps: dict):
    print("\n=== MN-AE 리소스 생성 ===")
    cse = config.CSE_NAME

    for i in range(1, config.NUM_CLIENTS + 1):
        ae_name   = f"MN-AE-{i}"
        print(f"\n[{ae_name}]")

        h = config.HEADERS.copy()
        h["Content-Type"] = "application/json;ty=2"
        h["X-M2M-Origin"] = f"CMN-AE-{i}"
        payload = {"m2m:ae": {"rn": ae_name, "api": f"N{ae_name}", "rr": True, "srv": ["2a"]}}
        r = _req.post(f"{config.BASE_URL}/{cse}", json=payload, headers=h)
        print(f"  ✓ AE: {ae_name} ({r.status_code})")
        time.sleep(0.3)

        mn_ae_path = f"{cse}/{ae_name}"
        om2m.create_container(mn_ae_path, "cnt-sensor-data", mni=25, acpi=[acps[f"sensor-mn{i}"]] if acps.get(f"sensor-mn{i}") else None)
        time.sleep(0.3)
        om2m.create_container(mn_ae_path, "cnt-local-model", mni=5,    acpi=[acps[f"cache-mn{i}"]]  if acps.get(f"cache-mn{i}")  else None)
        time.sleep(0.3)
        print(f"  ✓ {ae_name} 구조 완료")

    print("\n  dropbox per-node 컨테이너 생성...")
    in_updates_path = f"{cse}/{config.IN_AE_NAME}/cnt-local-updates"
    for i in range(1, config.NUM_CLIENTS + 1):
        om2m.create_container(in_updates_path, f"cnt-mn{i}", mni=50, acpi=[acps[f"dropbox-mn{i}"]] if acps.get(f"dropbox-mn{i}") else None)
        time.sleep(0.3)
        print(f"    ✓ {in_updates_path}/cnt-mn{i}")


def verify_resources():
    print("\n=== 리소스 확인 ===")
    cse = config.CSE_NAME
    paths = [
        f"{cse}/{config.IN_AE_NAME}",
        f"{cse}/{config.IN_AE_NAME}/cnt-fl-control",
        f"{cse}/{config.IN_AE_NAME}/cnt-global-model",
        f"{cse}/{config.IN_AE_NAME}/cnt-local-updates",
    ]
    for i in range(1, config.NUM_CLIENTS + 1):
        ae = f"MN-AE-{i}"
        paths += [
            f"{cse}/{ae}",
            f"{cse}/{ae}/cnt-sensor-data",
            f"{cse}/{ae}/cnt-local-model",
            f"{cse}/{config.IN_AE_NAME}/cnt-local-updates/cnt-mn{i}",
        ]

    h = config.HEADERS.copy()
    success = 0
    for path in paths:
        r = _req.get(f"{config.BASE_URL}/{path}", headers=h)
        status = "✓" if r.status_code == 200 else "✗"
        print(f"  {status} {path} ({r.status_code})")
        if r.status_code == 200:
            success += 1
    print(f"\n  총 {success}/{len(paths)} 리소스 확인")
    return success == len(paths)


def verify_acp():
    print("\n=== ACP 연결 확인 ===")
    cse = config.CSE_NAME
    check_paths = [
        f"{cse}/{config.IN_AE_NAME}/cnt-fl-control",
        f"{cse}/{config.IN_AE_NAME}/cnt-global-model",
        f"{cse}/{config.IN_AE_NAME}/cnt-local-updates",
    ]
    for i in range(1, config.NUM_CLIENTS + 1):
        ae = f"MN-AE-{i}"
        check_paths += [
            f"{cse}/{ae}/cnt-sensor-data",
            f"{cse}/{ae}/cnt-local-model",
            f"{cse}/{config.IN_AE_NAME}/cnt-local-updates/cnt-mn{i}",
        ]

    h = config.HEADERS.copy()
    all_ok = True
    for path in check_paths:
        r = _req.get(f"{config.BASE_URL}/{path}", headers=h)
        if r.status_code == 200:
            acpi = r.json().get("m2m:cnt", {}).get("acpi", [])
            short = path.replace(f"{cse}/", "")
            if acpi:
                print(f"  ✓ {short}  acpi={acpi}")
            else:
                print(f"  ⚠ acpi 없음: {short}")
                all_ok = False
        else:
            print(f"  ✗ {path} ({r.status_code})")
            all_ok = False

    if all_ok:
        print("\n  ✓ 모든 컨테이너 ACP 연결 완료!")
    else:
        print("\n  ⚠ 일부 acpi 미연결")
    return all_ok


if __name__ == "__main__":
    do_clean = "--clean" in sys.argv

    print("=" * 60)
    print("표준 oneM2M FL 리소스 구조 생성 (AE + ACP)")
    print("=" * 60)

    if do_clean:
        clean_db()
        print("\n  TinyIoT 재시작 대기 (3s)...")
        time.sleep(3)

    acps = setup_acp_resources()
    setup_in_cse_resources(acps)
    setup_mn_cse_resources(acps)
    ok_res = verify_resources()
    ok_acp = verify_acp()

    print("\n" + "=" * 60)
    if ok_res and ok_acp:
        print("✓ 리소스 생성 + ACP 연결 완료!")
    else:
        print("⚠ 일부 실패 — 위 로그 확인 필요")
    print("=" * 60)