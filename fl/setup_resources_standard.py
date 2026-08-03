"""
표준 oneM2M FL 리소스 구조 생성 (AE 방식) + ACP

ACP 정책:
  acp-fl-control
    - IN-AE 전체 권한
    - MN-AE-* RETRIEVE

  acp-global-model
    - IN-AE 전체 권한
    - MN-AE-* RETRIEVE

  acp-local-updates
    - IN-AE 전체 권한
    - MN-AE-* CREATE

  acp-dropbox-mnX
    - IN-AE 전체 권한
    - 해당 MN-AE-X만 CREATE

  acp-sensor-mnX
    - 해당 MN-AE-X 전체 권한

  acp-cache-mnX
    - 해당 MN-AE-X 전체 권한
    - IN-AE RETRIEVE

TinyIoT ACOP 비트마스크:
  CREATE    = 1
  RETRIEVE  = 2
  UPDATE    = 4
  DELETE    = 8
  NOTIFY    = 16
  DISCOVERY = 32
  ALL       = 63

사용 방법:
  python3 fl/setup_resources_standard.py
  python3 fl/setup_resources_standard.py --clean
"""

import os
import subprocess
import sys
import time

import requests as requests

import config
import onem2m_utils as om2m


ORIGIN_ADMIN = "CAdmin"
ORIGIN_IN_AE = "CIN-AE"

ORIGIN_MN_AE = [
    f"CMN-AE-{i}"
    for i in range(1, config.NUM_CLIENTS + 1)
]

ACOP_CREATE = 1
ACOP_RETRIEVE = 2
ACOP_UPDATE = 4
ACOP_DELETE = 8
ACOP_NOTIFY = 16
ACOP_DISCOVERY = 32
ACOP_ALL = 63


# ════════════════════════════════════════════════════════
# DB 초기화
# ════════════════════════════════════════════════════════

def clean_db() -> bool:
    """
    TinyIoT와 oneM2M Design Tool이 사용하는 PostgreSQL 테이블을 초기화한다.

    실제 TinyIoT 설정:
      DB 종류: PostgreSQL
      DB 호스트: localhost
      DB 포트: 5433
      DB 이름: tinydb
      TinyIoT DB 사용자: tinyuser

    DB 초기화 명령은 PostgreSQL 관리자 계정인 postgres로 실행한다.

    환경변수로 변경 가능:
      TINYIOT_DB_NAME
      TINYIOT_DB_PORT
      TINYIOT_DB_ADMIN_USER
    """

    print("\n=== TinyIoT DB 초기화 ===")

    db_name = os.getenv(
        "TINYIOT_DB_NAME",
        "tinydb",
    )

    db_port = os.getenv(
        "TINYIOT_DB_PORT",
        "5433",
    )

    db_admin_user = os.getenv(
        "TINYIOT_DB_ADMIN_USER",
        "postgres",
    )

    tables = [
        "general",
        "ae",
        "aea",
        "cnt",
        "cnta",
        "cin",
        "cina",
        "acp",
        "sub",
        "grp",
    ]

    table_sql = ", ".join(tables)

    print(f"  DB 이름: {db_name}")
    print(f"  DB 포트: {db_port}")
    print(f"  관리자 사용자: {db_admin_user}")

    # 실제 DB에 접속 가능한지 먼저 확인한다.
    check_command = [
        "sudo",
        "-u",
        db_admin_user,
        "psql",
        "-p",
        db_port,
        "-v",
        "ON_ERROR_STOP=1",
        "-d",
        db_name,
        "-tAc",
        "SELECT current_database();",
    ]

    check_result = subprocess.run(
        check_command,
        capture_output=True,
        text=True,
        check=False,
    )

    if check_result.returncode != 0:
        error_message = (
            check_result.stderr.strip()
            or check_result.stdout.strip()
            or "알 수 없는 PostgreSQL 연결 오류"
        )

        raise RuntimeError(
            "TinyIoT PostgreSQL DB에 연결하지 못했습니다.\n"
            f"DB: {db_name}\n"
            f"포트: {db_port}\n"
            f"관리자 사용자: {db_admin_user}\n"
            f"오류: {error_message}\n"
            "TinyIoT config.h의 PG_DBNAME과 PG_PORT, "
            "그리고 PostgreSQL 실행 상태를 확인하세요."
        )

    connected_database = check_result.stdout.strip()

    if connected_database != db_name:
        raise RuntimeError(
            "예상한 DB와 실제 접속된 DB가 다릅니다.\n"
            f"예상 DB: {db_name}\n"
            f"실제 DB: {connected_database}"
        )

    sql = (
        f"TRUNCATE TABLE {table_sql} "
        "RESTART IDENTITY CASCADE;"
    )

    command = [
        "sudo",
        "-u",
        db_admin_user,
        "psql",
        "-p",
        db_port,
        "-v",
        "ON_ERROR_STOP=1",
        "-d",
        db_name,
        "-c",
        sql,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        error_message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "알 수 없는 PostgreSQL 오류"
        )

        raise RuntimeError(
            "TinyIoT DB 초기화에 실패했습니다.\n"
            f"DB: {db_name}\n"
            f"포트: {db_port}\n"
            f"관리자 사용자: {db_admin_user}\n"
            f"오류: {error_message}\n"
            "테이블 이름과 PostgreSQL 권한을 확인하세요."
        )

    print(f"  ✓ DB 연결 확인: {connected_database}")
    print(f"  ✓ 초기화 테이블: {table_sql}")
    print("  ✓ TRUNCATE + RESTART IDENTITY + CASCADE 완료")

    time.sleep(1)

    return True


# ════════════════════════════════════════════════════════
# ACP 생성
# ════════════════════════════════════════════════════════

def setup_acp_resources() -> dict:
    print("\n=== ACP 생성 ===")

    cse = config.CSE_NAME
    acps: dict[str, str | None] = {}

    pvs_rules = [
        {
            "acor": [ORIGIN_ADMIN],
            "acop": ACOP_ALL,
        }
    ]

    # ────────────────────────────────────────────────────
    # acp-fl-control
    # ────────────────────────────────────────────────────

    resource_id = om2m.create_acp(
        cse,
        "acp-fl-control",
        pv_rules=[
            {
                "acor": [
                    ORIGIN_ADMIN,
                    ORIGIN_IN_AE,
                ],
                "acop": ACOP_ALL,
            },
            {
                "acor": ORIGIN_MN_AE,
                "acop": ACOP_RETRIEVE,
            },
        ],
        pvs_rules=pvs_rules,
    )

    acps["fl-control"] = resource_id

    print(
        "  ✓ acp-fl-control "
        f"ri={resource_id} "
        f"(MN: RETRIEVE={ACOP_RETRIEVE})"
    )

    time.sleep(0.2)

    # ────────────────────────────────────────────────────
    # acp-global-model
    # ────────────────────────────────────────────────────

    resource_id = om2m.create_acp(
        cse,
        "acp-global-model",
        pv_rules=[
            {
                "acor": [
                    ORIGIN_ADMIN,
                    ORIGIN_IN_AE,
                ],
                "acop": ACOP_ALL,
            },
            {
                "acor": ORIGIN_MN_AE,
                "acop": ACOP_RETRIEVE,
            },
        ],
        pvs_rules=pvs_rules,
    )

    acps["global-model"] = resource_id

    print(
        "  ✓ acp-global-model "
        f"ri={resource_id} "
        f"(MN: RETRIEVE={ACOP_RETRIEVE})"
    )

    time.sleep(0.2)

    # ────────────────────────────────────────────────────
    # acp-local-updates
    # ────────────────────────────────────────────────────

    resource_id = om2m.create_acp(
        cse,
        "acp-local-updates",
        pv_rules=[
            {
                "acor": [
                    ORIGIN_ADMIN,
                    ORIGIN_IN_AE,
                ],
                "acop": ACOP_ALL,
            },
            {
                "acor": ORIGIN_MN_AE,
                "acop": ACOP_CREATE,
            },
        ],
        pvs_rules=pvs_rules,
    )

    acps["local-updates"] = resource_id

    print(
        "  ✓ acp-local-updates "
        f"ri={resource_id} "
        f"(MN: CREATE={ACOP_CREATE})"
    )

    time.sleep(0.2)

    # ────────────────────────────────────────────────────
    # MN별 ACP
    # ────────────────────────────────────────────────────

    for index in range(1, config.NUM_CLIENTS + 1):
        mn_origin = f"CMN-AE-{index}"

        # acp-dropbox-mnX
        resource_id = om2m.create_acp(
            cse,
            f"acp-dropbox-mn{index}",
            pv_rules=[
                {
                    "acor": [
                        ORIGIN_ADMIN,
                        ORIGIN_IN_AE,
                    ],
                    "acop": ACOP_ALL,
                },
                {
                    "acor": [mn_origin],
                    "acop": ACOP_CREATE,
                },
            ],
            pvs_rules=pvs_rules,
        )

        acps[f"dropbox-mn{index}"] = resource_id

        print(
            f"  ✓ acp-dropbox-mn{index} "
            f"ri={resource_id} "
            f"(MN-{index}: CREATE={ACOP_CREATE})"
        )

        time.sleep(0.2)

        # acp-sensor-mnX
        resource_id = om2m.create_acp(
            cse,
            f"acp-sensor-mn{index}",
            pv_rules=[
                {
                    "acor": [
                        ORIGIN_ADMIN,
                        mn_origin,
                    ],
                    "acop": ACOP_ALL,
                }
            ],
            pvs_rules=pvs_rules,
        )

        acps[f"sensor-mn{index}"] = resource_id

        print(
            f"  ✓ acp-sensor-mn{index} "
            f"ri={resource_id} "
            f"(MN-{index}: ALL={ACOP_ALL})"
        )

        time.sleep(0.2)

        # acp-cache-mnX
        resource_id = om2m.create_acp(
            cse,
            f"acp-cache-mn{index}",
            pv_rules=[
                {
                    "acor": [
                        ORIGIN_ADMIN,
                        ORIGIN_IN_AE,
                    ],
                    "acop": ACOP_RETRIEVE,
                },
                {
                    "acor": [mn_origin],
                    "acop": ACOP_ALL,
                },
            ],
            pvs_rules=pvs_rules,
        )

        acps[f"cache-mn{index}"] = resource_id

        print(
            f"  ✓ acp-cache-mn{index} "
            f"ri={resource_id} "
            f"(IN-AE: RETRIEVE={ACOP_RETRIEVE}, "
            f"MN-{index}: ALL={ACOP_ALL})"
        )

        time.sleep(0.2)

    return acps


# ════════════════════════════════════════════════════════
# IN-AE 리소스 생성
# ════════════════════════════════════════════════════════

def setup_in_cse_resources(acps: dict) -> None:
    print("\n=== IN-AE 리소스 생성 ===")

    cse = config.CSE_NAME

    headers = config.HEADERS.copy()
    headers["Content-Type"] = "application/json;ty=2"
    headers["X-M2M-Origin"] = ORIGIN_IN_AE

    payload = {
        "m2m:ae": {
            "rn": config.IN_AE_NAME,
            "api": f"N{config.IN_AE_NAME}",
            "rr": True,
            "srv": ["2a"],
        }
    }

    response = requests.post(
        f"{config.BASE_URL}/{cse}",
        json=payload,
        headers=headers,
        timeout=10,
    )

    print(
        f"  ✓ AE: {config.IN_AE_NAME} "
        f"({response.status_code})"
    )

    time.sleep(0.3)

    in_ae_path = f"{cse}/{config.IN_AE_NAME}"

    om2m.create_container(
        in_ae_path,
        "cnt-fl-control",
        mni=50,
        acpi=(
            [acps["fl-control"]]
            if acps.get("fl-control")
            else None
        ),
    )

    time.sleep(0.3)

    om2m.create_container(
        in_ae_path,
        "cnt-global-model",
        mni=50,
        acpi=(
            [acps["global-model"]]
            if acps.get("global-model")
            else None
        ),
    )

    time.sleep(0.3)

    om2m.create_container(
        in_ae_path,
        "cnt-local-updates",
        mni=50,
        acpi=(
            [acps["local-updates"]]
            if acps.get("local-updates")
            else None
        ),
    )

    time.sleep(0.3)

    print("  ✓ IN-AE 리소스 구조 생성 완료")


# ════════════════════════════════════════════════════════
# MN-AE 리소스 생성
# ════════════════════════════════════════════════════════

def setup_mn_cse_resources(acps: dict) -> None:
    print("\n=== MN-AE 리소스 생성 ===")

    cse = config.CSE_NAME

    for index in range(1, config.NUM_CLIENTS + 1):
        ae_name = f"MN-AE-{index}"

        print(f"\n[{ae_name}]")

        headers = config.HEADERS.copy()
        headers["Content-Type"] = "application/json;ty=2"
        headers["X-M2M-Origin"] = f"CMN-AE-{index}"

        payload = {
            "m2m:ae": {
                "rn": ae_name,
                "api": f"N{ae_name}",
                "rr": True,
                "srv": ["2a"],
            }
        }

        response = requests.post(
            f"{config.BASE_URL}/{cse}",
            json=payload,
            headers=headers,
            timeout=10,
        )

        print(
            f"  ✓ AE: {ae_name} "
            f"({response.status_code})"
        )

        time.sleep(0.3)

        mn_ae_path = f"{cse}/{ae_name}"

        om2m.create_container(
            mn_ae_path,
            "cnt-sensor-data",
            mni=25,
            acpi=(
                [acps[f"sensor-mn{index}"]]
                if acps.get(f"sensor-mn{index}")
                else None
            ),
        )

        time.sleep(0.3)

        om2m.create_container(
            mn_ae_path,
            "cnt-local-model",
            mni=5,
            acpi=(
                [acps[f"cache-mn{index}"]]
                if acps.get(f"cache-mn{index}")
                else None
            ),
        )

        time.sleep(0.3)

        print(f"  ✓ {ae_name} 구조 생성 완료")

    # IN-AE 아래에 MN별 로컬 업데이트 컨테이너 생성
    print("\n  MN별 local-update dropbox 생성...")

    in_updates_path = (
        f"{cse}/"
        f"{config.IN_AE_NAME}/"
        "cnt-local-updates"
    )

    for index in range(1, config.NUM_CLIENTS + 1):
        om2m.create_container(
            in_updates_path,
            f"cnt-mn{index}",
            mni=50,
            acpi=(
                [acps[f"dropbox-mn{index}"]]
                if acps.get(f"dropbox-mn{index}")
                else None
            ),
        )

        time.sleep(0.3)

        print(
            "    ✓ "
            f"{in_updates_path}/cnt-mn{index}"
        )


# ════════════════════════════════════════════════════════
# 리소스 검증
# ════════════════════════════════════════════════════════

def verify_resources() -> bool:
    print("\n=== 리소스 확인 ===")

    cse = config.CSE_NAME

    paths = [
        f"{cse}/{config.IN_AE_NAME}",
        f"{cse}/{config.IN_AE_NAME}/cnt-fl-control",
        f"{cse}/{config.IN_AE_NAME}/cnt-global-model",
        f"{cse}/{config.IN_AE_NAME}/cnt-local-updates",
    ]

    for index in range(1, config.NUM_CLIENTS + 1):
        ae_name = f"MN-AE-{index}"

        paths.extend(
            [
                f"{cse}/{ae_name}",
                f"{cse}/{ae_name}/cnt-sensor-data",
                f"{cse}/{ae_name}/cnt-local-model",
                (
                    f"{cse}/{config.IN_AE_NAME}/"
                    f"cnt-local-updates/cnt-mn{index}"
                ),
            ]
        )

    headers = config.HEADERS.copy()

    success_count = 0

    for path in paths:
        try:
            response = requests.get(
                f"{config.BASE_URL}/{path}",
                headers=headers,
                timeout=10,
            )
        except requests.RequestException as error:
            print(f"  ✗ {path}: {error}")
            continue

        status = (
            "✓"
            if response.status_code == 200
            else "✗"
        )

        print(
            f"  {status} {path} "
            f"({response.status_code})"
        )

        if response.status_code == 200:
            success_count += 1

    print(
        f"\n  총 {success_count}/{len(paths)} "
        "리소스 확인"
    )

    return success_count == len(paths)


# ════════════════════════════════════════════════════════
# ACP 연결 검증
# ════════════════════════════════════════════════════════

def verify_acp() -> bool:
    print("\n=== ACP 연결 확인 ===")

    cse = config.CSE_NAME

    check_paths = [
        f"{cse}/{config.IN_AE_NAME}/cnt-fl-control",
        f"{cse}/{config.IN_AE_NAME}/cnt-global-model",
        f"{cse}/{config.IN_AE_NAME}/cnt-local-updates",
    ]

    for index in range(1, config.NUM_CLIENTS + 1):
        ae_name = f"MN-AE-{index}"

        check_paths.extend(
            [
                f"{cse}/{ae_name}/cnt-sensor-data",
                f"{cse}/{ae_name}/cnt-local-model",
                (
                    f"{cse}/{config.IN_AE_NAME}/"
                    f"cnt-local-updates/cnt-mn{index}"
                ),
            ]
        )

    headers = config.HEADERS.copy()

    all_ok = True

    for path in check_paths:
        try:
            response = requests.get(
                f"{config.BASE_URL}/{path}",
                headers=headers,
                timeout=10,
            )
        except requests.RequestException as error:
            print(f"  ✗ {path}: {error}")
            all_ok = False
            continue

        if response.status_code != 200:
            print(
                f"  ✗ {path} "
                f"({response.status_code})"
            )

            all_ok = False
            continue

        try:
            body = response.json()
        except ValueError:
            print(f"  ✗ JSON 파싱 실패: {path}")
            all_ok = False
            continue

        acpi = (
            body
            .get("m2m:cnt", {})
            .get("acpi", [])
        )

        short_path = path.replace(
            f"{cse}/",
            "",
        )

        if acpi:
            print(
                f"  ✓ {short_path} "
                f"acpi={acpi}"
            )
        else:
            print(
                f"  ⚠ acpi 없음: "
                f"{short_path}"
            )

            all_ok = False

    if all_ok:
        print("\n  ✓ 모든 Container의 ACP 연결 완료")
    else:
        print("\n  ⚠ 일부 Container의 ACP 연결 실패")

    return all_ok


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main() -> None:
    do_clean = "--clean" in sys.argv

    print("=" * 60)
    print("표준 oneM2M FL 리소스 구조 생성")
    print("=" * 60)

    if do_clean:
        clean_db()

        print(
            "\n  DB 초기화 반영 대기 "
            "(3초)..."
        )

        time.sleep(3)

    acps = setup_acp_resources()

    setup_in_cse_resources(
        acps,
    )

    setup_mn_cse_resources(
        acps,
    )

    resources_ok = verify_resources()
    acp_ok = verify_acp()

    print("\n" + "=" * 60)

    if resources_ok and acp_ok:
        print("✓ 리소스 생성 및 ACP 연결 완료")
    else:
        print(
            "⚠ 일부 작업 실패 — "
            "위 로그를 확인하세요."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()