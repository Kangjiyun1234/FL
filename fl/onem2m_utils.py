"""
oneM2M Utility Functions (TinyIoT-friendly)
- AE / CNT / CIN CRUD
- ACP (Access Control Policy) 생성
- Subscription
- Discovery with filterCriteria-like query (label-based)
- Retry: timeout/connection 에러 시 지수 백오프 재시도
"""

from __future__ import annotations

import json
import time
import requests
from typing import Dict, List, Optional, Any
import config


# -------------------------
# Retry 설정
# -------------------------
MAX_RETRY     = 4
RETRY_DELAY   = 3.0
RETRY_BACKOFF = 1.5

def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    delay = RETRY_DELAY
    last_exc = None
    for attempt in range(MAX_RETRY):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRY - 1:
                print(f"  ⏳ retry {attempt+1}/{MAX_RETRY-1} ({delay:.1f}s): {type(e).__name__}")
                time.sleep(delay)
                delay *= RETRY_BACKOFF
    raise last_exc


# -------------------------
# Headers
# -------------------------

def _headers(ty: Optional[int] = None) -> Dict[str, str]:
    h = config.HEADERS.copy()
    if ty is not None:
        h["Content-Type"] = f"application/json;ty={ty}"
    return h


# -------------------------
# ACP
# -------------------------

def create_acp(
    parent_path: str,
    acp_name: str,
    pv_rules: List[Dict],
    pvs_rules: List[Dict],
) -> Optional[str]:
    """
    ACP 생성 후 ri 반환.
    pv_rules/pvs_rules 예시:
      [{"acor": ["CAdmin", "CIN-AE"], "acop": 63}]
    """
    url = f"{config.BASE_URL}/{parent_path}"
    payload = {
        "m2m:acp": {
            "rn": acp_name,
            "pv":  {"acr": pv_rules},
            "pvs": {"acr": pvs_rules},
        }
    }
    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(1), timeout=10)
        if r.status_code == 201:
            ri = r.json().get("m2m:acp", {}).get("ri")
            return ri
        elif r.status_code == 409:
            # 이미 존재 → ri 조회
            get_url = f"{config.BASE_URL}/{parent_path}/{acp_name}"
            rg = _request_with_retry("get", get_url, headers=_headers(), timeout=10)
            if rg.status_code == 200:
                return rg.json().get("m2m:acp", {}).get("ri")
        print(f"✗ ACP create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ ACP create error: {e}")
        return None


def update_acpi(resource_path: str, acpi: List[str]) -> bool:
    """
    기존 컨테이너에 acpi 연결 (PUT/UPDATE).
    acpi: ACP ri 목록 e.g. ["1-20260309T221953xxxx"]
    """
    url = f"{config.BASE_URL}/{resource_path}"
    payload = {"m2m:cnt": {"acpi": acpi}}
    try:
        r = _request_with_retry("put", url, json=payload, headers=_headers(), timeout=10)
        return r.status_code in (200, 204)
    except Exception as e:
        print(f"✗ update_acpi error: {e}")
        return False


# -------------------------
# AE / CNT / CIN
# -------------------------

def create_application(ae_name: str) -> Optional[Dict]:
    url = f"{config.BASE_URL}/{config.CSE_NAME}"
    payload = {
        "m2m:ae": {
            "rn": ae_name,
            "api": f"N.{ae_name}.app",
            "rr": True,
            "srv": ["2a"],
        }
    }
    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(2), timeout=10)
        if r.status_code in (201, 409):
            return r.json() if r.status_code == 201 else {"exists": True}
        print(f"✗ AE create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ AE create error: {e}")
        return None


def delete_resource(resource_path: str) -> bool:
    """리소스 삭제 (존재하지 않으면 True 반환)."""
    url = f"{config.BASE_URL}/{resource_path}"
    try:
        r = _request_with_retry("delete", url, headers=_headers(), timeout=10)
        return r.status_code in (200, 202, 204, 404)
    except Exception as e:
        print(f"✗ DELETE error: {e}")
        return False


def create_container(
    parent_path: str,
    container_name: str,
    mni: int = 1000,
    mbs: int = 10_000_000,
    acpi: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Container 생성.
    mbs: max byte size (기본 10MB — TinyIoT 기본값 초과로 인한 cleanup crash 방지)
    acpi: ACP ri 목록. 지정 시 생성과 동시에 acpi 연결 시도,
          실패하면 생성 후 UPDATE로 재시도.
    """
    url = f"{config.BASE_URL}/{parent_path}"
    cnt_obj: Dict[str, Any] = {"rn": container_name, "mni": mni, "mbs": mbs}
    if acpi:
        cnt_obj["acpi"] = acpi

    payload = {"m2m:cnt": cnt_obj}
    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(3), timeout=10)
        if r.status_code in (201, 409):
            result = r.json() if r.status_code == 201 else {"exists": True}

            # acpi가 있는데 생성 응답에 acpi가 없으면 UPDATE로 재시도
            if acpi:
                created_acpi = (r.json().get("m2m:cnt", {}).get("acpi", [])
                                if r.status_code == 201 else [])
                if not created_acpi:
                    resource_path = f"{parent_path}/{container_name}"
                    ok = update_acpi(resource_path, acpi)
                    if not ok:
                        print(f"  ⚠ acpi UPDATE 실패: {resource_path}")

            return result
        print(f"✗ CNT create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ CNT create error: {e}")
        return None


def create_content_instance(
    container_path: str,
    content: Any,
    labels: Optional[List[str]] = None,
) -> Optional[Dict]:
    url = f"{config.BASE_URL}/{container_path}"
    con_str = (json.dumps(content, separators=(",", ":"), ensure_ascii=False)
               if isinstance(content, (dict, list)) else str(content))

    payload: Dict[str, Any] = {"m2m:cin": {"con": con_str}}
    if labels:
        payload["m2m:cin"]["lbl"] = labels

    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(4), timeout=45)
        if r.status_code == 201:
            return r.json()
        print(f"✗ CIN create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ CIN create error: {e}")
        return None


def get_latest_content_instance(container_path: str) -> Optional[Dict]:
    url = f"{config.BASE_URL}/{container_path}/la"
    try:
        r = _request_with_retry("get", url, headers=_headers(), timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"✗ get_latest CIN error: {e}")
        return None


def get_resource(resource_path: str) -> Optional[Dict]:
    url = f"{config.BASE_URL}/{resource_path.lstrip('/')}"
    try:
        r = _request_with_retry("get", url, headers=_headers(), timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"✗ get_resource error: {e}")
        return None


# -------------------------
# Subscription (TinyIoT)
# -------------------------

def create_subscription(
    parent_path: str,
    subscription_name: str,
    notification_uri: str,
    event_types: List[int],
    use_nct: Optional[int] = None,
) -> Optional[Dict]:
    url = f"{config.BASE_URL}/{parent_path}"
    sub_obj: Dict[str, Any] = {
        "rn": subscription_name,
        "nu": [notification_uri],
        "enc": {"net": event_types},
    }
    if use_nct is not None:
        sub_obj["nct"] = use_nct

    payload = {"m2m:sub": sub_obj}
    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(23), timeout=10)
        if r.status_code in (201, 409):
            return r.json() if r.status_code == 201 else {"exists": True}
        print(f"✗ SUB create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ SUB create error: {e}")
        return None


def delete_subscription(parent_path: str, subscription_name: str) -> bool:
    url = f"{config.BASE_URL}/{parent_path}/{subscription_name}"
    try:
        r = _request_with_retry("delete", url, headers=_headers(), timeout=10)
        return r.status_code in (200, 404)
    except Exception:
        return False


# -------------------------
# Discovery (label-based)
# -------------------------

def discover_uril_by_label(container_path: str, label: str, lim: int = 20) -> List[str]:
    base = f"{config.BASE_URL}/{container_path}"
    candidates = [
        f"{base}?fu=1&ty=4&lim={lim}&lbl={label}",
        f"{base}?fu=1&ty=4&lim={lim}&lbl={requests.utils.quote(label)}",
        f"{base}?fu=1&lbl={label}",
    ]
    for url in candidates:
        try:
            r = requests.get(url, headers=_headers(), timeout=10)
            if r.status_code == 200:
                data = r.json()
                uril = data.get("m2m:uril", [])
                if isinstance(uril, list) and uril:
                    return uril
        except Exception:
            continue
    return []


def list_uril(container_path: str, lim: int = 50) -> List[str]:
    url = f"{config.BASE_URL}/{container_path}?fu=1&ty=4&lim={lim}"
    try:
        r = requests.get(url, headers=_headers(), timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        uril = data.get("m2m:uril", [])
        return uril if isinstance(uril, list) else []
    except Exception:
        return []


def retrieve_cins_by_label(
    container_path: str,
    label: str,
    lim: int = 30,
    fallback_scan: int = 120,
    newest_first: bool = True,
) -> List[Dict]:
    def _ct(res: Dict) -> str:
        try:
            return (res.get("m2m:cin") or {}).get("ct", "") or ""
        except Exception:
            return ""

    matched: List[Dict] = []

    uril = discover_uril_by_label(container_path, label, lim=lim)
    if uril:
        for u in uril:
            cin = get_resource(u.lstrip("/"))
            if not cin or "m2m:cin" not in cin:
                continue
            if label in (cin["m2m:cin"].get("lbl") or []):
                matched.append(cin)
        if matched:
            matched.sort(key=_ct, reverse=newest_first)
            return matched

    uril2 = list_uril(container_path, lim=fallback_scan)
    if not uril2:
        return []

    for u in reversed(uril2):
        cin = get_resource(u.lstrip("/"))
        if not cin or "m2m:cin" not in cin:
            continue
        if label in (cin["m2m:cin"].get("lbl") or []):
            matched.append(cin)

    matched.sort(key=_ct, reverse=newest_first)
    return matched


def retrieve_latest_cin_by_label(
    container_path: str,
    label: str,
    lim: int = 30,
    fallback_scan: int = 80,
) -> Optional[Dict]:
    cins = retrieve_cins_by_label(
        container_path, label, lim=lim, fallback_scan=fallback_scan
    )
    return cins[0] if cins else None


def retrieve_first_cin_by_label(container_path: str, label: str) -> Optional[Dict]:
    return retrieve_latest_cin_by_label(container_path, label)