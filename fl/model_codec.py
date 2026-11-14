"""
model_codec.py — FL 모델 가중치를 oneM2M contentInstance로 실어 나르기 위한 코덱

[왜 필요한가]
기존 구현은 contentInstance의 con에 로컬 파일 경로(model_path)만 넣고,
실제 PyTorch 가중치는 같은 PC의 파일시스템으로 주고받았다.
그래서 (1) MN-AE를 다른 머신에 올리면 즉시 깨지고,
(2) "oneM2M 리소스로 연합학습을 수행한다"는 주장도 성립하지 않았다.
(oneM2M이 나르는 것은 제어 메시지와 메타데이터뿐이었다)

[무엇을 하는가]
state_dict -> torch.save 직렬화 -> (선택) fp16 축소 -> gzip 압축 -> base64 문자열.
이 문자열을 con에 넣어 CSE를 통해 전송한다. 수신측은 역순으로 복원하고,
sha256으로 무결성을 검증한다.

[크기와 CSE 한계]
Conv1DAE(latent 32, seq_len 2560) = 749,281 파라미터 기준
    fp32 그대로   : base64 3.92 MB
    fp16 + gzip   : base64 1.78 MB   <- 기본값

그런데 TinyIoT는 요청 페이로드 상한이 64KB다(config.h MAX_PAYLOAD_SIZE,
httpd.c BUF_SIZE). 실측하면 60KB CIN은 생성되고 64KB는 400
"Invalid JSON format"으로 거부된다(버퍼에서 잘려 JSON이 깨진 것).

그래서 가중치는 한 CIN에 들어가지 않는다. 두 가지 경로를 모두 지원한다.

  (1) 단일 CIN  — base64가 상한 이하면 con에 그대로 넣는다
  (2) 청크 분할 — 상한을 넘으면 조각내어 전용 자식 컨테이너에 쌓고,
                  본 CIN에는 조각 개수와 해시만 담은 매니페스트를 싣는다

청크는 자식 컨테이너(cnt-weights)에 쌓기 때문에, 부모 컨테이너에 걸린
구독(CREATE_OF_DIRECT_CHILD)을 건드리지 않는다. 즉 조각 수십 개가
알림을 폭주시키지 않고, 매니페스트를 담은 CIN 하나만 알림을 발생시킨다.
각 조각 CIN은 rn을 명시적으로 주므로(w{round}_{0000}) 수신측이 discovery
없이 이름으로 바로 RETRIEVE 할 수 있다.

[fp16에 대하여]
가중치를 half로 줄여 보내는 것은 FL에서 흔한 양자화 압축이다.
상대오차가 4e-04 수준이라 AE 재구성 오차에 묻히지만, 민감하면
config.FL_WEIGHTS_FP16 = False 로 끄면 된다(전송량은 두 배).
"""

import base64
import gzip
import hashlib
import io
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

CODEC_FP16_GZIP = "fp16+gzip+b64"
CODEC_FP32_GZIP = "fp32+gzip+b64"

# 조각 CIN 하나에 들어가는 con은 {"i":번호,"d":"조각"} 형태의 JSON이다.
# 번호와 따옴표, 중괄호 등 부가 문자를 넉넉히 빼두기 위한 여유분.
CHUNK_OVERHEAD_BYTES = 2048


def encode_state_dict(
    state_dict: Dict[str, Any],
    fp16: bool = True,
    gzip_level: int = 6,
) -> Dict[str, Any]:
    """
    state_dict를 contentInstance에 넣을 수 있는 base64 문자열로 만든다.

    반환값은 con에 그대로 펼쳐 넣을 수 있는 dict이다:
        weights_b64    : base64 문자열 (실제 가중치)
        weights_codec  : 복원 방법 식별자
        weights_sha256 : 압축된 바이트의 해시 (무결성 검증용)
        weights_bytes  : base64 문자열 길이 (크기 가드용)
    """
    if fp16:
        payload_sd = {
            k: (v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v)
            for k, v in state_dict.items()
        }
    else:
        payload_sd = state_dict

    buf = io.BytesIO()
    torch.save(payload_sd, buf)
    raw = gzip.compress(buf.getvalue(), gzip_level)
    b64 = base64.b64encode(raw).decode("ascii")

    return {
        "weights_b64": b64,
        "weights_codec": CODEC_FP16_GZIP if fp16 else CODEC_FP32_GZIP,
        "weights_sha256": hashlib.sha256(raw).hexdigest(),
        "weights_bytes": len(b64),
    }


def split_for_cin(
    enc: Dict[str, Any],
    max_cin_bytes: int,
    overhead: int = CHUNK_OVERHEAD_BYTES,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    encode_state_dict 결과를 CIN 크기 상한에 맞게 조각낸다.

    돌려주는 것:
        chunks   : base64 조각 문자열 리스트 (순서가 곧 조각 번호)
        manifest : 본 CIN에 실을 메타데이터. weights_b64는 빠져 있고
                   대신 weights_chunked / weights_chunk_count 가 들어간다.
                   조각이 어느 컨테이너에 어떤 이름으로 쌓였는지는
                   경로를 아는 호출측이 추가로 채워 넣는다.
    """
    b64 = enc["weights_b64"]
    size = max(1024, int(max_cin_bytes) - int(overhead))

    chunks = [b64[i:i + size] for i in range(0, len(b64), size)]

    manifest = {
        "weights_chunked": True,
        "weights_chunk_count": len(chunks),
        "weights_chunk_size": size,
        "weights_codec": enc["weights_codec"],
        "weights_sha256": enc["weights_sha256"],
        "weights_bytes": enc["weights_bytes"],
    }
    return chunks, manifest


def decode_state_dict(
    meta: Dict[str, Any],
    fetch_chunk: Optional[Callable[[int], Optional[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    con(또는 그 일부)에서 state_dict를 복원한다.

    세 가지 경우를 처리한다.
      - weights_b64 가 있으면 그대로 복원 (단일 CIN)
      - weights_chunked 가 True 면 fetch_chunk(i)로 조각을 받아 이어붙인 뒤 복원
      - 둘 다 없으면 None. 호출측이 기존 model_path 방식으로 넘어가면 된다.

    fetch_chunk 는 조각 번호를 받아 base64 조각 문자열을 돌려주는 함수다.
    수신측이 CSE에서 어떻게 가져올지는 호출측이 정한다(이 모듈은 통신을 모른다).
    """
    b64 = meta.get("weights_b64")

    if not b64 and meta.get("weights_chunked"):
        if fetch_chunk is None:
            raise ValueError(
                "가중치가 조각으로 와 있는데 fetch_chunk가 없다"
            )

        count = int(meta.get("weights_chunk_count", 0))
        if count <= 0:
            raise ValueError("weights_chunk_count가 비정상이다")

        parts = []
        for i in range(count):
            part = fetch_chunk(i)
            if not part:
                raise ValueError(
                    f"가중치 조각 {i}/{count}를 가져오지 못했다"
                )
            parts.append(part)
        b64 = "".join(parts)

    if not b64:
        return None

    # 전송된 바이트(압축 상태) 그대로를 검증한다.
    # 인코딩 쪽도 같은 바이트를 해싱하므로 두 값이 일치해야 한다.
    compressed = base64.b64decode(b64)

    expected = meta.get("weights_sha256")
    if expected:
        actual = hashlib.sha256(compressed).hexdigest()
        if actual != expected:
            raise ValueError(
                f"weights sha256 mismatch: expected {expected[:12]}…, "
                f"got {actual[:12]}…"
            )

    raw = gzip.decompress(compressed)

    try:
        sd = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only 인자가 없는 구버전 torch
        sd = torch.load(io.BytesIO(raw), map_location="cpu")

    # fp16으로 보냈으면 학습/집계를 위해 fp32로 되돌린다
    if meta.get("weights_codec") == CODEC_FP16_GZIP:
        sd = {
            k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
            for k, v in sd.items()
        }
    return sd


def chunk_resource_name(round_num: int, index: int) -> str:
    """조각 CIN의 리소스 이름. 수신측이 discovery 없이 바로 RETRIEVE 하기 위함."""
    return f"w{int(round_num)}_{int(index):04d}"


def describe(meta: Dict[str, Any]) -> str:
    """로그용 한 줄 요약."""
    n = meta.get("weights_bytes", 0)
    codec = meta.get("weights_codec", "?")
    if meta.get("weights_chunked"):
        return (
            f"{codec} {n / 1024:.0f}KB "
            f"({meta.get('weights_chunk_count')}조각)"
        )
    return f"{codec} {n / 1024:.0f}KB (단일 CIN)"
