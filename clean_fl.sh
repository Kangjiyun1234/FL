#!/bin/bash

# ============================================================
# FL 데모 초기화 스크립트
#
# 실행 작업:
#   1. 준비된 PKL 데이터 확인
#   2. TinyIoT DB 초기화
#   3. oneM2M 리소스 재생성
#   4. 이전 글로벌·로컬·캐시 모델 삭제
#   5. 새 데모 실행 표식 생성
#   6. 센서 데이터 경로 재등록
#
# 실행하지 않는 작업:
#   - PRONOSTIA 전처리
#
# 전처리가 필요한 경우:
#   ./prepare_data.sh
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

MODEL_BASE_DIR="${FL_MODEL_BASE_DIR:-/tmp/fl_models}"
RUN_MARKER="${FL_DEMO_RUN_MARKER:-${MODEL_BASE_DIR}/.demo_run_started}"
DATA_DIR="${FL_PKL_DIR:-/tmp/fl_data/femto}"

echo "============================================================"
echo "FL 데모 초기화"
echo "============================================================"


# ------------------------------------------------------------
# 1. 전처리 데이터 확인
# ------------------------------------------------------------

echo
echo "[1/6] 전처리 데이터 확인"

for node in mn1 mn2 mn3
do
    data_file="${DATA_DIR}/${node}.pkl"

    if [ ! -f "${data_file}" ]; then
        echo
        echo "오류: 전처리 데이터가 없습니다."
        echo "  ${data_file}"
        echo
        echo "먼저 전처리를 실행하세요."
        echo "  ./prepare_data.sh"
        exit 1
    fi

    echo "  ✓ ${data_file}"
done


# ------------------------------------------------------------
# 2. TinyIoT DB 초기화 및 리소스 재생성
# ------------------------------------------------------------

echo
echo "[2/6] TinyIoT DB 초기화 및 oneM2M 리소스 재생성"

python3 fl/setup_resources_standard.py --clean


# ------------------------------------------------------------
# 3. 이전 모델 및 캐시 삭제
# ------------------------------------------------------------

echo
echo "[3/6] 이전 글로벌·로컬·캐시 모델 삭제"

rm -rf "${MODEL_BASE_DIR}/global"
rm -rf "${MODEL_BASE_DIR}/local"
rm -rf "${MODEL_BASE_DIR}/cache"

mkdir -p "${MODEL_BASE_DIR}/global"
mkdir -p "${MODEL_BASE_DIR}/local"
mkdir -p "${MODEL_BASE_DIR}/cache"

echo "  ✓ ${MODEL_BASE_DIR}/global"
echo "  ✓ ${MODEL_BASE_DIR}/local"
echo "  ✓ ${MODEL_BASE_DIR}/cache"


# ------------------------------------------------------------
# 4. 새 데모 실행 표식 생성
# ------------------------------------------------------------

echo
echo "[4/6] 새 데모 실행 표식 생성"

mkdir -p "$(dirname "${RUN_MARKER}")"

rm -f "${RUN_MARKER}"
touch "${RUN_MARKER}"

echo "  ✓ ${RUN_MARKER}"


# ------------------------------------------------------------
# 5. 센서 데이터 경로 재등록
# ------------------------------------------------------------

echo
echo "[5/6] 센서 데이터 경로 oneM2M 등록"

python3 fl/data_generator.py


# ------------------------------------------------------------
# 6. 완료
# ------------------------------------------------------------

echo
echo "[6/6] 초기화 완료"

echo
echo "다음 명령으로 데모를 실행하세요."
echo "  ./run_fl.sh"
echo
echo "Design Tool이 열려 있다면 새로고침하세요."