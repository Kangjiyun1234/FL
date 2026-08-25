#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PIDS=()


cleanup() {

  if [ ${#PIDS[@]} -gt 0 ]; then

    kill "${PIDS[@]}" \
      2>/dev/null || true

  fi
}


trap cleanup EXIT INT TERM


echo "[1/6] Preparing Isaac Sim data..."

python3 fl/prepare_data_isaac.py


echo "[2/6] Setting up oneM2M resources..."

python3 fl/setup_resources_standard.py


echo "[3/6] Generating sensor-data metadata..."

python3 fl/data_generator.py


# ============================================================
# Dashboard를 FL보다 먼저 시작
#
# 처음부터 chronological stream을 보기 위함
# ============================================================

echo "[4/6] Starting Dashboard first..."

echo "  -> http://localhost:7000"

python3 -u fl/dashboard_server.py &

PIDS+=("$!")

echo "  Dashboard PID: ${PIDS[-1]}"

sleep 2


# ============================================================
# MN-AE
# ============================================================

echo "[5/6] Starting MN-AEs..."


python3 -u fl/mn_ae_standard.py 0 5001 &

PIDS+=("$!")

echo "  MN-AE-1 PID: ${PIDS[-1]}"


python3 -u fl/mn_ae_standard.py 1 5002 &

PIDS+=("$!")

echo "  MN-AE-2 PID: ${PIDS[-1]}"


python3 -u fl/mn_ae_standard.py 2 5003 &

PIDS+=("$!")

echo "  MN-AE-3 PID: ${PIDS[-1]}"


# MN-AE subscription 생성 대기
sleep 5


# ============================================================
# IN-AE
# ============================================================

echo "[6/6] Starting IN-AE..."

python3 -u fl/in_ae_standard.py &

PIDS+=("$!")

echo "  IN-AE PID: ${PIDS[-1]}"


echo
echo "============================================================"
echo "Isaac Sim oneM2M-FL demo running"
echo "Dashboard: http://localhost:7000"
echo "============================================================"
echo
echo "Ctrl+C to stop."


wait