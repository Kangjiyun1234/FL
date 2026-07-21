#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PIDS=()
cleanup() {
  if [ ${#PIDS[@]} -gt 0 ]; then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[1/6] Preparing FEMTO data..."
python3 fl/prepare_data_femto.py

echo "[2/6] Setting up oneM2M resources..."
python3 fl/setup_resources_standard.py

echo "[3/6] Generating data..."
python3 fl/data_generator.py

echo "[4/6] Starting MN-AEs first..."
python3 -u fl/mn_ae_standard.py 0 5001 &
PIDS+=("$!")
echo "  MN-AE-1 PID: ${PIDS[-1]}"

python3 -u fl/mn_ae_standard.py 1 5002 &
PIDS+=("$!")
echo "  MN-AE-2 PID: ${PIDS[-1]}"

python3 -u fl/mn_ae_standard.py 2 5003 &
PIDS+=("$!")
echo "  MN-AE-3 PID: ${PIDS[-1]}"

# 각 MN-AE가 cnt-fl-control subscription을 생성할 시간 확보
sleep 5

echo "[5/6] Starting IN-AE after MN subscriptions..."
python3 -u fl/in_ae_standard.py &
PIDS+=("$!")
echo "  IN-AE PID: ${PIDS[-1]}"

sleep 2

echo "[6/6] Starting Dashboard..."
echo "  -> http://localhost:7000"
python3 fl/dashboard_server.py