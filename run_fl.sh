#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "[1/6] Preparing FEMTO data..."
python3 fl/prepare_data_femto.py

echo "[2/6] Setting up oneM2M resources..."
python3 fl/setup_resources_standard.py

echo "[3/6] Generating data..."
python3 fl/data_generator.py

echo "[4/6] Starting IN-AE..."
python3 fl/in_ae_standard.py &
IN_PID=$!
echo "  IN-AE PID: $IN_PID"
sleep 3

echo "[5/6] Starting MN-AEs..."
python3 fl/mn_ae_standard.py 0 5001 &
echo "  MN-AE-1 PID: $!"
python3 fl/mn_ae_standard.py 1 5002 &
echo "  MN-AE-2 PID: $!"
python3 fl/mn_ae_standard.py 2 5003 &
echo "  MN-AE-3 PID: $!"
sleep 2

echo "[6/6] Starting Dashboard..."
echo "  → http://localhost:7000"
python3 fl/dashboard_server.py
