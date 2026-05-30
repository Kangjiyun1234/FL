#!/bin/bash
# FL 재실행 전 초기화: TinyIoT DB + 모델 파일 + 리소스 재생성 + 센서 데이터 재업로드
set -e

cd "$(dirname "$0")"

echo "[1/4] Cleaning TinyIoT DB and recreating resources..."
python3 fl/setup_resources_standard.py --clean

echo "[2/4] Removing old model files..."
rm -f /tmp/fl_models/global/global_round*.pt
rm -f /tmp/fl_models/local/mn*/round*.pt
echo "  ✓ model files removed"

echo "[3/4] Uploading sensor data..."
python3 fl/data_generator.py

echo "[4/4] Done. Ready to run FL."
echo "  → Start: bash run_fl.sh"
