#!/usr/bin/env bash
# run_femto_isaac_demo.sh
#
# FEMTO-Isaac Sim FL demo runner
#
# 실행:
#   ./run_femto_isaac_demo.sh
#
# 옵션:
#   ./run_femto_isaac_demo.sh --skip-clean
#   ./run_femto_isaac_demo.sh --no-prompt
#   ./run_femto_isaac_demo.sh --exit-on-complete
#   ./run_femto_isaac_demo.sh --full-train
#
# 주의:
# - TinyIoT는 이 스크립트가 직접 실행하지 않음. 먼저 실행되어 있어야 함.
# - Isaac Sim GUI도 이 스크립트가 직접 실행하지 않음.
# - Isaac Sim v7은 안내 문구가 뜬 뒤 Script Editor에서 직접 실행하고 Enter를 누르면 됨.
# - 이 runner는 Isaac replay demo 전용 reset을 사용함.
#   기존 clean_fl.sh처럼 /tmp/fl_data/femto/mn*.pkl을 요구하거나 fl/data_generator.py를 실행하지 않음.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "${SCRIPT_DIR}/fl" && -f "${SCRIPT_DIR}/clean_fl.sh" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

cd "$ROOT_DIR"

ONEM2M_BASE_URL="${ONEM2M_BASE_URL:-http://127.0.0.1:3000}"
CSE_NAME="${CSE_NAME:-TinyIoT}"

DASHBOARD_PORT="${DASHBOARD_PORT:-7000}"
MN1_PORT="${MN1_PORT:-5001}"
MN2_PORT="${MN2_PORT:-5002}"
MN3_PORT="${MN3_PORT:-5003}"

FEMTO_REPLAY_DIR="${FEMTO_REPLAY_DIR:-/mnt/c/Projects/bearing_testbed/data/femto_replay}"
FL_PKL_DIR="${FL_PKL_DIR:-/mnt/c/Projects/bearing_testbed/data/fl_buffer}"

FL_MODEL_BASE_DIR="${FL_MODEL_BASE_DIR:-/tmp/fl_models}"
FL_DEMO_RUN_MARKER="${FL_DEMO_RUN_MARKER:-${FL_MODEL_BASE_DIR}/.demo_run_started}"

FL_SENSOR_ROUND_WAIT_SEC="${FL_SENSOR_ROUND_WAIT_SEC:-90}"
FL_SENSOR_ROUND_POLL_SEC="${FL_SENSOR_ROUND_POLL_SEC:-0.5}"
FL_DEMO_ROUND_TRAIN_N="${FL_DEMO_ROUND_TRAIN_N:-50}"

ISAAC_SCRIPT_WIN="${ISAAC_SCRIPT_WIN:-C:\\Projects\\bearing_testbed\\scripts\\isaac_femto_demo_gateway_v7.py}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${FL_DEMO_LOG_DIR:-logs/femto_isaac_demo/${RUN_ID}}"

DO_CLEAN=1
PROMPT_BEFORE_IN=1
EXIT_ON_COMPLETE=0

for arg in "$@"; do
  case "$arg" in
    --skip-clean)
      DO_CLEAN=0
      ;;
    --no-prompt)
      PROMPT_BEFORE_IN=0
      ;;
    --exit-on-complete)
      EXIT_ON_COMPLETE=1
      ;;
    --full-train)
      FL_DEMO_ROUND_TRAIN_N=0
      ;;
    -h|--help)
      sed -n '1,75p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$LOG_DIR"

PIDS=()
NAMES=()
TAIL_PID=""

log() {
  printf '[demo-runner] %s\n' "$*"
}

die() {
  printf '[demo-runner] ERROR: %s\n' "$*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

check_tinyiot() {
  have_cmd curl || die "curl is required. Install it with: sudo apt install curl"

  local url="${ONEM2M_BASE_URL}/${CSE_NAME}"
  local http_code
  local curl_exit

  log "Checking TinyIoT endpoint: ${url}"

  set +e
  http_code="$(
    curl -sS --max-time 3 \
      -o "${LOG_DIR}/tinyiot_check_body.txt" \
      -w "%{http_code}" \
      -H "X-M2M-Origin: CAdmin" \
      -H "X-M2M-RVI: 2a" \
      -H "Accept: application/json" \
      "${url}"
  )"
  curl_exit=$?
  set -e

  if [[ "${curl_exit}" -ne 0 || "${http_code}" == "000" ]]; then
    die "TinyIoT endpoint did not respond. Check ONEM2M_BASE_URL=${ONEM2M_BASE_URL}"
  fi

  log "TinyIoT endpoint responded with HTTP ${http_code}; continuing."
}

check_replay_streams() {
  log "Checking FEMTO replay stream files: ${FEMTO_REPLAY_DIR}"

  local missing=0
  for node in mn1 mn2 mn3; do
    local f="${FEMTO_REPLAY_DIR}/${node}_stream.npz"
    if [[ ! -f "${f}" ]]; then
      log "Missing: ${f}"
      missing=1
    else
      log "  OK ${f}"
    fi
  done

  if [[ "${missing}" == "1" ]]; then
    die "FEMTO replay stream files are missing. Run prepare_data_femto_replay.py and copy *_stream.npz to ${FEMTO_REPLAY_DIR}"
  fi
}

reset_demo_state() {
  log "Resetting FEMTO-Isaac demo state"

  {
    echo "============================================================"
    echo "FEMTO-Isaac demo reset"
    echo "============================================================"

    echo
    echo "[1/4] Stop previous FL/Dashboard processes"
    pkill -TERM -f "fl/dashboard_server.py" 2>/dev/null || true
    pkill -TERM -f "fl/in_ae_standard.py" 2>/dev/null || true
    pkill -TERM -f "fl/mn_ae_standard.py" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "fl/dashboard_server.py" 2>/dev/null || true
    pkill -KILL -f "fl/in_ae_standard.py" 2>/dev/null || true
    pkill -KILL -f "fl/mn_ae_standard.py" 2>/dev/null || true
    echo "  OK processes stopped"

    echo
    echo "[2/4] Clear model/cache directories"
    rm -rf "${FL_MODEL_BASE_DIR}/global"
    rm -rf "${FL_MODEL_BASE_DIR}/local"
    rm -rf "${FL_MODEL_BASE_DIR}/cache"
    rm -f "${FL_DEMO_RUN_MARKER}"

    mkdir -p "${FL_MODEL_BASE_DIR}/global"
    mkdir -p "${FL_MODEL_BASE_DIR}/local"
    mkdir -p "${FL_MODEL_BASE_DIR}/cache"
    echo "  OK ${FL_MODEL_BASE_DIR}/global"
    echo "  OK ${FL_MODEL_BASE_DIR}/local"
    echo "  OK ${FL_MODEL_BASE_DIR}/cache"

    echo
    echo "[3/4] Recreate TinyIoT oneM2M resources"
    python3 fl/setup_resources_standard.py --clean

    echo
    echo "[4/4] Create demo run marker"
    mkdir -p "$(dirname "${FL_DEMO_RUN_MARKER}")"
    touch "${FL_DEMO_RUN_MARKER}"
    echo "  OK ${FL_DEMO_RUN_MARKER}"

    echo
    echo "============================================================"
    echo "FEMTO-Isaac demo reset complete"
    echo "============================================================"
    echo "Note: sensor-data metadata will be published by Isaac Sim v7."
    echo "      This runner intentionally does not call fl/data_generator.py."
    echo "============================================================"
  } | tee "${LOG_DIR}/demo_reset.log"
}

wait_http_port() {
  local name="$1"
  local url="$2"
  local max_sec="${3:-30}"
  local start
  start="$(date +%s)"

  while true; do
    if curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then
      log "${name} ready: ${url}"
      return 0
    fi

    if (( $(date +%s) - start >= max_sec )); then
      log "${name} was not confirmed within ${max_sec}s: ${url}"
      return 1
    fi

    sleep 1
  done
}

wait_port_open() {
  local name="$1"
  local port="$2"
  local max_sec="${3:-30}"
  local start
  start="$(date +%s)"

  while true; do
    if python3 - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    sock.close()
PY
    then
      log "${name} port ready: ${port}"
      return 0
    fi

    if (( $(date +%s) - start >= max_sec )); then
      log "${name} port was not confirmed within ${max_sec}s: ${port}"
      return 1
    fi

    sleep 1
  done
}

start_bg() {
  local name="$1"
  shift

  local logfile="${LOG_DIR}/${name}.log"

  log "Starting ${name}"
  log "  log: ${logfile}"

  (
    cd "$ROOT_DIR"
    "$@"
  ) >"${logfile}" 2>&1 &

  local pid=$!
  PIDS+=("$pid")
  NAMES+=("$name")
  log "  pid: ${pid}"
}

cleanup() {
  local code=$?

  if [[ -n "${TAIL_PID}" ]]; then
    kill "${TAIL_PID}" >/dev/null 2>&1 || true
  fi

  if [[ ${#PIDS[@]} -gt 0 ]]; then
    log "Stopping background processes..."
    for pid in "${PIDS[@]}"; do
      kill "${pid}" >/dev/null 2>&1 || true
    done

    sleep 1

    for pid in "${PIDS[@]}"; do
      kill -9 "${pid}" >/dev/null 2>&1 || true
    done
  fi

  log "Logs saved: ${LOG_DIR}"
  exit "$code"
}

trap cleanup EXIT INT TERM

print_isaac_instruction() {
  cat <<EOF

────────────────────────────────────────────────────────
Isaac Sim v7 step
────────────────────────────────────────────────────────

Run this in Isaac Sim Script Editor:

exec(open(r"${ISAAC_SCRIPT_WIN}", encoding="utf-8").read())

Expected Isaac Sim log:
[start] FEMTO Isaac synced demo started - v7
Waiting for IN-AE FL_TRAINING round command...

After confirming that Isaac Sim is waiting, return here and press Enter.
Then IN-AE will start.

EOF
}

main() {
  log "root: ${ROOT_DIR}"
  log "logs: ${LOG_DIR}"
  log "FEMTO_REPLAY_DIR: ${FEMTO_REPLAY_DIR}"
  log "FL_PKL_DIR: ${FL_PKL_DIR}"
  log "FL_DEMO_ROUND_TRAIN_N: ${FL_DEMO_ROUND_TRAIN_N}"
  log "Isaac script: ${ISAAC_SCRIPT_WIN}"

  check_tinyiot
  check_replay_streams

  if [[ ! -d "${FL_PKL_DIR}" ]]; then
    log "Note: FL_PKL_DIR does not exist yet: ${FL_PKL_DIR}"
    log "Isaac Sim v7 can create the buffer pkl files when it starts."
  fi

  if [[ "${DO_CLEAN}" == "1" ]]; then
    reset_demo_state
  else
    log "Skipping demo reset"
  fi

  start_bg "dashboard" env \
    FL_PKL_DIR="${FL_PKL_DIR}" \
    python3 -u fl/dashboard_server.py

  wait_http_port "dashboard" "http://127.0.0.1:${DASHBOARD_PORT}/" 30 || true

  start_bg "mn1" env \
    FL_SENSOR_ROUND_WAIT_SEC="${FL_SENSOR_ROUND_WAIT_SEC}" \
    FL_SENSOR_ROUND_POLL_SEC="${FL_SENSOR_ROUND_POLL_SEC}" \
    FL_DEMO_ROUND_TRAIN_N="${FL_DEMO_ROUND_TRAIN_N}" \
    python3 -u fl/mn_ae_standard.py 0 "${MN1_PORT}"

  start_bg "mn2" env \
    FL_SENSOR_ROUND_WAIT_SEC="${FL_SENSOR_ROUND_WAIT_SEC}" \
    FL_SENSOR_ROUND_POLL_SEC="${FL_SENSOR_ROUND_POLL_SEC}" \
    FL_DEMO_ROUND_TRAIN_N="${FL_DEMO_ROUND_TRAIN_N}" \
    python3 -u fl/mn_ae_standard.py 1 "${MN2_PORT}"

  start_bg "mn3" env \
    FL_SENSOR_ROUND_WAIT_SEC="${FL_SENSOR_ROUND_WAIT_SEC}" \
    FL_SENSOR_ROUND_POLL_SEC="${FL_SENSOR_ROUND_POLL_SEC}" \
    FL_DEMO_ROUND_TRAIN_N="${FL_DEMO_ROUND_TRAIN_N}" \
    python3 -u fl/mn_ae_standard.py 2 "${MN3_PORT}"

  wait_port_open "mn1" "${MN1_PORT}" 30 || true
  wait_port_open "mn2" "${MN2_PORT}" 30 || true
  wait_port_open "mn3" "${MN3_PORT}" 30 || true

  print_isaac_instruction

  if [[ "${PROMPT_BEFORE_IN}" == "1" ]]; then
    read -r -p "Press Enter after Isaac Sim v7 is waiting for the FL round command: " _
  else
    log "--no-prompt: Starting IN-AE without manual confirmation"
  fi

  start_bg "in_ae" python3 -u fl/in_ae_standard.py

  tail -n +1 -F \
    "${LOG_DIR}/dashboard.log" \
    "${LOG_DIR}/mn1.log" \
    "${LOG_DIR}/mn2.log" \
    "${LOG_DIR}/mn3.log" \
    "${LOG_DIR}/in_ae.log" &
  TAIL_PID=$!

  log "IN-AE started."
  log "Dashboard: http://localhost:${DASHBOARD_PORT}"
  log "Log directory: ${LOG_DIR}"

  local in_pid="${PIDS[-1]}"
  wait "${in_pid}" || true

  log "IN-AE process finished."

  if [[ "${EXIT_ON_COMPLETE}" == "1" ]]; then
    log "--exit-on-complete: stopping services"
    return 0
  fi

  cat <<EOF

────────────────────────────────────────────────────────
FL run appears to be complete.
Dashboard:
  http://localhost:${DASHBOARD_PORT}

Press Ctrl+C here to stop all background processes.
Logs:
  ${LOG_DIR}
────────────────────────────────────────────────────────

EOF

  while true; do
    sleep 3600
  done
}

main "$@"
