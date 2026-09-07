#!/usr/bin/env bash
# run_femto_isaac_demo.sh
#
# FEMTO-Isaac Sim FL demo runner
#
# 실행:
#   ./run_femto_isaac_demo.sh
# 또는 scripts 아래에 둔 경우:
#   ./scripts/run_femto_isaac_demo.sh
#
# 옵션:
#   ./run_femto_isaac_demo.sh --skip-clean
#   ./run_femto_isaac_demo.sh --no-prompt
#   ./run_femto_isaac_demo.sh --exit-on-complete
#   ./run_femto_isaac_demo.sh --full-train
#
# 주의:
# - TinyIoT는 이 스크립트가 켜지 않음. 먼저 켜져 있어야 함.
# - Isaac Sim GUI도 이 스크립트가 직접 켜지 않음.
# - Isaac Sim v6은 안내 문구가 뜬 뒤 Script Editor에서 직접 실행하고 Enter를 누르면 됨.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# root 바로 아래에 둬도 되고 scripts/ 아래에 둬도 되게 자동 판별
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

FL_PKL_DIR="${FL_PKL_DIR:-/mnt/c/Projects/bearing_testbed/data/fl_buffer}"
FL_SENSOR_ROUND_WAIT_SEC="${FL_SENSOR_ROUND_WAIT_SEC:-90}"
FL_SENSOR_ROUND_POLL_SEC="${FL_SENSOR_ROUND_POLL_SEC:-0.5}"

# demo 시간 단축용. 각 round train 80개 중 기본 50개만 균등 샘플링.
# 전체 validation/test stream은 줄이지 않음.
FL_DEMO_ROUND_TRAIN_N="${FL_DEMO_ROUND_TRAIN_N:-50}"

ISAAC_SCRIPT_WIN="${ISAAC_SCRIPT_WIN:-C:\\Projects\\bearing_testbed\\scripts\\isaac_femto_demo_gateway_v6.py}"

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
      sed -n '1,65p' "$0"
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
  have_cmd curl || die "curl이 필요함. sudo apt install curl 로 설치해."

  log "check TinyIoT: ${ONEM2M_BASE_URL}/${CSE_NAME}"

  if ! curl -fsS --max-time 3 \
    -H "X-M2M-Origin: CAdmin" \
    -H "Accept: application/json" \
    "${ONEM2M_BASE_URL}/${CSE_NAME}" >/dev/null; then
    die "TinyIoT가 안 켜져 있거나 ${ONEM2M_BASE_URL}/${CSE_NAME} 접근이 안 됨. TinyIoT 먼저 켜."
  fi

  log "TinyIoT OK"
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
      log "${name} not confirmed within ${max_sec}s: ${url}"
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
      log "${name} port not confirmed within ${max_sec}s: ${port}"
      return 1
    fi

    sleep 1
  done
}

start_bg() {
  local name="$1"
  shift

  local logfile="${LOG_DIR}/${name}.log"

  log "start ${name}"
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
    log "stopping background processes..."
    for pid in "${PIDS[@]}"; do
      kill "${pid}" >/dev/null 2>&1 || true
    done

    sleep 1

    for pid in "${PIDS[@]}"; do
      kill -9 "${pid}" >/dev/null 2>&1 || true
    done
  fi

  log "logs saved: ${LOG_DIR}"
  exit "$code"
}

trap cleanup EXIT INT TERM

print_isaac_instruction() {
  cat <<EOF

────────────────────────────────────────────────────────
Isaac Sim v6 실행 단계
────────────────────────────────────────────────────────

Isaac Sim Script Editor에서 아래 코드 실행:

exec(open(r"${ISAAC_SCRIPT_WIN}", encoding="utf-8").read())

정상 로그:
[start] FEMTO Isaac synced demo started - v6
Waiting for IN-AE FL_TRAINING round command...

위 로그를 확인한 뒤 이 터미널로 돌아와 Enter를 누르면 IN-AE가 시작됨.

EOF
}

main() {
  log "root: ${ROOT_DIR}"
  log "logs: ${LOG_DIR}"
  log "FL_PKL_DIR: ${FL_PKL_DIR}"
  log "FL_DEMO_ROUND_TRAIN_N: ${FL_DEMO_ROUND_TRAIN_N}"
  log "Isaac script: ${ISAAC_SCRIPT_WIN}"

  check_tinyiot

  if [[ ! -d "${FL_PKL_DIR}" ]]; then
    log "warning: FL_PKL_DIR does not exist yet: ${FL_PKL_DIR}"
    log "Isaac Sim v6가 실행되면 buffer pkl을 생성할 수 있음."
  fi

  if [[ "${DO_CLEAN}" == "1" ]]; then
    [[ -x ./clean_fl.sh ]] || die "clean_fl.sh 실행 권한이 없음. chmod +x clean_fl.sh 확인."
    log "run clean_fl.sh"
    ./clean_fl.sh | tee "${LOG_DIR}/clean_fl.log"
  else
    log "skip clean_fl.sh"
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
    read -r -p "Isaac Sim v6 실행 확인 후 Enter를 누르면 IN-AE 시작: " _
  else
    log "--no-prompt: start IN-AE without waiting"
  fi

  start_bg "in_ae" python3 -u fl/in_ae_standard.py

  # in_ae 로그까지 포함해서 하나로 추적
  tail -n +1 -F \
    "${LOG_DIR}/dashboard.log" \
    "${LOG_DIR}/mn1.log" \
    "${LOG_DIR}/mn2.log" \
    "${LOG_DIR}/mn3.log" \
    "${LOG_DIR}/in_ae.log" &
  TAIL_PID=$!

  log "IN-AE started."
  log "대시보드: http://localhost:${DASHBOARD_PORT}"
  log "로그 위치: ${LOG_DIR}"

  local in_pid="${PIDS[-1]}"
  wait "${in_pid}" || true

  log "IN-AE process finished."

  if [[ "${EXIT_ON_COMPLETE}" == "1" ]]; then
    log "--exit-on-complete: stop all services"
    return 0
  fi

  cat <<EOF

────────────────────────────────────────────────────────
FL run 완료로 보임.
Dashboard 최종 화면을 확인하려면 브라우저에서 유지:
  http://localhost:${DASHBOARD_PORT}

종료하려면 이 터미널에서 Ctrl+C.
로그 위치:
  ${LOG_DIR}
────────────────────────────────────────────────────────

EOF

  while true; do
    sleep 3600
  done
}

main "$@"
