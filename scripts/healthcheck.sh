#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

check_url() {
  local url="$1"
  local label="$2"

  if curl -fsS "$url" >/dev/null; then
    echo "[ok] $label"
  else
    echo "[fail] $label"
    exit 1
  fi
}

check_post() {
  local url="$1"
  local body="$2"
  local label="$3"

  if curl -fsS -X POST "$url" -H "Content-Type: application/json" -d "$body" >/dev/null; then
    echo "[ok] $label"
  else
    echo "[fail] $label"
    exit 1
  fi
}

echo "Checking Auctus v2 services..."
check_url "http://localhost:8000/" "FastAPI backend"
check_url "http://localhost:9200/" "OpenSearch"
check_url "http://localhost:5601/" "OpenSearch Dashboards"
check_post "http://localhost:8000/search" '{"query":"test","filters":null}' "Search endpoint"

echo "All healthchecks passed."
