#!/usr/bin/env bash
# Run a description_eval stage inside the arq-worker container.
#
# The worker mounts the main checkout's backend at /app (it has the InvenioRDM
# connector, portkey, minio and sentence-transformers). This copies the eval
# package and its data in, runs the stage against /app's code, and copies the
# data directory (gold, collected, cache, results) back out.
#
#   ./run_in_worker.sh gold --records 9wnjp-kez15
#   ./run_in_worker.sh compare
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-auctus-arq-worker}"
REMOTE=/tmp/description_eval_pkg

mkdir -p "$HERE/data"
docker exec "$CONTAINER" rm -rf "$REMOTE"
docker exec "$CONTAINER" mkdir -p "$REMOTE/eval"
docker exec "$CONTAINER" touch "$REMOTE/eval/__init__.py"
docker cp "$HERE" "$CONTAINER:$REMOTE/eval/description_eval"

# Copy the data back even when the stage fails, so partial gold and the LLM cache
# (paid-for calls) are never lost to the next run's cleanup.
copy_back() {
  rm -rf "$HERE/data.incoming"
  docker cp "$CONTAINER:$REMOTE/eval/description_eval/data" "$HERE/data.incoming"
  rm -rf "$HERE/data" && mv "$HERE/data.incoming" "$HERE/data"
}
trap copy_back EXIT

# /app first so `storage` and `crawlers` resolve; the eval package from $REMOTE.
docker exec -w "$REMOTE" \
  -e PYTHONPATH="$REMOTE:/app" \
  -e DESCRIPTION_EVAL_DATA="$REMOTE/eval/description_eval/data" \
  -e OPENSEARCH_HOST=opensearch \
  "$CONTAINER" python -m eval.description_eval.run "$@"
