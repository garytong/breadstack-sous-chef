#!/bin/sh
# The model image's entrypoint (lab/model-image/Dockerfile): Ollama's server,
# and a preload of the model as soon as the server answers. The preload is
# what makes the pre-warm work: an instance kept running
# (gpu_model_min_instances = 1) holds the model in the GPU before the first
# question. A cold start's first question also waits for one load instead
# of queueing behind it. Cloud Run's startup probe (GET /) passes once the
# server listens; a request that arrives during the load waits inside
# Ollama. OLLAMA_KEEP_ALIVE=-1 keeps the model loaded until the instance
# stops.
#
# The "loaded in N s" line is the model-load part of a cold start, which the
# runbook reads from the service's logs.
#
# sh is PID 1 here: it passes Cloud Run's SIGTERM on to the server and
# exits with the server's status.

/bin/ollama serve &
server=$!
trap 'kill -TERM "$server" 2>/dev/null' TERM INT

(
  i=0
  until /bin/ollama list >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 120 ]; then
      echo "sous-chef: the server did not answer within 120 s; no preload" >&2
      exit 0
    fi
    sleep 1
  done
  started=$(date +%s)
  # An empty prompt is a load-only request: nothing is generated. No
  # --keepalive: the CLI wants a Go duration, and the server's
  # OLLAMA_KEEP_ALIVE=-1 (keep it forever) applies anyway.
  if /bin/ollama run "$SOUS_CHEF_MODEL" "" </dev/null >/dev/null 2>&1; then
    echo "sous-chef: $SOUS_CHEF_MODEL loaded in $(($(date +%s) - started)) s"
  else
    echo "sous-chef: the preload of $SOUS_CHEF_MODEL failed; the first question loads it" >&2
  fi
) &

wait "$server"
status=$?
# A trapped signal interrupts the wait; wait again for the server to exit.
if kill -0 "$server" 2>/dev/null; then
  wait "$server"
  status=$?
fi
exit "$status"
