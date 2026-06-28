#!/usr/bin/env bash
# Avvia N simulatori in parallelo, mantenendo N e NUM_WORKERS allineati.
#
#   ./run_parallel.sh          # 4 worker (default)
#   ./run_parallel.sh 8        # 8 worker
#   ./run_parallel.sh 4 -d     # in background (detached)
#
# Dopo l'avvio, lancia la pipeline nello STESSO ambiente con:
#   NUM_WORKERS=<N> python -m pipeline.orchestrator ...
# oppure esporta la variabile:  export NUM_WORKERS=<N>
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-4}"
shift || true
EXTRA_ARGS=("$@")   # es. -d per detached

echo ">> Genero docker-compose.parallel.yml con $N worker..."
python3 gen_parallel_compose.py "$N"

echo ">> Avvio $N container (porte 8000-$((8000 + N - 1)))..."
docker compose -f docker-compose.parallel.yml up --build "${EXTRA_ARGS[@]}"

cat <<EOF

------------------------------------------------------------
Container avviati. Per lanciare i test sul pool, usa NUM_WORKERS=$N, es:

    export NUM_WORKERS=$N
    python -m pipeline.orchestrator          # o il tuo entrypoint

Per fermare:  docker compose -f docker-compose.parallel.yml down
------------------------------------------------------------
EOF
