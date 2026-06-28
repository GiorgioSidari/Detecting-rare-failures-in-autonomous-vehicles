#!/usr/bin/env python3
"""
Genera un docker-compose per eseguire N simulatori (container Unity) in parallelo.

Ogni container e' un SimulatorServer identico, sequenziale al suo interno; il
parallelismo si ottiene avviandone N su porte host distinte (8000, 8001, ...).
Il client (scenarios/lane_keeping/config.py) distribuisce i job sul pool.

Uso:
    python gen_parallel_compose.py            # default: N = NUM_WORKERS env o 4
    python gen_parallel_compose.py 8          # genera 8 worker (porte 8000-8007)
    python gen_parallel_compose.py 4 --out docker-compose.parallel.yml

IMPORTANTE: il numero N qui DEVE combaciare con NUM_WORKERS usato dal client.
Il launcher run_parallel.sh li tiene automaticamente allineati.
"""
from __future__ import annotations

import argparse
import os

# Un solo servizio "costruisce" l'immagine; gli altri la riusano (build una volta).
IMAGE_TAG = "paologinefra/se2rp-simulator"
DNN_MODEL = "./Simulator/SelfDrivingModels/mixed-chauffeur.h5"
BASE_PORT = 8000

HEADER = """\
# --- GENERATO da gen_parallel_compose.py -- non modificare a mano ------------
# {n} simulatori in parallelo su porte host {base}..{last} (container: sempre 8000).
# Avvio:  docker compose -f {out} up --build
# Il client deve usare lo stesso numero di worker: NUM_WORKERS={n}
services:
"""

SERVICE_FIRST = """\
  simulator-{i}:
    platform: linux/amd64
    # Solo questo servizio ha 'build': l'immagine viene costruita una volta
    # e riutilizzata da tutti gli altri worker (stesso tag {image}).
    build:
      context: .
      dockerfile: Simulator/Dockerfile
    image: {image}
    container_name: se2rp-simulator-{i}
    ports:
      - "{host_port}:8000"
    environment:
      - DNN_MODEL_PATH={model}
    stdin_open: true
    tty: true
    restart: unless-stopped
"""

SERVICE_REST = """\
  simulator-{i}:
    platform: linux/amd64
    image: {image}
    container_name: se2rp-simulator-{i}
    ports:
      - "{host_port}:8000"
    environment:
      - DNN_MODEL_PATH={model}
    stdin_open: true
    tty: true
    restart: unless-stopped
    depends_on:
      - simulator-0
"""


def render(n: int, out: str) -> str:
    n = max(1, n)
    parts = [HEADER.format(n=n, base=BASE_PORT, last=BASE_PORT + n - 1, out=out)]
    for i in range(n):
        tmpl = SERVICE_FIRST if i == 0 else SERVICE_REST
        parts.append(tmpl.format(
            i=i,
            image=IMAGE_TAG,
            host_port=BASE_PORT + i,
            model=DNN_MODEL,
        ))
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera compose per N simulatori paralleli.")
    parser.add_argument(
        "n", nargs="?", type=int,
        default=int(os.getenv("NUM_WORKERS", "4")),
        help="numero di worker/container (default: NUM_WORKERS env o 4)",
    )
    parser.add_argument(
        "--out", default="docker-compose.parallel.yml",
        help="file di output (default: docker-compose.parallel.yml)",
    )
    args = parser.parse_args()

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(args.n, args.out))
    n = max(1, args.n)
    print(f"Scritto {path} con {n} worker (porte {BASE_PORT}-{BASE_PORT + n - 1}).")
    print(f"Avvia con:  docker compose -f {args.out} up --build")
    print(f"E lancia il client con:  NUM_WORKERS={n}")


if __name__ == "__main__":
    main()
