#!/usr/bin/env python3
"""
Genera un compose con N worker per il braccio C2.

Script **nuovo**, affiancato a `gen_parallel_compose.py` invece che modificarlo:
quel file è preesistente e serve al braccio C1.

Perché non basta un file di override
------------------------------------
`docker-compose.parallel.yml` ha i servizi `simulator-0..N`, quindi un override
dovrebbe conoscerne il numero in anticipo. Generarli è più semplice che
sovrascriverli, ed è la stessa scelta fatta dallo script originale.

Uso:
    python gen_c2_compose.py            # 4 worker (porte 8000-8003)
    python gen_c2_compose.py 2 --speed-scale 0.6

Poi:
    docker compose -f docker-compose.c2.parallel.yml up --build
    # e lato client:  NUM_WORKERS=4
"""
from __future__ import annotations

import argparse
import os

IMAGE_TAG = "paologinefra/se2rp-simulator"
BASE_PORT = 8000

HEADER = """\
# --- GENERATO da gen_c2_compose.py -- non modificare a mano ------------------
# {n} worker del braccio C2 su porte host {base}..{last} (container: sempre 8000).
#
# C2 = controller state-based condiviso con MetaDrive e CARLA. Isola l'effetto
# del simulatore tenendo costante il controller.
#
# Avvio:  docker compose -f {out} up --build
# Client: NUM_WORKERS={n}
#
# ⚠️ Throughput contro fedeltà. Misurato su Udacity: 20.8 Hz con 1 container,
# 8.5 Hz con 4, perché le istanze Unity si contendono la CPU. Il controller
# compensa (il limite di sterzata è in unità/secondo, convertito col dt reale),
# ma il control rate resta una variabile confondente rispetto a MetaDrive e
# CARLA, dove è esatto per costruzione. Per la campagna di confronto conviene
# 1 worker; per le prove esplorative, 4.
services:
"""

SERVICE = """\
  simulator-{i}:
    platform: linux/amd64
{build}    image: {image}
    container_name: se2rp-c2-{i}
    # ⚠️ Il `command` sostituisce l'INTERO CMD del Dockerfile, che non si limita
    # a lanciare uvicorn: avvia prima Xvfb ed esporta DISPLAY=:99.
    #
    # Unity è un'applicazione grafica e pretende un display anche in headless:
    # senza Xvfb il processo parte ma non riesce a creare il contesto GL, non si
    # connette al server socket.io, e il container resta per sempre su
    # "sleep...and repeat to connect" senza un errore esplicito.
    #
    # Quindi qui il bootstrap va replicato per intero, cambiando solo il modulo
    # uvicorn. Se un giorno cambia il CMD del Dockerfile, va aggiornato anche qui.
    #
    # `$$` e non `$`: in un compose `${{VAR}}` lo espande Docker sull'HOST, dove
    # XVFB_RESOLUTION non esiste. `$$` lo passa letterale a /bin/sh, che lo
    # risolve dentro il container contro la ENV del Dockerfile.
    command: '/bin/sh -c "rm -f /tmp/.X99-lock; Xvfb :99 -screen 0 $${{XVFB_RESOLUTION:-320x240x24}} -ac +extension GLX +extension RANDR +render -noreset & sleep 3 && export DISPLAY=:99 && uvicorn Simulator.c2.server:app --host 0.0.0.0 --port 8000"'
    ports:
      - "{host_port}:8000"
    environment:
      - LK_SPEED_SCALE={speed_scale}
      - LK_STEERING_SIGN={steering_sign}
      - LK_OBS_LATENCY={obs_latency}
      - LK_OBS_LAG_TAU={obs_lag_tau}
      - LK_STEER_NOISE={steer_noise}
    stdin_open: true
    tty: true
    restart: unless-stopped
{depends}"""

BUILD_BLOCK = """\
    # Solo questo servizio costruisce: l'immagine viene riusata dagli altri.
    build:
      context: .
      dockerfile: Simulator/Dockerfile
"""

DEPENDS_BLOCK = """\
    depends_on:
      - simulator-0
"""


def render(n: int, out: str, *, speed_scale: float, steering_sign: float,
           obs_latency: int, obs_lag_tau: float, steer_noise: float) -> str:
    n = max(1, n)
    parts = [HEADER.format(n=n, base=BASE_PORT, last=BASE_PORT + n - 1, out=out)]
    for i in range(n):
        parts.append(SERVICE.format(
            i=i,
            image=IMAGE_TAG,
            host_port=BASE_PORT + i,
            build=BUILD_BLOCK if i == 0 else "",
            depends="" if i == 0 else DEPENDS_BLOCK,
            speed_scale=speed_scale,
            steering_sign=steering_sign,
            obs_latency=obs_latency,
            obs_lag_tau=obs_lag_tau,
            steer_noise=steer_noise,
        ))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", nargs="?", type=int,
                    default=int(os.getenv("NUM_WORKERS", "4")),
                    help="numero di worker (default: NUM_WORKERS env o 4)")
    ap.add_argument("--out", default="docker-compose.c2.parallel.yml")
    ap.add_argument("--speed-scale", type=float, default=1.0, dest="speed_scale",
                    help="taratura del punto operativo (§7.1)")
    ap.add_argument("--steering-sign", type=float, default=-1.0, dest="steering_sign",
                    help="gate V3; -1.0 è il valore misurato corretto per Unity")
    ap.add_argument("--obs-latency", type=int, default=0, dest="obs_latency")
    ap.add_argument("--obs-lag-tau", type=float, default=0.0, dest="obs_lag_tau")
    ap.add_argument("--steer-noise", type=float, default=0.0, dest="steer_noise")
    args = ap.parse_args()

    text = render(args.n, args.out, speed_scale=args.speed_scale,
                  steering_sign=args.steering_sign, obs_latency=args.obs_latency,
                  obs_lag_tau=args.obs_lag_tau, steer_noise=args.steer_noise)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print(f"Scritto {path} con {args.n} worker "
          f"(porte {BASE_PORT}-{BASE_PORT + args.n - 1}).")
    print(f"Avvia con:  docker compose -f {args.out} up --build")
    print(f"E lancia il client con:  NUM_WORKERS={args.n}")


if __name__ == "__main__":
    main()
