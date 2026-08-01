# Esecuzione parallela dei simulatori

## Il problema
`docker compose up` avvia **un solo** container simulatore. Dentro
`SimulatorServer.py` c'è un unico thread con una sola istanza Unity, che pesca i
job da una coda **uno alla volta**. Quindi 20–50 simulazioni venivano eseguite in
sequenza: con il vecchio client il tempo di attesa arrivava a `N × 90s`
(75 minuti per N=50), dando l'impressione di un blocco/loop mentre in realtà stava
solo smaltendo la coda.

## La soluzione
Il parallelismo si ottiene avviando **più container** (più istanze Unity), ognuno
sulla sua porta, e distribuendo i job sul pool. Con `W` worker il throughput è
circa `W×`. Ogni container resta sequenziale al suo interno: è corretto così, un
processo Unity non è thread-safe.

Modifiche principali:
- `scenarios/lane_keeping/config.py` → `run_simulation` ora distribuisce i job su
  un pool di endpoint tramite una coda condivisa (bilanciamento dinamico), con
  **timeout per-job** (`DEFAULT_TIMEOUT`, 90s) invece del vecchio deadline globale
  `N × 90s`. Un container bloccato su Unity fa fallire **solo** il suo job in ~90s,
  non congela più l'intero batch. I worker non raggiungibili vengono scartati a
  runtime con un health-check su `/health`.
- `gen_parallel_compose.py` → genera un `docker-compose.parallel.yml` con N servizi
  su porte `8000 .. 8000+N-1`. L'immagine viene costruita una sola volta e
  riutilizzata da tutti i worker.
- `run_parallel.sh` → launcher che tiene allineati numero di container e
  `NUM_WORKERS`.

## Uso

### 1. Avvia i container (default 4 worker, porte 8000–8003)
```bash
cd opensbt-core
./run_parallel.sh 4          # oppure: ./run_parallel.sh 8 per 8 worker
```
Equivalente manuale:
```bash
python3 gen_parallel_compose.py 4
docker compose -f docker-compose.parallel.yml up --build
```

### 2. Lancia i test usando lo STESSO numero di worker
```bash
export NUM_WORKERS=4
python -m pipeline.orchestrator    # o il tuo entrypoint
```

Il client costruisce il pool da (in ordine di priorità):
1. `SIMULATOR_URLS` — lista esplicita, es.
   `export SIMULATOR_URLS="http://localhost:8000,http://localhost:8001"`
2. `NUM_WORKERS` (+ `SIMULATOR_BASE_PORT`, default 8000) → `localhost:8000..8000+N-1`
3. default: 4 worker.

**Importante:** `NUM_WORKERS` (client) deve combaciare con il numero di container
avviati. Se non combaciano, i worker mancanti vengono semplicemente ignorati con
un warning — non è un errore fatale, gira solo con quelli disponibili.

### Fermare
```bash
docker compose -f docker-compose.parallel.yml down
```

## Quanti worker?
Ogni container Unity gira sotto emulazione amd64 con rendering software: è pesante
su CPU e RAM. Regola pratica: `W ≈ core_fisici / 2`, tenendo d'occhio la RAM.
Parti da 4 e aumenta solo se la macchina regge (`docker stats` per monitorare).

## Retro-compatibilità
- `LaneKeepingScenario()` → pool da env (default 4).
- `LaneKeepingScenario(simulator_url="http://localhost:8001")` → singolo container
  (comportamento Step D invariato, ma ora con timeout per-job).
- `LaneKeepingScenario(simulator_urls=[...])` → pool esplicito.
- Il vecchio `docker-compose.yml` a container singolo continua a funzionare: il
  client scarta le porte non attive e gira su 1 worker.

## Miglioramenti all'analisi (QoI e rare failure)

Oltre alla parallelizzazione sono stati corretti alcuni problemi che emergevano
solo una volta che le simulazioni arrivavano a completarsi:

- **Scenari non validi esclusi.** Alcune simulazioni terminano in 1 solo step con
  XTE ≈ 0 (episodio abortito) e la QoI le premiava come "sicure"; inoltre il
  campionamento poteva generare parametri incoerenti (`min_speed > max_speed`).
  Questi casi non sono scenari reali: ora vengono **esclusi** dall'analisi (margine
  NaN) e i tassi/rare failure si calcolano solo sugli scenari validi. I conteggi
  sono in `PipelineResult.n_invalid`/`n_degenerate` e stampati dal runner.
- **Rare failure con risoluzione temporale.** Quando l'auto esce di corsia l'XTE
  satura al massimo, quindi decine di fallimenti hanno lo stesso margine (~−0.5) e
  la selezione "bottom-k%" era arbitraria. `find_rare_failures` ora accetta un
  `tiebreak` (tempo di sopravvivenza in step): a parità di margine sono più rari i
  crash che escono **prima**. Il margine resta il criterio primario.

## Runner con report chiaro

`scripts/run_lanekeeping.py` esegue la pipeline e stampa un report leggibile:

```bash
python scripts/run_lanekeeping.py --n 50 --workers 4            # spazio esteso completo
python scripts/run_lanekeeping.py --n 50 --preset realistic     # ODD realistico proposto
python scripts/run_lanekeeping.py --n 30 --seed 7 --quiet       # senza progresso per-job
```

Il preset `realistic` campiona un ODD plausibile (angoli 0–45°, velocità min/max
NON sovrapposte così da non generare mai `min_speed > max_speed`, segmenti più
lunghi). Non serve ad abbassare artificiosamente il tasso: serve a misurare i
fallimenti in condizioni realistiche. Il report mostra failure rate e rare failure
rate calcolati **solo sugli scenari validi**, gli scenari non validi esclusi,
la distribuzione dei margini e la tabella degli scenari più critici (con numero
di step di sopravvivenza). Con `--max-speed X` si può forzare il tetto di velocità
su qualsiasi preset per studiarne l'effetto.
