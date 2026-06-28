# Come funziona il Simulatore

## Indice
1. [Cos'è il sistema](#cosè-il-sistema)
2. [Flusso completo](#flusso-completo)
3. [Input — UdacitySimulatorConfig](#input--udacitysimulatorconfigedge)
4. [Come viene costruita la strada](#come-viene-costruita-la-strada)
5. [L'agente: la rete neurale che guida](#lagente-la-rete-neurale-che-guida)
6. [Il loop di simulazione](#il-loop-di-simulazione)
7. [Output — UdacitySimulationOutput](#output--udacitysimulationoutput)
8. [Perché la traiettoria sinistra ≠ quella destra](#perché-la-traiettoria-sinistra--quella-destra)
9. [Codice della simulazione commentato riga per riga](#codice-commentato-riga-per-riga)

---

## Cos'è il sistema

È un sistema di **test automatizzato per auto a guida autonoma**.

L'idea: dato un percorso stradale arbitrario (definito da angoli), la rete neurale che guida l'auto riesce a restare in corsia? Il sistema risponde con dati precisi: posizione frame-per-frame, velocità, sterzo, e soprattutto il **Cross-Track Error (XTE)** — cioè quanto l'auto si discosta dal centro della corsia.

```
  TU DEFINISCI LA STRADA        IL SIMULATORE GUIDA          TU ANALIZZI I DATI
  ┌─────────────────────┐       ┌────────────────────┐       ┌──────────────────┐
  │  angles = [0,30,-20]│──────>│  Rete neurale      │──────>│  XTE, velocità,  │
  │  maxTime = 30s      │       │  guida l'auto nel  │       │  sterzo,         │
  │  maxXTE = 3m        │       │  simulatore Unity  │       │  traiettoria     │
  └─────────────────────┘       └────────────────────┘       └──────────────────┘
```

---

## Flusso completo

```
POST /simulate
      │
      ▼
  Genera jobId ──────────────> mette in coda
      │                               │
      │                               ▼
      │                    simulationThread (background)
      │                         │
      │                    1. CustomRoadGenerator.generate(angles)
      │                         │  converte angoli → nodi di controllo
      │                         │  applica spline Catmull-Rom → strada liscia
      │                         │
      │                    2. road.get_string_repr()
      │                         │  serializza la strada per Unity
      │                         │
      │                    3. env.reset(track_string)
      │                         │  invia la strada al simulatore Unity
      │                         │  attende che Unity carichi il percorso
      │                         │
      │                    4. loop frame-per-frame:
      │                         │  - cattura immagine telecamera (320x160px)
      │                         │  - rete neurale predice sterzo e gas
      │                         │  - Unity avanza di un frame
      │                         │  - registra: pos, speed, xte, sterzo, gas
      │                         │  - controlla condizioni di stop
      │                         │
      │                    5. results[jobId] = {status:"done", output:{...}}
      │
      ▼
GET /simulate/{jobId}  ──────> restituisce lo stato o il risultato
```

---

## Input — UdacitySimulatorConfig

```python
UdacitySimulatorConfig(
    angles         = [0, 30, -20, 10],   # lista di angoli in gradi per ogni segmento
    maxTime        = 30,                 # secondi massimi di simulazione
    maxXTE         = 3.0,                # metri: se l'auto esce di tanto, si ferma
    minSpeed       = 10,                 # km/h: velocità minima target
    maxSpeed       = 30,                 # km/h: velocità massima target
    map_size       = 250,                # dimensione della mappa in metri (quadrata)
    segLength      = 25,                 # lunghezza di ogni segmento stradale in metri
    initial_position = (0,0,0, 8.0),    # (x, y, z, larghezza_carreggiata)
)
```

### Gli angoli: come definiscono la strada

Gli angoli sono **assoluti rispetto all'asse X**, non relativi al segmento precedente.

```
angles = [0, 0, 0, 0]          angles = [0, 30, 30, 30]       angles = [0, 45, -45, 45]

  ↑ ↑ ↑ ↑                        ↑                               ↑   ↗
  │ │ │ │  (rettilineo)           │ ↗ ↗ ↗  (curva graduale)      │       ↘ ↗  (zigzag)
  └─┘─┘─┘                        └─┘                             └─┘
```

Ogni angolo dice: "il segmento i-esimo punta in direzione `angle` gradi rispetto all'asse X orizzontale".

Esempio visivo con `angles = [0, 45, 90, 135]`:

```
                    ●  (fine)
                   /
                  / segmento 3 (135°, va in alto-sinistra)
                 /
                ●
                │
                │ segmento 2 (90°, va dritto in su)
                │
                ●
               /
              / segmento 1 (45°, va in diagonale)
             /
            ●──────────●  (inizio → segmento 0 a 0°, va a destra)
```

---

## Come viene costruita la strada

Il processo ha **due fasi**:

### Fase 1 — Nodi di controllo (angoli → punti)

```python
# In CustomRoadGenerator.generate_control_nodes()

cumulative_angle = 0
for i in range(num_control_nodes):
    cumulative_angle += angles[i]          # accumula l'angolo
    new_node = _get_next_node(             # calcola il punto successivo
        prev_node, curr_node,
        cumulative_angle, seg_length
    )
```

Produce una lista di punti grezzi (poligonale):

```
angoli = [0, 30, -20]    →    punti di controllo:

    ●──────●
           │ \
           │  ●──────●
           │
     (angoli a gomito, non smussati)
```

### Fase 2 — Spline Catmull-Rom (punti → curva liscia)

```python
sample_nodes = catmull_rom(control_nodes, num_spline_nodes=20)
```

La spline interpola tra i punti e crea una curva fluida. Ogni segmento tra due nodi di controllo diventa 20 punti interpolati.

```
Prima (nodi grezzi):            Dopo (Catmull-Rom):

    ●──────●                        ●──────●
           │ \                             │  ⌒ \
           │  ●──────●                    │      ●──────●
                                    (curva fluida, realistica)
```

**Questa è la strada che viene mandata al simulatore Unity.**

---

## L'agente: la rete neurale che guida

Il file del modello è `mixed-chauffeur.h5` — una rete neurale convoluzionale (CNN) addestrata su immagini di guida umana.

```
INPUT:                          OUTPUT:
┌─────────────────────┐         ┌─────────────┐
│ Immagine 320×160px  │──────>  │  sterzo     │  (da -1.0 a +1.0)
│ dalla telecamera    │  CNN    │             │
│ frontale dell'auto  │         └─────────────┘
└─────────────────────┘
```

Il throttle (accelerazione) non viene predetto dalla rete: viene calcolato con una formula:

```python
# In SupervisedAgent.predict()
throttle = clip(1.0 - sterzo² - (velocità_attuale / velocità_limite)²)
```

Logica: se sterzo è grande (curva stretta) → throttle basso (frena). Se velocità alta → throttle basso (decelera). Semplice ma efficace.

---

## Il loop di simulazione

```python
# In UdacitySimulator.simulate()

while not self.done:

    # 1. Predici le azioni dalla telecamera
    actions = agent.predict(obs=immagine_corrente, state={"speed": speed})
    #    actions[0][0] = sterzo   (es. -0.3 = curva sinistra)
    #    actions[0][1] = throttle (es.  0.7 = accelera al 70%)

    # 2. Manda le azioni al simulatore Unity, ricevi il frame successivo
    obs, done, info = env.step(actions)
    #    obs  = nuova immagine 320×160
    #    done = True se Unity dice che la simulazione è finita
    #    info = {"speed": ..., "pos": (x,y,z), "cte": ..., "hit": ...}

    # 3. Registra i dati del frame
    speed = info["speed"]
    xte   = info["cte"]           # Cross-Track Error in metri
    simulationOutput.addStats(
        position = info["pos"],   # (x, y, z) dell'auto nel mondo 3D
        speed    = speed,
        xte      = xte,
        steering = actions[0][0],
        throttle = actions[0][1]
    )

    # 4. Controlla le condizioni di stop
    elapsed = time.time() - start
    if elapsed > maxTime:          # tempo scaduto
        self.done = True
    if abs(xte) > maxXTE:          # auto uscita di strada
        self.done = True
    if done:                       # Unity ha detto stop (collisione, ecc.)
        self.done = True
```

### Cos'è il CTE / XTE

```
         ╔══════════════════╗
         ║   CARREGGIATA    ║
         ║                  ║
         ║   ─ ─ ─ ─ ─ ─   ║  ← centro corsia
         ║        ↑         ║
         ║       XTE=1.2m   ║
         ║        │         ║
         ║       🚗         ║  ← posizione auto
         ║                  ║
         ╚══════════════════╝
              larghezza ~8m
```

XTE positivo = auto a destra del centro. XTE negativo = auto a sinistra. Se `|XTE| > maxXTE` la simulazione si interrompe: l'auto è uscita di corsia.

---

## Output — UdacitySimulationOutput

```python
{
  "status": "done",
  "output": {
    "road": [                         # punti della strada generata dalla spline
      [x, y, z, width],              # (x, y, z in metri, width = larghezza carreggiata)
      ...                            # ~100+ punti che descrivono il percorso
    ],
    "positions": [                    # posizione dell'auto frame per frame
      [x, y, z],
      ...
    ],
    "speeds":    [0.0, 5.2, 18.3, ...],   # km/h per ogni frame
    "xtes":      [0.0, 0.1, 0.8, ...],   # metri di scostamento per ogni frame
    "steerings": [0.0, 0.1, -0.3, ...],  # angolo sterzo (-1=sinistra, +1=destra)
    "throttles": [0.5, 0.8, 0.6, ...],   # accelerazione (0=niente, 1=piena)
    "elapsedTime":  28.4,                # secondi totali di simulazione
    "iterations":   1420                 # numero di frame eseguiti
  }
}
```

### Esempio di output visivo

```
XTE nel tempo (auto in curva difficile):

  3.0 ┤                              ← soglia maxXTE (auto esce)
      │                    ╭─────
  2.0 ┤              ╭─────╯
      │         ╭────╯
  1.0 ┤    ╭────╯
      │────╯
  0.0 ┤
      └──────────────────────────>
         0s    10s    20s    28s
                              ↑
                         simulazione interrotta
```

```
Velocità nel tempo:

  30 ┤               ────────────────  ← maxSpeed raggiunta
     │          ────╯
  15 ┤     ────╯
     │────╯
   0 ┤
     └──────────────────────────────>
       partenza      regime costante
```

---

## Perché la traiettoria sinistra ≠ quella destra

**Risposta breve: sono due cose diverse.**

| Pannello sinistro (anteprima) | Pannello destro (risultato) |
|---|---|
| Generata dal **frontend** in JavaScript | Generata dal **simulatore** in Python/Unity |
| Algoritmo semplice: `x += cos(angolo) * segLen` | Algoritmo completo: nodi di controllo + spline Catmull-Rom |
| Non cumula gli angoli correttamente | Cumula gli angoli e applica la spline fluida |
| Solo a scopo visivo orientativo | Questa è la **strada reale** percorsa dall'auto |

Il frontend fa un'approssimazione rapida per dare un'idea visiva, ma il vero percorso è molto diverso perché:
1. Il generatore accumula l'angolo progressivamente (`cumulative_angle += angle[i]`)
2. La spline Catmull-Rom arrotonda tutti gli spigoli
3. Il punto di partenza nel simulatore non è `(0,0)` ma dipende da `initial_position`

Per rendere l'anteprima accurata bisognerebbe replicare esattamente la logica di `CustomRoadGenerator` in JavaScript.

---

## Codice commentato riga per riga

```python
def simulate(self, simulator_config: UdacitySimulatorConfig) -> UdacitySimulationOutput:

    # ── Configura i limiti di velocità dell'agente ─────────────────────────
    self.agent.setSpeedLimits(
        minSpeed=simulator_config.minSpeed,   # es. 10 km/h
        maxSpeed=simulator_config.maxSpeed    # es. 30 km/h
    )

    # ── Crea il generatore di strade ───────────────────────────────────────
    test_generator = CustomRoadGenerator(
        map_size=simulator_config.map_size,           # dimensione mappa (default 250m)
        num_control_nodes=len(simulator_config.angles), # quanti segmenti ha la strada
        seg_length=simulator_config.segLength         # lunghezza di ogni segmento (default 25m)
    )

    try:
        simulationOutput = UdacitySimulationOutput()  # accumulatore dei risultati

        angles = simulator_config.angles              # es. [0, 30, -20, 10]

        # ── Genera la strada dai parametri ────────────────────────────────
        road = test_generator.generate(
            starting_pos=simulator_config.initial_position,  # punto di partenza nel mondo
            angles=angles,                                    # angoli → nodi di controllo → spline
            simulator_name=UDACITY_SIM_NAME                  # determina il formato della strada
        )
        # 'road' ora contiene ~100 punti 3D che descrivono la curva

        # Salva la geometria della strada nell'output
        simulationOutput.road = road.get_concrete_representation(to_plot=True)
        # → lista di tuple (x, y, z, width) usata per il grafico del percorso

        # ── Serializza la strada per Unity ────────────────────────────────
        waypoints = road.get_string_repr()
        # Unity riceve la strada come stringa di coordinate, es:
        # "10.0,0.0,0.0|14.2,3.1,0.0|18.5,8.3,0.0|..."

        # ── Carica il percorso nel simulatore ─────────────────────────────
        obs = self.env.reset(skip_generation=False, track_string=waypoints)
        # - invia 'waypoints' a Unity via socket
        # - Unity genera fisicamente la strada nel mondo 3D
        # - ritorna la prima immagine dalla telecamera dell'auto (320×160 px)

        speed = 0           # velocità iniziale (auto ferma)
        self.done = False
        self.loopStartTime = time.time()
        iterations = 0

        # ── Loop principale: un'iterazione = un frame ──────────────────────
        while not self.done:

            # 1. La rete neurale guarda l'immagine e decide cosa fare
            actions = self.agent.predict(
                obs=obs,                                        # immagine 320×160 corrente
                state=dict(speed=speed, simulator_name=UDACITY_SIM_NAME)
            )
            # actions = [[sterzo, throttle]]
            # sterzo:   -1.0 (max sinistra) .. 0.0 (dritto) .. +1.0 (max destra)
            # throttle:  0.0 (fermo)        ..                   1.0 (gas pieno)

            # 2. Clipping: assicura che le azioni siano nel range fisico possibile
            if isinstance(self.env.action_space, gym.spaces.Box):
                actions = np.clip(
                    actions,
                    self.env.action_space.low,    # sterzo min, throttle min
                    self.env.action_space.high    # sterzo max, throttle max
                )

            # 3. Manda le azioni a Unity e ricevi il frame successivo
            obs, done, info = self.env.step(actions)
            # obs  → nuova immagine 320×160 (il "prossimo frame" da analizzare)
            # done → True se Unity ha rilevato una condizione di fine (collisione, ecc.)
            # info → dizionario con:
            #   "speed" → velocità in km/h
            #   "pos"   → (x, y, z) posizione nel mondo
            #   "cte"   → Cross-Track Error in metri (distanza dal centro corsia)
            #   "hit"   → "none" oppure tipo di collisione

            speed = info.get("speed", 0.0)        # aggiorna velocità corrente

            # 4. Calcola l'XTE, eventualmente cappato al massimo
            xte = info['cte']
            if CAP_XTE and abs(xte) > MAX_XTE:
                # Se l'auto è uscita di molto, non registrare valori eccessivi
                xte = MAX_XTE if xte > 0 else -MAX_XTE

            # 5. Salva i dati del frame corrente
            simulationOutput.addStats(
                position=info['pos'],       # (x, y, z) → traiettoria dell'auto
                speed=speed,                # km/h
                xte=xte,                    # metri di scostamento dalla corsia
                steering=actions[0][0],     # angolo di sterzo usato
                throttle=actions[0][1]      # gas usato
            )

            # 6. Controlla le condizioni di stop
            elapsed = time.time() - self.loopStartTime
            if elapsed > simulator_config.maxTime:    # tempo scaduto
                self.done = True
            if abs(info["cte"]) > simulator_config.maxXTE:  # auto fuori corsia
                self.done = True
            if done:                                  # Unity ha detto stop
                self.done = True

            iterations += 1

        # ── Fine del loop ─────────────────────────────────────────────────
        elapsedTime = time.time() - self.loopStartTime

        # Resetta il simulatore (prepara Unity per la simulazione successiva)
        self.env.reset(skip_generation=False, track_string=waypoints)

        # Aggiunge statistiche globali all'output
        simulationOutput.elapsedTime = elapsedTime   # secondi totali
        simulationOutput.iterations = iterations     # quanti frame sono stati eseguiti

    except Exception as e:
        raise e

    return simulationOutput
    # → contiene: road, positions, speeds, xtes, steerings, throttles, elapsedTime, iterations
```

---

## Parametri di default (config.py)

| Parametro | Default | Significato |
|---|---|---|
| `ROAD_WIDTH` | 8.0 m | Larghezza della carreggiata |
| `SEG_LENGTH` | 25 m | Lunghezza di ogni segmento |
| `NUM_SAMPLED_POINTS` | 100 | Punti della spline per visualizzazione |
| `MAX_XTE` | 3.0 m | Soglia globale XTE (hard cap) |
| `CAP_XTE` | True | Cappa l'XTE a MAX_XTE nei dati |
| `STEERING_CORRECTION` | 1 | Moltiplicatore sterzo (1 = nessuna correzione) |
| `MAP_SIZE` | 250 m | Dimensione della mappa quadrata |
| `INPUT_SHAPE` | (160, 320, 3) | Dimensione immagine telecamera |
| `CROP_UDACITY` | [60, -25] | Crop verticale dell'immagine (rimuove cielo e cofano) |
