# Risultati della sperimentazione

Confronto fra algoritmi di ricerca di fallimenti rari su due simulatori.
Tutti i numeri vengono dai file in `results/`; provenienza e comandi sono
nell'Appendice D.

| sezione | contenuto |
|---|---|
| §1 | risultati su **Udacity** |
| §2 | risultati su **MetaDrive** |
| §3 | **caratteristiche tecniche** dei due simulatori, condizioni di arresto di una simulazione, confronto diretto |
| §4 | **LHS contro random**: cosa ci si aspettava, cosa è successo |
| App. A–D | variabili e metriche, statistica, geometria dei fallimenti, riproducibilità |

---

## 0. In breve

**Che cosa distinguono le etichette.** `[lhs]` / `[random]` è il **disegno
campionario**, cioè come vengono estratti i punti; `+randacq` spegne
l'**acquisizione**, cioè il batch da simulare viene pescato a caso dal pool
invece che per entropia massima (stessa regione di campionamento, apprendimento
spento — §2.3). Dentro `active_boundary` ci sono quattro decisioni di
campionamento e l'etichetta del disegno ne governa due, entrambe attraverso lo
stesso oggetto `self.sampler`:

| # | decisione | costa simulazioni? | `[lhs]` | `[random]` | `[…+randacq]` |
|---|---|---|---|---|---|
| 1 | design iniziale, 40 punti | **sì, 40** | LHS | uniforme i.i.d. | come il disegno |
| 2 | pool di candidati, 4000 a iterazione | no, solo modello | LHS | uniforme i.i.d. | come il disegno |
| 3 | scelta del batch dal pool, 5 × 16 punti | **sì, 80** | entropia + diversità | entropia + diversità | **a caso dal pool** |
| 4 | integrazione ODD per `P(fail)`, 8000 punti | no, solo modello | LHS | **LHS** | LHS |

Due letture da evitare. `[random]` **non** significa «random solo all'inizio»: il
campionatore governa anche il pool di ogni iterazione e il batch è un
sottoinsieme del pool, quindi tutti e 120 i punti simulati discendono da
estrazioni non stratificate. E `[random]` **non** è privo di LHS: la riga 4 resta
stratificata in ogni configurazione, perché è la quadratura Monte Carlo che stima
`P(fail)` e il rumore di integrazione non deve contaminare il confronto — non
costa simulazioni e non entra in `rari/100`. Nella famiglia `cross_entropy` il
campionatore governa i batch della discesa CE e la miscela difensiva finale (via
`from_dists`), con un'inversione: l'originale `rare_event.py` estraeva già
i.i.d., quindi lì la configurazione *nuova* è quella stratificata.

**La risposta**, su MetaDrive, ODD largo, 12 seed (`cmp_md12_full`) — classifica
completa con conteggi e budget speso in §2.1:

```
configurazione                   rari/100   pct ODD
active_boundary[random]             41.89       1.7
active_boundary[lhs]                41.59       1.6
active_boundary[random+randacq]     11.33       0.8
active_boundary[lhs+randacq]        10.19       0.5
cross_entropy[lhs]                   9.46      21.6
cross_entropy[random]                6.02      27.0
plain_sampling[lhs]                  1.43       9.3
plain_sampling[random]               1.00      13.7
```

1. **Active boundary con acquisizione attiva è primo con un margine di più di un
   ordine di grandezza**: 29× il pavimento sul disegno LHS, 42× sul disegno
   random, in tutte le campagne e su entrambi i simulatori.
2. **Il vantaggio si decompone**: ~**78%** apprendimento attivo, ~**22%** regione
   di campionamento, misurato spegnendo l'acquisizione e lasciando tutto il
   resto identico (§2.3).
3. **La cross-entropy si colloca sul livello di active boundary con
   l'apprendimento spento**, ma trova una popolazione di fallimenti diversa:
   percentile ODD mediano 21.6 contro 0.5 (§2.4).
4. **LHS non batte sempre il campionamento casuale**, e §4 spiega perché la letteratura si aspettava il contrario.

---

# §1. Risultati su Udacity

**Backend A.** Unity dentro Docker, DNN `mixed-chauffeur.h5` che guida
dall'immagine. ODD ristretto (`--max-angle 8 --max-speed 9.6 --max-seg 12`),
punto operativo non tarato. Su questo ODD **il DNN fallisce nel 4.3% dei casi**:
è il numero che rende leggibile il confronto fra sistemi sotto test di §3.5.

## 1.1 La classifica delle sei configurazioni — `cmp_rare`, 6 seed

```
#  configurazione            sim   fall   rari  rari/100    peggiore  pct ODD  escl
1  active_boundary[random]   641    184    151     23.56      -0.521      1.7    68
2  active_boundary[lhs]      660    226    154     23.33      -0.579      2.4    99
3  cross_entropy[random]     583     59     22      3.77      -0.516     27.4     5
4  plain_sampling[lhs]       690     30      8      1.16      -0.514     28.8    14
5  plain_sampling[random]    688     33      7      1.02      -0.552     25.5    14
6  cross_entropy[lhs]        654     92      2      0.31      -0.567     38.5     5
```

**L'ordinamento replica quello di MetaDrive**: active boundary primo con un
fattore 20 sul pavimento (23.33 contro 1.16), fallimenti al **1.7°–2.4°
percentile** di verosimiglianza operativa contro il 25°–38° di tutti gli altri, e
disegni campionari indistinguibili (23.33 contro 23.56). L'unica differenza è
interna alla cross-entropy: qui `ce[random]` (3.77) supera `ce[lhs]` (0.31), che
scende **sotto il pavimento**.

**Statuto della campagna: descrittivo.** È anteriore alla registrazione della
provenienza per seed nel `.npz`, quindi nessun test appaiato è ricostruibile a
posteriori. I numeri sono veri, il test non esiste.

## 1.2 LHS contro random su active boundary — `cmp_ab12`, 12 seed

Campagna dedicata alle sole due configurazioni `active_boundary`, con provenienza
per seed, quindi **testabile**:

| metrica | LHS | random | vittorie LHS  |
|---|---|---|---------------|
| fallimenti trovati | 23.7 | 19.3 | 7/12          |


---

# §2. Risultati su MetaDrive

**Backend B.** MetaDrive in-process, headless, controllore laterale su stato
esatto. Due campagne principali, con punti operativi tarati per bisezione.

## 2.1 ODD largo, controllore ideale — `cmp_md12_full`, 12 seed

`speed_scale = 0.3625`. Taglio di rarità calibrato su 200 000 estrazioni dall'ODD.

| # | configurazione | sim | fall | rari | **rari/100** | peggiore | pct ODD |
|---|---|---|---|---|---|---|---|
| 1 | `active_boundary[random]` | 1406 | 732 | 589 | **41.89** | −0.961 | 1.7 |
| 2 | `active_boundary[lhs]` | 1409 | 725 | 586 | **41.59** | −0.964 | 1.6 |
| 3 | `active_boundary[random+randacq]` | 1359 | 180 | 154 | **11.33** | −0.961 | 0.8 |
| 4 | `active_boundary[lhs+randacq]` | 1344 | 152 | 137 | **10.19** | −0.961 | 0.5 |
| 5 | `cross_entropy[lhs]` | 1057 | 447 | 100 | **9.46** | −0.345 | 21.6 |
| 6 | `cross_entropy[random]` | 1179 | 371 | 71 | **6.02** | −0.252 | 27.0 |
| 7 | `plain_sampling[lhs]` | 1401 | 39 | 20 | **1.43** | −0.357 | 9.3 |
| 8 | `plain_sampling[random]` | 1395 | 33 | 14 | **1.00** | −0.345 | 13.7 |

`sim` non è 1440 per tutti: la cross-entropy si ferma prima e le run invalide
(3–5%) sono escluse. È il motivo per cui la metrica normalizza sul budget
realmente speso. Rapporti a disegno appaiato: `ab[lhs]`/`plain[lhs]` = **29.1×**,
`ab[random]`/`plain[random]` = **41.9×**.

### L'ordinamento regge

Ogni coppia di configurazioni è confrontata **dentro ogni seed** (le differenze
dei 12 valori di `rari/100`, Wilcoxon signed-rank), e i 28 p risultanti sono
corretti con **Holm**, perché fra 28 confronti simultanei un falso positivo
sarebbe quasi garantito. Esito: **16 coppie su 28 sopravvivono, e sono tutte fra
livelli diversi** — nessuna coppia interna a un livello si separa, quindi le
differenze fra i due disegni campionari e dentro il gruppo intermedio non sono
dimostrate. Le 16 stanno tutte a p Holm = 0.014, che è il minimo ottenibile con
12 seed e 28 confronti: la dimensione degli effetti va letta nei rapporti, non
lì. Dettaglio dei confronti e delle quattro coppie che restano al di qua della
soglia in Appendice B.3.

Accanto alla matrice esplorativa gira una **sequenza pre-registrata** di cinque
ipotesi ordinate, ciascuna al pieno α = 0.05, fermandosi alla prima che non
rigetta. Il piano è in `docs/preregistrazione_classifica.json`, committato
**prima** di generare i dati:

```
H1  active_boundary[lhs] > plain_sampling[lhs]    41.59 vs  1.42   12/12   p = 0.00024   RIGETTA
H2  active_boundary[lhs] > cross_entropy[lhs]     41.59 vs 10.53   11/12   p = 0.00049   RIGETTA
H3  active_boundary[rnd] > cross_entropy[rnd]     41.88 vs  6.44   12/12   p = 0.00024   RIGETTA
H4  cross_entropy[lhs]  <> plain_sampling[lhs]    10.53 vs  1.42    9/12   p = 0.034     RIGETTA
H5  active_boundary[lhs] <> active_boundary[rnd]  41.59 vs 41.88    7/12   p = 0.791     si ferma
```

`H5` era ultima per costruzione: l'attesa era un pareggio, e un'attesa nulla
messa prima bloccherebbe tutto ciò che segue.

> I test appaiati usano le medie **per seed** di `rare/100`, che differiscono
> leggermente dagli aggregati della classifica (`cross_entropy[lhs]`: 9.46
> aggregato, 10.53 per seed). Gli aggregati pesano ogni simulazione allo stesso
> modo, le medie per seed pesano ogni seed allo stesso modo.

## 2.2 ODD ristretto, controllore degradato — `cmp_md12_narrow_lag2`, 12 seed

Sistema sotto test diverso: (il
controllore sterza su osservazioni stantie). Campionamento cieco al 7.4–8.3%,
dentro la banda tarata. Le configurazioni `+randacq` non sono state eseguite qui, quindi
il confronto è a sei configurazioni.

| # | configurazione | sim | fall | rari | rari/100 | peggiore | pct ODD | escl |
|---|---|---|---|---|---|---|---|---|
| 1 | `active_boundary[random]` | 1003 | 543 | 377 | **37.59** | −0.584 | 3.6 | 346 |
| 2 | `active_boundary[lhs]` | 944 | 496 | 335 | **35.49** | −0.574 | 4.3 | 311 |
| 3 | `cross_entropy[lhs]` | 952 | 482 | 105 | **11.03** | −0.434 | 27.9 | 10 |
| 4 | `cross_entropy[random]` | 1038 | 464 | 75 | **7.23** | −0.494 | 31.5 | 10 |
| 5 | `plain_sampling[lhs]` | 1385 | 103 | 37 | **2.67** | −0.524 | 16.4 | 46 |
| 6 | `plain_sampling[random]` | 1378 | 115 | 29 | **2.10** | −0.528 | 21.9 | 54 |

```
H1  active_boundary[lhs] > plain_sampling[lhs]    35.17 vs  2.67   12/12   p = 0.00024   RIGETTA
H2  active_boundary[lhs] > cross_entropy[lhs]     35.17 vs 11.14   11/12   p = 0.00049   RIGETTA
H3  active_boundary[rnd] > cross_entropy[rnd]     37.21 vs  8.61   11/12   p = 0.00049   RIGETTA
H4  cross_entropy[lhs]  <> plain_sampling[lhs]    11.14 vs  2.67    7/12   p = 0.791     si ferma
H5  (non testata)
```

**L'ordinamento fra le tre famiglie replica identico** su un altro ODD e un altro
sistema sotto test: fattore 13 sul pavimento, fattore 3 sulla cross-entropy,
disegni campionari indistinguibili (6/12, p = 0.622). Le otto coppie che reggono
Holm sono tutte e sole quelle fra famiglie diverse che coinvolgono active
boundary. Le differenze rispetto a §2.1 sono due, entrambe attese: `H4` qui non
rigetta (7/12 contro 9/12), e i tassi assoluti sono più alti perché il
controllore è degradato.

## 2.3 Da dove viene il vantaggio: la decomposizione 78/22

Le configurazioni `+randacq` sono identiche ad `active_boundary` in tutto — stesso design
iniziale, stesso pool, stesso budget, stesso GP ancora addestrato a ogni
iterazione — tranne che il batch è pescato **a caso** dal pool invece che per
entropia massima. Sono la stessa regione di campionamento **senza
apprendimento**, e servono perché `active_boundary` campiona **uniformemente
sulla scatola** mentre `plain_sampling` estrae dalle **marginali operative
dell'ODD**: sotto il taglio di rarità cade il **54.4%** dei punti uniformi contro
il **9.8%** di quelli ODD, quindi il fattore aggregato mette insieme *dove* si
campiona e *cosa* si impara.

| rari/100 | acquisizione = entropy | acquisizione = random | pavimento ODD |
|---|---|---|---|
| **disegno = lhs** | 41.59 | 10.19 | 1.43 |
| **disegno = random** | 41.89 | 11.33 | 1.00 |

```
quota_apprendimento = (rari[active_boundary] − rari[+randacq]) / (rari[active_boundary] − rari[plain])
```

| disegno | quota aggregata | per seed, media ± dev.std | min | max |
|---|---|---|---|---|
| `lhs` | 0.782 | 0.782 ± 0.049 | 0.713 | 0.877 |
| `random` | 0.747 | 0.751 ± 0.061 | 0.637 | 0.826 |

**Circa il 78% del vantaggio è apprendimento attivo, il restante 22% è la regione
di campionamento**, replicato su due disegni indipendenti con dispersione fra
seed piccola. Entrambi i gradini sono reali (`results/rank_md12_2x2.json`, Holm
su 15 coppie):

| confronto | cosa misura | vittorie | p Holm |
|---|---|---|---|
| `ab[lhs]` vs `ab[lhs+randacq]` | l'acquisizione per entropia | 12/12 | 0.007 * |
| `ab[lhs+randacq]` vs `plain[lhs]` | la regione di campionamento | 12/12 | 0.007 * |
| `ab[random]` vs `ab[random+randacq]` | l'acquisizione, replica | 12/12 | 0.007 * |
| `ab[random+randacq]` vs `plain[random]` | la regione, replica | 12/12 | 0.007 * |
| `ab[lhs]` vs `ab[random+randacq]` | tutto insieme, contro il caso puro | 12/12 | 0.007 * |
| `ab[random+randacq]` vs `ab[lhs+randacq]` | **la sola stratificazione** | 7/12 | 0.835 |

**La quota non dipende dalla soglia di rarità**: sullo stesso file, cambiando
solo `--rarity-q`, vale 0.775 / 0.782 / 0.806 sul disegno `lhs` e 0.732 / 0.747 /
0.764 su quello `random` ai tagli 5% / 10% (usato ovunque) / 20%. Tre punti di
quota su un fattore 4 di variazione del taglio.

## 2.4 La cross-entropy

**Il posizionamento.** Il metodo il cui unico scopo sono gli eventi rari è
**terzo o ultimo** in ogni campagna, con 3–5 volte meno rari di active boundary
con acquisizione attiva. Il percentile ODD mediano dei suoi fallimenti
(21.6–38.5) dice che lavora nella parte **probabile** del dominio.

**Le due famiglie non cercano nello stesso posto, e non possono.**
`active_boundary` campiona uniformemente sulla scatola (`from_unit`);
`cross_entropy` campiona dall'ODD (`from_dists`). La differenza è **strutturale
al metodo, non una scelta implementativa**: la CE stima una probabilità con pesi
di importanza `w = f/q`, quindi deve restare su distribuzioni assolutamente
continue rispetto a `f`. Non *può* spostarsi sulla scatola senza perdere lo
stimatore, che è la sua ragione d'essere. Active boundary non ha questo vincolo
perché non stima nulla: cerca e basta.

**Il termine di paragone corretto** è quindi `+randacq`, che campiona nella
stessa regione di active boundary *senza* apprendere:

| configurazione | rari/100 | pct ODD mediano | fallimenti totali | di cui rari |
|---|---|---|---|---|
| `active_boundary[lhs]` | 41.59 | 1.6 | 725 | 586 |
| `active_boundary[random+randacq]` | 11.33 | 0.8 | 180 | 154 |
| `active_boundary[lhs+randacq]` | 10.19 | 0.5 | 152 | 137 |
| `cross_entropy[lhs]` | 9.46 | 21.6 | 447 | 100 |
| `cross_entropy[random]` | 6.02 | 27.0 | 371 | 71 |

Nessuno dei quattro confronti `+randacq` contro `cross_entropy` è significativo
(sono le righe 3–6 della tabella in Appendice B.3: p Holm da 0.771 a 1.000). **La
cross-entropy, con tutto il suo macchinario adattivo, arriva sullo stesso livello
che active boundary raggiunge con l'apprendimento spento** — cioè al livello che
si ottiene campionando uniformemente sulla scatola senza imparare niente.

**Non è però un'equivalenza.** A parità di `rari/100` le due configurazioni non
trovano lo stesso *tipo* di fallimento: percentile ODD mediano **0.5 per
`[lhs+randacq]` e 21.6 per `cross_entropy[lhs]`**. Il primo trova pochi
fallimenti tutti nell'estrema coda (137 rari su 152); il secondo ne trova il
triplo (447) quasi tutti in scenari ordinari, di cui solo 100 scivolano sotto il
taglio. **Stesso punteggio, due popolazioni diverse.**

**Dove si legge P(fallimento).** Dal campionamento cieco, o da active boundary. A
12 seed le due stime convergono:

```
cmp_md12                     P(fail)      CV
active_boundary[lhs]         0.02391     0.14
active_boundary[random]      0.02589     0.28
plain_sampling[lhs]          0.02779     0.67
plain_sampling[random]       0.02358     0.57
```

CV = coefficiente di variazione. deviazione standard diviso la media, calcolata sui 12 seed. Misura quanto la stima balla da un seed al'altro 

---

# §3. Caratteristiche tecniche dei due simulatori

## 3.1 Backend A — Udacity + Docker

**Stack.** Build Unity dentro Docker con Xvfb (rendering software), fuori
processo, raggiunto via HTTP: FastAPI `SimulatorServer`, `POST /simulate` con
polling, sonda `GET /health`.

**Parallelismo e tolleranza ai guasti** (`scenarios/lane_keeping/config.py`):

| parametro | valore | ruolo |
|---|---|---|
| `DEFAULT_NUM_WORKERS` | 4 | container paralleli, porte 8000+ |
| `DEFAULT_TIMEOUT` | 90 s | per singolo job |
| `POLL_INTERVAL` | 0.5 s | frequenza di polling |
| `MAX_JOB_RETRIES` | 2 | rimessa in coda su un altro worker |
| `MAX_WORKER_FAILURES` | 2 | oltre il quale il worker va in quarantena |


**Il control rate è emergente.** Il loop è `DNN.predict → env.step → nuovo frame
Unity`, e il rendering software domina ogni passo. Misurato: **8.5–20.8 Hz a
seconda del carico macchina**. Da qui i due indicatori di fedeltà registrati per
ogni run — `control_hz` (iterazioni / tempo trascorso) e `meters_per_step`
(velocità media × tempo per iterazione).

**Il payload** trasmesso al simulatore:

```python
{"angles": [int(round(a)) for a in row[:5]],
 "minSpeed": int(round(row[5])), "maxSpeed": int(round(row[6])),
 "segLength": int(round(row[7])), "map_size": int(round(map_size)),
 "maxTime": 30, "maxXTE": MAX_XTE}
```

I parametri sono **quantizzati a interi**. Misurato sui bounds dell'ODD largo,
questo sposta la centerline di **29.2 cm in media** (75 cm di picco) rispetto
alla strada generata dal θ continuo.

Oltre al DNN, lo stesso backend può guidare con un **controllore su stato
esatto** — lo stesso di MetaDrive — scegliendo quale server portare su in
`docker compose`. Il simulatore non cambia: cambia solo chi decide sterzo e
acceleratore. È il percorso che rende possibile il confronto cross-simulatore
(§3.5), e vive in un package additivo che non tocca `Simulator/lanekeeping/`.

## 3.2 Backend B — MetaDrive

**Stack.** MetaDrive 0.4.3, **in-process**: nessun Docker, nessuna GPU, nessun
rendering nel loop. Richiede Python < 3.12.

**Il control rate è un parametro, non un esito** — ed è la ragione dichiarata per
cui questo backend esiste. MetaDrive separa due frequenze:

| parametro | valore | che cosa significa |
|---|---|---|
| `physics_world_step_size` | 0.02 s | quanto avanza la fisica a ogni passo interno — 50 passi di fisica per secondo simulato |
| `decision_repeat` | 5 | quanti passi di fisica vengono eseguiti prima di chiedere una nuova azione al controllore |

Il controllore decide quindi una volta ogni `5 × 0.02 = 0.1 s` di tempo
**simulato**: **10 decisioni al secondo, esatte e identiche a ogni esecuzione**,
perché non dipendono dalla velocità della macchina che sta simulando. Ne seguono
due grandezze esatte, che su Udacity sono invece misurabili solo a posteriori:

- **`dt = 0.1 s`** — l'intervallo fra una decisione e la successiva;
- **`meters_per_step = velocità / 10`** — i metri che il veicolo percorre senza
  poter correggere la traiettoria.

**Configurazione dell'ambiente.** Nella modalità usata per il confronto
l'ambiente è uno `ScenarioOnlineEnv`, costruito in `scenario_map.py` con queste
chiavi:

| chiave | valore | a che cosa corrisponde |
|---|---|---|
| `use_render` | `False` | non apre nessuna finestra e non disegna nulla: la simulazione è solo fisica e stato. È ciò che porta una run a ~1.5 s |
| `image_observation` | `False` | l'osservazione restituita dall'ambiente non contiene la telecamera. Il controllore non guarda pixel: legge posizione, assetto e velocità esatti |
| `agent_policy` | `EnvInputPolicy` | disattiva i piloti interni di MetaDrive: l'azione eseguita è quella che passiamo noi a `env.step`, non una politica del simulatore |
| `no_traffic`, `no_light`, `no_static_vehicles` | `True` | strada vuota: nessun altro veicolo, nessun semaforo, nessun ostacolo fermo. Isola il lane keeping, così l'unica causa di fallimento resta l'uscita di corsia |
| `num_scenarios`, `start_scenario_index`, `sequential_seed` | `1`, `0`, `True` | l'ambiente contiene **una sola** strada, quella che gli passiamo: nessuna mappa campionata dal simulatore |
| `physics_world_step_size`, `decision_repeat` | `0.02`, `5` | il control rate esatto di cui sopra |
| `horizon` | `budget_steps` | dopo quanti passi MetaDrive tronca l'episodio per conto proprio. Va tenuto allineato al budget dello scenario (§3.3), altrimenti il budget non ha effetto |

La modalità legacy `pgblock` non usa questa classe di ambiente ma un
`MetaDriveEnv`, che ha chiavi diverse per le stesse intenzioni — lì la strada
vuota si ottiene con `traffic_density: 0.0` e la mappa si dichiara come stringa
di blocchi (`map: "CSCCS"`).

**Costruzione della strada** — due modalità, e solo una è valida per il
confronto fra backend:

| `geometry` | come nasce la strada |
|---|---|
| `"udacity"` (default) | la strada è la **polilinea esplicita** della centerline Catmull-Rom condivisa, passata a `ScenarioOnlineEnv`. Nessun PGBlock. È l'unica modalità in cui vale «stessa θ = stessa strada» |
| `"pgblock"` | i blocchi stradali nativi di MetaDrive: dei 5 angoli dello scenario tengono solo la **media**, quindi tutti i blocchi hanno lo stesso raggio e la direzione di svolta viene dal seed. La strada che ne esce non è quella di Udacity — legacy, serve solo a riprodurre campagne vecchie |

## 3.3 Le condizioni di arresto di una simulazione

Le condizioni di arresto appartengono a quattro famiglie, ed è utile dichiarare
subito qual è quella *normale*:

| | condizione | significato |
|---|---|---|
| **uscita normale** | la strada è finita | il veicolo ha percorso tutto il tracciato |
| fallimento | `\|XTE\| > MAX_XTE` (2.5 m) | è uscito di corsia |
| fallimento | il simulatore segnala `out_of_road` / `crash` | terminazione autoritativa del backend |
| rete di sicurezza | budget dell'episodio esaurito | non è arrivato in fondo entro il tempo concesso |


### Le due condizioni di fallimento, e perché servono entrambe

**La prima è nostra: `|XTE| > MAX_XTE`** (2.5 m), con l'errore laterale misurato
da `road_frame` sulla polilinea condivisa. È l'**unica definizione di uscita di
corsia identica sui due backend** — non dipende da come il simulatore rappresenta
le corsie né da quanto sono larghe — ed è per questo il criterio su cui poggia il
confronto: se ogni backend usasse la propria nozione di «fuori strada», una
differenza fra backend potrebbe nascere dalla misura invece che dalla dinamica.

**La seconda è del simulatore: `out_of_road` / `crash`.** Il backend può chiudere
l'episodio *prima* che la nostra soglia scatti — urto contro una barriera, uscita
dalla superficie percorribile che il simulatore conosce e noi no. Lì la
traiettoria si interrompe, e senza tener conto del segnale la run risulterebbe un
successo troncato con il massimo `|XTE|` ancora sotto i 2.5 m.

**Non sono ridondanti: sbagliano in direzioni opposte.** La nostra soglia vede
uscite di corsia che il simulatore non segnala; il simulatore vede arresti che la
soglia non farebbe mai scattare. In più l'XTE relativo alla corsia **satura** una
volta usciti, quindi a fine run sotto-rileva la gravità. Gli esiti si compongono
perciò in una cascata in cui **vince l'ultima condizione che corrisponde** —
`max_step` → `arrive_dest` → `strada_completata` → `out_of_road` → `crash` → la
nostra soglia sull'ultimo punto della traiettoria — e a valle, su
`out_of_road`/`crash`, la QoI **forza** un margine negativo ordinato per gravità,
`−0.001 − (1 − frazione_percorsa)`, così i fallimenti non collassano tutti sullo
stesso numero: chi esce di strada subito è peggiore di chi esce quasi in fondo.

### I due loop, affiancati

| | **Udacity — DNN** | **MetaDrive** |
|---|---|---|
| strada finita | assente | `state["beyond_end"]` |
| XTE oltre soglia | `abs(cte) > maxXTE` | `abs(lateral_error) > MAX_XTE` |
| segnale del simulatore | `done` da Unity | `terminated or truncated` |
| limite temporale | `elapsed > maxTime` — **30 s wall clock fissi** | `steps < budget` — **step simulati dallo scenario** |

Su MetaDrive la condizione di fine strada **scavalca deliberatamente**
`arrive_dest` del simulatore, che calcola la destinazione dalla traccia dell'ego:
non coincide necessariamente con l'ultimo punto della centerline condivisa, e
l'unica definizione che conta è «ho percorso tutto il tracciato».

### Il budget dell'episodio

Su MetaDrive l'orizzonte non è una costante: è **derivato dallo scenario**
(`scenarios/common/episode_budget.py`). Su Udacity il DNN gira invece con un
`maxTime` fisso di 30 s di orologio, che è il limite della catena originale e il
motivo per cui la regola è stata scritta:

```
budget_seconds = clamp( lunghezza_strada / velocità_target × 1.5 ,  10 s ,  120 s )
```

velocità_target = 0.5 × (min_speed + max_speed) × speed_scale

lunghezza_strada ≈ 4 × segment_length (+2–4% per la curvatura)

## 3.4 La tabella delle differenze

Il confronto fra backend ha senso solo perché **cinque cose sono identiche per
costruzione**: i nove parametri dello scenario, la mezzeria generata da
`road_polyline`, la definizione dell'errore laterale e di assetto (`road_frame`
sulla stessa polilinea), il controllore laterale e la QoI. Tutto il resto è il backend, ed è ciò che il
confronto interroga.

| | **A — Udacity + Docker** | **B — MetaDrive** |
|---|---|---|
| esecuzione | container Unity, fuori processo, HTTP: POST, job id, polling | in-process, headless: una simulazione è una chiamata di funzione |
| rendering | software, Xvfb — domina il costo | nessuno nel loop |
| **control rate** | **emergente, 8.5–20.8 Hz** | **configurato, 10.0 Hz esatti** |
| `dt` per il controllore | misurato a posteriori | esatto: `decision_repeat × physics_step` |
| passo di fisica | interno a Unity, non esposto | 0.02 s, esposto e modificabile |
| dinamica del veicolo | modello proprio di Unity | motore fisico Bullet |
| semantica dell'azione | (sterzo, acceleratore) normalizzata| (sterzo, acceleratore) normalizzata|
| stato esposto | posizione e cross-track error; **assetto e tangente assenti** | posizione e assetto leggibili, `dt` costante noto |
| gate di fedeltà | esiste, indicatori sempre misurati | assente di proposito: non c'è nulla da sorvegliare |
| sistema sotto test | DNN `mixed-chauffeur.h5` su immagine | controllore laterale su stato esatto |
| percezione | telecamera | nessuna |
| parallelismo | 4 container, pool con quarantena | `n_jobs`, in-process |
| **costo per simulazione** | **~30 s** | **~1.4–2.0 s** |
| parametri ricevuti | **arrotondati a interi** | float pieni |
| strada | catena Catmull-Rom nativa | stessa polilinea, via `ScenarioOnlineEnv` |
| terminazione | `maxTime = 30 s`, `maxXTE`, `done` (§3.3) | `beyond_end`, `out_of_road`/`crash`, `MAX_XTE`, budget (§3.3) |
| modalità di guasto dominante | percezione | dinamica di guida |

**Il costo è ciò che ha reso possibile la statistica.** Un fattore 20 fra i due
modi di esecuzione: le campagne a 12 seed sono ~2 h su MetaDrive e sarebbero ~21 h
su Udacity, e con 8 seed il pavimento del p sarebbe 0.0078 e nessuna coppia
passerebbe Holm (Appendice B).

**Il control rate è la differenza che pesa di più sul piano metodologico.** I
metri percorsi fra due decisioni sono velocità diviso control rate: su Udacity
quella grandezza resta una **variabile nascosta**, su MetaDrive è esatta,
riproducibile e sweepabile.

**Dinamica e semantica dell'azione non sono sotto il nostro controllo.** MetaDrive
poggia su Bullet, Unity ha il proprio modello. Entrambi accettano una coppia
(sterzo, acceleratore) normalizzata, ma la convenzione di segno di Unity è
invertita rispetto a quella condivisa — da cui `STEERING_SIGN = -1`, determinato
sperimentalmente perché non documentato — e la mappatura da comando normalizzato
ad angolo di ruota non è la stessa: **lo stesso comando produce due sterzate
diverse**.

**L'informazione esposta impone un'asimmetria da dichiarare.** Su MetaDrive
posizione e assetto si leggono direttamente e il passo temporale è una costante
nota; la telemetria di Unity espone posizione e cross-track error ma **non la
tangente della corsia né l'assetto**, che vanno ricostruiti proiettando la
posizione sulla polilinea e stimando l'orientamento da due posizioni consecutive
(da cui la necessità, su Udacity, di conservare posizione e istante precedenti
anche per misurare il `dt` reale). Il limite da registrare è che l'heading così
ottenuto è quello della **traiettoria**, non dell'assetto: le due grandezze
coincidono solo senza slittamento, e a queste velocità la differenza è piccola ma
non nulla.

Per tutte queste ragioni il confronto fra backend **non paragona i tassi di
fallimento assoluti** — che dipendono insieme dalla difficoltà del simulatore,
dalla bontà del controllore su quel simulatore e dal punto operativo a cui
ciascuno è tarato — ma soltanto l'accordo sull'**ordinamento** degli scenari e la
**sovrapposizione delle regioni di fallimento**.

## 3.5 I due backend a confronto diretto

### Parità geometrica

L'assunto su cui poggia il confronto: la stessa θ deve produrre la stessa strada.
Con i PGBlock era **falso** — sul design LHS a 60 punti: lunghezza 202.7 m su
Udacity contro 124.0 m su MetaDrive (range 201–205 contro 51–200), Spearman fra
le lunghezze −0.026. Con `ScenarioOnlineEnv` e la polilinea esplicita la parità è
misurata: **0.021 cm medi, 1.88 cm max**, lunghezza 202.55 m contro 202.61 m
(−0.03%), ed è sorvegliata da un gate (`scripts/diag_road_parity.py`) che si
rifiuta di dichiarare OK se non ha verificato nulla. La verifica copre la catena
geometrica; la quantizzazione a interi del payload Udacity (§3.1) resta a monte
di essa.

### I moduli condivisi

Quattro moduli in **una sola copia**, puri numpy, che non importano alcun
simulatore, in `scenarios/common/`; le copie dentro il container sono confrontate
per **hash del file**, non per comportamento. Sono ciò che rende possibile il
confronto cross-simulatore: valgono sui percorsi che quel confronto usa, non
sulla catena originale del DNN, che ha la propria strada, la propria misura
dell'errore e il proprio orizzonte a 30 s.

| modulo | perché deve essere identico |
|---|---|
| `road_geometry.py` | un solo generatore di strada |
| `driver.py` | un solo controllore laterale su entrambi i backend |
| `road_frame.py` | una sola definizione di errore laterale e di assetto |
| `episode_budget.py` | quanto dura un episodio, derivato dallo scenario |

Su `episode_budget` il numero che lo giustifica è il ~80% contro ~92% di §3.3.
Su `road_frame`: l'XTE si misura con `road_frame` sulla centerline condivisa, non
con l'API della corsia di MetaDrive, che restituisce il laterale col **segno
opposto** (misurato: −8.530 contro +8.530 sullo stesso punto). La ragione di
fondo non è il segno ma il principio: se ogni simulatore riporta l'errore con la
propria convenzione, una differenza fra backend può nascere dalla **misura**
invece che dalla dinamica, e a valle è indistinguibile.

---

# §4. LHS contro random

## 4.1 Che cosa fa LHS, e perché ci si aspettava che aiutasse

**Il meccanismo.** Dovendo scegliere 5 valori di velocità fra 0 e 100 km/h, il
campionamento casuale può darti `12, 18, 24, 31, 88`: quattro valori ammassati in
basso e un buco fra 31 e 88. LHS divide prima l'intervallo in 5 fasce di uguale
ampiezza e pesca un valore a caso dentro ciascuna. Con *n* punti gli strati sono
*n*, larghi `1/n` del range, e nessuno resta vuoto; il campionamento casuale ne
lascia vuoto circa il **37%**, cioè `(1 − 1/n)ⁿ`. Con 9 parametri LHS fa questa
operazione su ogni asse contemporaneamente.

**L'ipotesi che ha motivato il lavoro.** Trovare un fallimento raro significa far
cadere almeno un punto del disegno dentro una regione piccola dello spazio dei
parametri. Se il campionamento casuale lascia scoperto il 37% delle fasce di ogni
asse e la regione di fallimento sta in una di quelle, il disegno la manca: **i
fallimenti rari stanno nei buchi che il campionamento casuale lascia, e LHS quei
buchi li chiude.**

**Le due condizioni perché funzioni.** La garanzia di LHS è su un parametro alla
volta: assicura che fra i 40 scenari ce ne sia uno con `angle_1` molto alto e uno
con `min_speed` molto alta, **non che quei valori capitino nello stesso
scenario** — e uno scenario è una combinazione simultanea di tutti e nove.

| forma della regione di fallimento | LHS aiuta? |
|---|---|
| decisa da **un parametro** («oltre gli 80° di curvatura si esce di strada, qualunque sia il resto») | **sì**: LHS è obbligato a provare tutta la gamma di quel parametro |
| decisa da una **combinazione** («curva stretta *e* velocità alta *e* segmento corto, insieme») | **no**: LHS non controlla quali valori si presentano insieme |

È il contenuto del risultato di **Stein (1987, *Technometrics* 29(2), 143–151)**:
LHS riduce la varianza dovuta agli effetti singoli e lascia intatta quella delle
interazioni. La seconda condizione è la **risoluzione**: gli strati sono larghi
`1/n` e LHS garantisce *in quale* strato cade un punto, non *dove* dentro lo
strato. Una regione più stretta di uno strato non gli è visibile.

> Perché LHS paghi servono quindi due cose insieme: che la regione sia decisa da
> pochi parametri, e che su quei parametri sia larga almeno quanto uno strato.

## 4.2 Il regime in cui LHS vince davvero — misurato

`scripts/validate_model_comparison.py` mette alla prova le due condizioni su
regioni **note analiticamente**: si genera un disegno di 64 punti in 4 dimensioni,
si guarda se **almeno uno** cade dentro la regione, e si ripete 4000 volte. La
percentuale è la frazione di repliche in cui il disegno ha centrato la regione —
cioè quante volte una campagna avrebbe scoperto quel fallimento.

Il **volume** è la frazione di cubo unitario che la regione occupa, cioè la
probabilità che un singolo punto a caso ci finisca dentro: si moltiplicano le
ampiezze imposte su ciascun asse, contando 1 per gli assi lasciati liberi (in
`|p − c| < r` la fascia è larga `2r`; in `p > t` è larga `1 − t`).

La colonna **atteso i.i.d.** è il valore esatto `1 − (1 − volume)ⁿ` e serve da
controllo di sanità: la colonna `random` la segue, quindi l'impianto misura ciò
che dice di misurare. **Il verdetto:** con un asse solo LHS scopre la regione
nell'88% delle campagne contro il 72% del campionamento casuale — **16 punti
percentuali**. Appena due assi devono cospirare, il vantaggio scende a 1–2 punti.

## 4.3 Perché in questo problema non è successo

**La regione di fallimento è una combinazione.** Il dato che lo mostra è
`axis_spread` (Appendice C.3), la frazione del range che i fallimenti coprono
asse per asse: su `cmp_md12` scende sotto 0.9 soltanto su due assi su nove, e mai
sotto 0.56; sulle campagne Udacity non scende sotto 0.97 su nessun asse. **I
fallimenti si trovano un po' ovunque lungo ogni singolo parametro**, quindi non
esiste la fascia che la stratificazione saprebbe coprire: siamo fuori dalla prima
riga della tabella di §4.1, e questo basta all'argomento senza dover stabilire
quale combinazione li produca.

**E il regime numerico peggiora le cose.** Il design iniziale è di **40 punti in
9 dimensioni**. Ripetendo l'esperimento di §4.2 in quel regime, a **volume fisso**
0.004 e cambiando soltanto su quanti assi si distribuisce (tenere il volume fisso
è essenziale: altrimenti non si saprebbe se il calo viene dalla forma o dalla
dimensione), più assi sono coinvolti più larga deve essere ciascuna condizione,
perché è la loro intersezione a stringere:

| assi che decidono | ampiezza per asse | LHS | random | rapporto |
|---|---|---|---|---|
| 1 | 0.0040 | 15.9% | 14.7% | 1.09 |
| 2 | 0.0632 | 15.5% | 14.0% | 1.11 |
| 3 | 0.1587 | 14.6% | 14.2% | 1.02 |
| 4 | 0.2515 | 15.1% | 15.0% | 1.01 |
| 6 | 0.3984 | 14.2% | 15.4% | 0.93 |
| 9 | 0.5415 | 14.0% | 15.0% | 0.93 |

**Nessun guadagno a nessun livello**, nemmeno nella riga a un asse — che in §4.2
era quella vincente. Il motivo è la seconda condizione: lì la fascia è larga
0.0040, mentre con 40 punti gli strati sono larghi `1/40 = 0.025`. La regione è
**sei volte più fine della risoluzione di LHS**.


La lettura ingenua del primo risultato era «il GP lava via il disegno iniziale».
Non regge: il pareggio c'è anche col GP spento e anche senza alcun algoritmo.
**Non è il modello a mascherare un effetto della stratificazione — su questa
metrica l'effetto non esiste a nessuno dei tre livelli.** Lo stesso test appaiato
per seed sui fallimenti trovati, ripetuto sulle tre famiglie in entrambe le
campagne MetaDrive, dà vittorie LHS fra 4/12 e 8/12 e p fra 0.269 e 0.964:
nessuno si avvicina alla soglia. L'unica eccezione in tutto il lavoro è
`cmp_ab12` su Udacity (23.7 contro 19.3, 7/12, p = 0.074), già commentata in
§1.2.

**Il pareggio è replicato su due simulatori, due ODD, due sistemi sotto test, tre
famiglie di ricerca e tre livelli di guida dell'algoritmo.** Con il pavimento del
p a 0.00049 si sa anche quale effetto minimo era rilevabile, e non è stato
rilevato.

**La conclusione difendibile.** L'ipotesi che ha motivato il lavoro è
**smentita** — o più precisamente non rilevabile con un effetto grande abbastanza
da contare, in ogni configurazione provata. È un risultato negativo pulito e
replicato, con il pavimento del p noto, e ha una spiegazione meccanicistica
verificata: **la stratificazione paga quando il fallimento è guidato da un
parametro dominante su una scala confrontabile con `1/n`, e questo problema non è
di quel tipo.**

---


# Appendice A — Variabili e metriche

## A.1 Lo scenario e i nove parametri

Uno scenario di **lane keeping**: si genera una strada da 9 parametri, ci si fa
guidare un agente, e si misura quanto è uscito dalla corsia. Ogni simulazione
restituisce una traiettoria `(T, 4)`: `x`, `y`, `xte` (cross-track error),
`steering`.

| # | parametro | unità | range default | cosa controlla |
|---|---|---|---|---|
| 0–4 | `angle_1` … `angle_5` | gradi | [0, 85] | orientamento assoluto dei 5 segmenti. Valori alti = curve strette |
| 5 | `min_speed` | m/s | [5, 15] | limite inferiore della banda di velocità |
| 6 | `max_speed` | m/s | [10, 30] | limite superiore |
| 7 | `segment_length` | m | [10, 40] | lunghezza di ciascun segmento |
| 8 | `map_size` | m | [150, 350] | lato della mappa |

I due backend dichiarano **gli stessi identici bounds**.

## A.2 L'ODD: bounds e marginali operative

**(a) I bounds** — la scatola da cui si estrae. Due configurazioni:

| | ODD largo (default) | ODD ristretto |
|---|---|---|
| `angle_1..5` | [0, 85]° | **[0, 8]°** |
| `min_speed` | [5, 15] | [5, 9.6] |
| `max_speed` | [10, 30] | **[8.6, 9.6]** |
| `segment_length` | [10, 40] | [10, 12] |
| `map_size` | [150, 350] | [150, 350] |

Criterio di scelta: il tasso di `plain_sampling` deve cadere fra il 2% e il 10%.
Sopra, i fallimenti non sono rari e ogni metodo li trova; sotto, non se ne
trovano abbastanza per misurare qualcosa.

Ogni classifica è **interna alla propria campagna**: `rank_arms.py` prende l'ODD
dai metadati e `compare_rankings` si rifiuta di affiancare classifiche calibrate
su scatole diverse.

**(b) Le marginali operative** (`param_distributions`) — con quale probabilità
si presenta ciascun valore *dentro* la scatola:

| parametri | distribuzione | effetto |
|---|---|---|
| `angle_1..5` | truncnorm, μ = `lower`, σ = 0.5 · range | curve dolci molto più probabili di quelle strette |
| `min_speed`, `max_speed` | truncnorm, μ = `lower` + 0.4 · range, σ = 0.3 · range | velocità intermedie più probabili degli estremi |
| `segment_length`, `map_size` | uniforme | nessuna preferenza |

«Fallimento raro» è definito rispetto a queste marginali, non alla scatola.
Estrarre uniformemente e estrarre dall'ODD dà tassi che differiscono di un
fattore ~5 (misurato: 17.2% uniforme contro 3.2% sotto ODD).

## A.3 La QoI: il margine di sicurezza

```
g(θ) = 0.6 · M1  +  0.2 · M2  +  0.2 · M3
```

| termine | formula | cosa penalizza | range |
|---|---|---|---|
| **M1** | `MAX_XTE − max\|xte\|`, con `MAX_XTE = 2.5 m` | uscita di corsia | illimitato |
| **M2** | `−(max\|s\| − mean\|s\|) / 0.4`, troncato | picchi di sterzo | [−1, 0] |
| **M3** | `−(valid − first_near) / valid`, soglia a 0.7 · `MAX_XTE` | avvicinamento precoce al bordo | [−1, 0] |

**Soglia di fallimento: `g(θ) < 0`.** Il margine è **continuo** e non binario di
proposito: l'etichetta fallimento/successo cambia su ~20% dei punti per solo
rumore del simulatore, mentre il margine è una misura in cui il rumore si media.

**Run invalide → `g = NaN`**, escluse da tutti i tassi. Tre cause: episodio
degenere (< 3 passi), parametri incoerenti (`min_speed > max_speed`), fedeltà di
controllo insufficiente. Nelle campagne sono il 3–5%.

## A.4 La metrica di classifica: `rari/100`

Contare i fallimenti premia per costruzione chi cerca la frontiera. La metrica
usata è più stretta: **un fallimento è raro quando il punto in cui avviene è
improbabile sotto l'ODD**.

1. si estraggono 200 000 punti di riferimento dalle marginali operative `f`;
2. per ciascuno si calcola `log f(θ) = Σⱼ log fⱼ(θⱼ)`;
3. `log_f_cut` = decile inferiore di quelle log-densità;
4. un fallimento è **raro** se `log f(θ) ≤ log_f_cut`.

Il campione di riferimento è **esterno, con seed fisso, indipendente dalla
campagna**: una configurazione non può gonfiare la metrica cercando più forte.

| metrica | definizione |
|---|---|
| `rare` | fallimenti sotto il taglio |
| **`rare/100`** | `100 × rare / n_evaluations` — la metrica di classifica |
| `pct ODD` | percentile operativo **mediano** dei fallimenti trovati. Più basso = la configurazione lavora più in fondo alla coda |
| `escl` | regioni raggiunte da quella configurazione e da nessun'altra |

`log_f_cut` vale −36.954 sull'ODD largo e −18.657 su quello ristretto.

## A.5 Il punto operativo

Due manopole che **non fanno parte dello scenario** ma decidono se il sistema è
testabile. Tarate per **bisezione** sul tasso di fallimento, con lo stesso seed e
lo stesso design a ogni valutazione, e registrate **dentro** i file di risultato
insieme all'ODD su cui sono state tarate.

- **`speed_scale`** — moltiplica la velocità obiettivo. A 1.0 su ODD largo il
  controllore fallisce nel 94.8% dei casi, a 0.15 nello 0%. Valore tarato per
  l'ODD largo: **0.3625**, che dà il 17.2% sotto design uniforme.
- **`obs_lag_tau`** — costante di tempo di una media esponenziale sulle
  osservazioni, `a = exp(−dt/τ)`: il controllore sterza su uno stato **stantio**.
  Riproduce il modo di guasto di una rete su immagine invece di limitarsi ad
  aumentare la velocità, e siccome un ritardo fisso in secondi diventa staleness
  crescente in metri, i fallimenti restano dipendenti dallo scenario.

> **Attivare `obs_lag` cambia il sistema sotto test.** Una campagna con
> `obs_lag > 0` è un terzo sistema, non una replica del secondo.

## A.6 Le campagne eseguite

Una **campagna** = 3, 6 o 8 configurazioni × N seed × 120 simulazioni. Il seed controlla
il design campionario.

| campagna | backend | seed | ODD | punto operativo | esito |
|---|---|---|---|---|---|
| `cmp_rare` | A | 6 | ristretto | non tarato | 6 configurazioni, descrittiva (§1.1) |
| `cmp_ab12` | A | 12 | ristretto | non tarato | solo `active_boundary` (§1.2) |
| `cmp_md12` | B | 12 | largo | `speed_scale = 0.3625` | 6 configurazioni, **dimostrata** |
| `cmp_md12_narrow` | B | 12 | ristretto | `speed_scale = 0.3625` | controllore ideale, 0 fallimenti (§3.5) |
| `cmp_md12_narrow_lag2` | B | 12 | ristretto | `speed_scale = 1.0`, `obs_lag = 0.2456` | 6 configurazioni, **replica** (§2.2) |
| `confronto_cross` | A+B | — | largo | design condiviso | cross-simulatore (§3.5) |
| `cmp_md12_ctrl` / `_ctrl_rand` | B | 12 | largo | `speed_scale = 0.3625` | 3+3 configurazioni, controlli di acquisizione |
| `cmp_md12_2x2` | B | 12 | largo | `speed_scale = 0.3625` | unione **verificata**, base del disegno 2×2 |
| `cmp_md12_full` | B | 12 | largo | `speed_scale = 0.3625` | unione verificata a **8 configurazioni**, la classifica di §2.1 |

**L'ordine di esecuzione delle coppie (configurazione, seed) è mescolato.** Non è un
dettaglio: gli stessi 30 punti rivalutati tre volte di fila su Udacity hanno dato
**37.9%, 20.7%, 6.9%** di fallimenti, un declino monotono mentre i container si
scaldano. Eseguendo una configurazione per volta, il primo avrebbe un vantaggio
sistematico che nessuna analisi a valle potrebbe separare dall'effetto del
metodo. `drift_diagnostic` la misura poi a posteriori, con un flag `computable`
che distingue «non c'è deriva» da «non l'ho potuta misurare».

## A.7 Perché è lecito unire due campagne

Ogni campagna costa ~2 h e la seconda è stata decisa dopo aver letto la prima,
quindi nessun `.npz` contiene entrambe le configurazioni di controllo.
`scripts/merge_campaigns.py` li unisce ma **si rifiuta** se i metadati non
coincidono o se le configurazioni in comune non riproducono.

La verifica contro `cmp_md12` è passata al livello più forte disponibile: le
quattro configurazioni condivise producono **lo stesso insieme di margini, seed
per seed**, entro 1e-9 — non gli stessi array, perché l'ordine di esecuzione è
mescolato per campagna, ma lo stesso multiinsieme di esiti per ogni seed. **La
pipeline è deterministica data la coppia (configurazione, seed).** I conteggi di
regioni restano invece non confrontabili fra campagne, perché DBSCAN gira sui
soli fallimenti presenti e `eps` è stimato dai dati (`active_boundary[lhs]`: 494
regioni in `cmp_md12`, 285 in `cmp_md12_ctrl`): solo `rare_per_100` è
per-configurazione e confrontabile.

---

# Appendice B — La statistica

## B.1 Appaiato per seed

La varianza seed-a-seed è maggiore della differenza fra le configurazioni:
`active_boundary[lhs]` va da 30.3 a 49.1 rari/100 a seconda del seed, mentre la
differenza fra i due disegni campionari è 0.3. Confrontare medie aggregate
butterebbe via il segnale.

Si confrontano le due configurazioni **dentro ogni seed** e si testano le differenze con
**Wilcoxon signed-rank** (non parametrico: con una manciata di seed la normalità
non è verificabile), unilaterale dove la direzione è dichiarata in anticipo.

## B.2 Il pavimento del p

Sotto l'ipotesi nulla il segno di ogni differenza è un lancio di moneta, quindi
con *n* seed ci sono `2ⁿ` configurazioni e una sola è tutta positiva:

```
p bilaterale minimo = 2 / 2ⁿ
```

| n seed | 3 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| p minimo | 0.250 | 0.031 | 0.0078 | 0.0020 | 0.00049 |

Un risultato al pavimento significa «non si può fare meglio con questo numero di
seed», non «effetto enorme». Con 3 seed **niente può essere significativo**: sui
3 seed iniziali un effetto di 30× (43.59 contro 1.42 rari/100, 3 vittorie su 3)
dava p = 0.125.

## B.3 Due livelli di analisi

**Esplorativo** — tutte le coppie, con correzione di **Holm**: si ordinano i p, si
moltiplica l'*i*-esimo per `(m − i + 1)` (il moltiplicatore cala perché a ogni
rigetto restano meno ipotesi che potrebbero essere nulle), si tronca a 1 e si
impone che i valori corretti non diminuiscano scendendo. Su 15 coppie la soglia
per la coppia più forte è `0.05/15 = 0.0033`:

| n seed | pavimento | basta per Holm su 15 coppie? |
|---|---|---|
| 6 | 0.031 | no |
| 8 | 0.0078 | no |
| 10 | 0.0020 | sì |
| 12 | 0.00049 | sì, con margine |

È il motivo per cui le campagne dimostrative sono a 12 seed e non a 8.

**Le 12 coppie di `cmp_md12_full` che non si separano**, nell'ordine in cui
`rank_arms.py` le registra; `vittorie A` conta i seed in cui la prima
configurazione ha battuto la seconda:

| A | B | vittorie A | p | p Holm |
|---|---|---|---|---|
| `ab[random]` | `ab[lhs]` | 5/12 | 0.791 | 1.000 |
| `ab[random+randacq]` | `ab[lhs+randacq]` | 7/12 | 0.380 | 1.000 |
| `ab[random+randacq]` | `ce[lhs]` | 8/12 | 0.470 | 1.000 |
| `ab[random+randacq]` | `ce[random]` | 10/12 | 0.077 | 0.771 |
| `ab[lhs+randacq]` | `ce[lhs]` | 7/12 | 0.733 | 1.000 |
| `ab[lhs+randacq]` | `ce[random]` | 9/12 | 0.151 | 1.000 |
| `ce[lhs]` | `ce[random]` | 6/12 | 0.301 | 1.000 |
| `ce[lhs]` | `plain[lhs]` | 9/12 | 0.034 | 0.410 |
| `ce[lhs]` | `plain[random]` | 8/12 | 0.054 | 0.591 |
| `ce[random]` | `plain[lhs]` | 6/12 | 0.278 | 1.000 |
| `ce[random]` | `plain[random]` | 6/12 | 0.278 | 1.000 |
| `plain[lhs]` | `plain[random]` | 8/12 | 0.278 | 1.000 |

Da qui si leggono tre cose. I `p Holm` a **1.000** sono valori troncati, non
misurati: il p corretto ha superato 1, e la monotonia appiattisce su quel valore
tutti i confronti più deboli. Le quattro coppie `cross_entropy` contro
`plain_sampling` **non** reggono pur avendo lo stesso divario delle `+randacq`
che invece reggono (9.46 contro 1.43 e 10.19 contro 1.43): la differenza è la
varianza fra seed, non l'entità dell'effetto. E `non si separano` significa
assenza di prova, non prova di equivalenza.

> **Perché i p Holm che reggono sono tutti uguali.** Quattordici confronti sono
> vinti 12/12, quindi hanno tutti il p minimo 0.00049, e la monotonia appiattisce
> il gruppo sul massimo incontrato: `0.00049 × 28 = 0.0137`. È il valore più
> significativo che 12 seed e 28 coppie possano produrre insieme, non una misura.

**Confermativo** — una **sequenza fissa** pre-registrata di 5 ipotesi ordinate,
ciascuna al pieno α = 0.05, fermandosi alla prima che fallisce. Il tasso di
errore familiare resta 0.05 **senza correzione**, perché ogni test è raggiunto
solo se tutti i precedenti hanno rigettato. Il prezzo è che l'ordine è un impegno
preso prima dei dati. I piani sono in `docs/preregistrazione_*.json`, committati
prima di generare i dati.

`H4` (§2.1) è il caso che mostra la differenza fra i due protocolli: la stessa
coppia `cross_entropy[lhs]` vs `plain_sampling[lhs]` rigetta nella sequenza
(p = 0.034, dichiarata in anticipo) e non regge nella matrice esplorativa
(p Holm = 0.410, pescata fra 28 confronti). Stesso dato, prezzi diversi.

## B.4 Le protezioni nel codice

- **`VOID`**: se entrambe le configurazioni sono identicamente zero su una metrica, il
  test non restituisce un pareggio ma un verdetto esplicito di campagna senza
  dati. Un pareggio si legge come evidenza; l'assenza di dati no.
- **Anti-imbroglio dell'unilaterale**: `p = (p_two / 2.0) if correct_way else 1.0`.
  Dimezzare il p bilaterale è lecito solo se l'effetto va nella direzione
  dichiarata in anticipo; se va al contrario, per quanto grande sia, l'ipotesi ha
  fallito.
- **Gate su `p_fail_usable`**: quando lo stimatore di probabilità di una configurazione è
  fuori dal proprio regime di validità, `p_fail` viene escluso dal confronto
  invece di produrre un pareggio privo di significato.

## B.5 Il regime di validità dello stimatore cross-entropy

`ModelComparison.default()` con `budget = 120` produce `final_samples = 45`, di
cui `int(0.2 × 45) = 9` estratti dalla distribuzione nominale. Gli altri 36
vengono dalla proposta `q` e sono ripesati per `f/d`, che va a zero appena `q` si
allontana da `f` — cioè appena la discesa CE fa il suo mestiere. Restano 9 punti
utili: **lo stimatore è quantizzato a multipli di 1/9.** Un controllo su bersagli
analitici a 9 dimensioni individua il regime:

```
bersaglio     budget                            P vera     stima CE
unimodale     validato (spi=500, final=8000)    0.00034    0.000249    ok
unimodale     campagna (spi=15,  final=45)      0.00034    0 in 12/12
multimodale   validato                          0.20230    0.2038      ok
multimodale   campagna                          0.20230    solo {1/9, 2/9, 3/9}
```

**È il budget a determinare il regime, non la multimodalità del bersaglio.** Il
codice lo dichiara da solo (`MIN_DEFENSIVE_SAMPLES = 30`, warning alla
costruzione della configurazione e `p_fail_usable` propagato fino al report) — un
warning e non un'eccezione, di proposito: la discesa CE resta una ricerca
legittima anche con la stima di probabilità fuori regime, e la classifica la
valuta su ciò che ha **trovato**, che non ne è toccato.

---

# Appendice C — Geometria dei fallimenti

## C.1 Come si costruiscono le regioni

I fallimenti vengono raggruppati con **DBSCAN** nel cubo unitario a 9 dimensioni
(`min_samples = 2`, raggio `eps` stimato dai dati: un raggio fisso si rompe al
cambiare della dimensione, e in 9-D due punti uniformi distano ~1.22).

| metrica | definizione |
|---|---|
| `regions` | gruppi formati |
| **localised** | un gruppo è localizzato quando la sua estensione su almeno un asse è sotto il 50% di quella che *n* punti coprirebbero per caso. La scala con *n* è necessaria: 3 punti coprono in media metà di un asse |
| `p_hit_odd` | probabilità che **una** estrazione dall'ODD cada nella regione |
| `n50` | quante estrazioni cieche servirebbero per incontrarla una volta su due |

I box delle regioni sono **localizzazioni descrittive**, non condizioni
necessarie: il clustering rende compatti i gruppi per costruzione.

## C.2 `structure_score` — l'unico test non circolare

Confronta la frazione di fallimenti che il clustering raggruppa nei dati veri
contro gli stessi dati con **gli assi mescolati indipendentemente** (20
permutazioni, che distruggono ogni struttura congiunta preservando le marginali).
Restituisce uno z-score; soglia dichiarata nel codice: **z > 2**.

```
cmp        (A, ODD largo,   3 seed)   z = +0.10   nessuna struttura
cmp_rare   (A, ristretto,   6 seed)   z = +1.77   sotto soglia
cmp_ab12   (A, ristretto,  12 seed)   z = +3.55   STRUTTURA
cmp_md     (B, largo,       3 seed)   z = -3.93   nessuna struttura
cmp_md12   (B, largo,      12 seed)   z = -1.20   nessuna struttura
cmp_md12_narrow_lag2 (B, degradato)   z = +2.80   STRUTTURA
```

## C.3 Localizzazione per asse

`axis_spread` = frazione del range dell'ODD **di quella campagna** coperta dai
fallimenti. Normalizzata per il range, quindi interna a un ODD.

| campagna | asse minimo | valore | assi localizzati (< 0.9) |
|---|---|---|---|
| `cmp_rare` (A, ristretto) | min_speed | 0.972 | nessuno |
| `cmp_ab12` (A, ristretto) | min_speed | 0.974 | nessuno |
| `cmp_md12` (B, largo) | angle_1 | 0.565 | `angle_1`, `max_speed` |
| `cmp_md12_narrow_lag2` (B, degr.) | min_speed | 0.417 | `min_speed` |

Le frazioni vanno lette insieme allo span in unità fisiche, perché il
denominatore è la larghezza della scatola: l'`angle_1` «localizzato» di B copre
47 gradi, sei volte l'intero asse di A.

## C.4 Il risultato geometrico solido

Nella campagna degradata: `min_speed` localizzato a 0.42, struttura z = +2.8, e
la regione dominante (R9, 1140 fallimenti) è `min_speed ∈ [7.58, 9.57]`. I
fallimenti stanno dove il veicolo va più veloce, perché con un ritardo fisso in
secondi la staleness in metri cresce con la velocità. **Il meccanismo che
`obs_lag` doveva riprodurre si legge nei dati.**

Le regioni più rare trovate (`cmp_md12`) arrivano a `P(hit) = 3.1 × 10⁻¹¹`: un
disegno cieco ne servirebbe 2.3 × 10¹⁰ per incontrarne una volta su due. **884 dei
945 gruppi di fallimento sono raggiunti dalla sola famiglia active boundary**:
403 esclusivi di `[random]`, 399 esclusivi di `[lhs]`, 82 condivisi fra i due,
contro 2 e 4 della cross-entropy e 13 e 14 del pavimento.

---

# Appendice D — Riproducibilità

I risultati stanno in `results/` (una coppia `.npz` + `.json` per campagna), i
piani pre-registrati in `docs/preregistrazione_*.json`, committati prima dei
dati. **L'elenco dei file, i comandi che li generano e la mappa dei moduli sono
in `docs/comandi_riproduzione.md`.**

Il punto che conta qui: `scripts/rank_arms.py` **non simula**. Rilegge il `.npz`
e ricalcola classifica, test appaiati e sequenza pre-registrata in una decina di
secondi, quindi **ogni tabella di questo documento è rigenerabile senza toccare
un simulatore**.
