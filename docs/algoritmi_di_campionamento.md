# Gli algoritmi di campionamento

Questo documento spiega i tre algoritmi che il progetto mette a confronto —
**plain sampling**, **cross-entropy** e **active boundary** — partendo da zero.
Presuppone un background ingegneristico ma nessuna conoscenza pregressa di
campionamento, importance sampling o modelli surrogati. I risultati sperimentali
non stanno qui: sono in [`risultati_sperimentazione.md`](risultati_sperimentazione.md).

---

## 1. Il problema

### Lo scenario è un vettore di nove numeri

Una simulazione di lane keeping è definita da nove parametri: cinque angoli che
danno la forma della strada, la velocità minima e massima della banda in cui il
veicolo viaggia, la lunghezza dei segmenti stradali e la dimensione della mappa.
Quel vettore si chiama **θ** (theta).

```
θ = (angle_1, …, angle_5, min_speed, max_speed, segment_length, map_size)
```

Fissato θ, la simulazione è deterministica: stessa θ, stessa strada, stessa
guida, stesso esito. Lo **spazio dei parametri** è quindi l'insieme dei θ
ammissibili, e cercare fallimenti significa cercare punti dentro quello spazio.

Per lavorarci comodamente si normalizza ogni parametro nel suo intervallo, così
che lo spazio diventi il **cubo unitario** in nove dimensioni: ogni coordinata fra
0 e 1, tutti gli assi con lo stesso peso a prescindere dalle unità fisiche.

### La funzione da valutare è cara

Ogni θ produce un numero, il **margine di sicurezza** `g(θ)`: positivo se il
veicolo ha tenuto la corsia, negativo se ha fallito. È una funzione come le
altre, con una particolarità che governa tutto il resto: **per conoscerne il
valore in un punto bisogna eseguire una simulazione**, che costa ~1.5 secondi su
MetaDrive e ~30 su Udacity.

Non esiste una formula per `g`. Non se ne conosce la derivata. L'unica cosa che
si può fare è chiedere il valore in un punto e pagarlo. In letteratura una
funzione così si chiama **black box costosa**, ed è la ragione per cui esistono
gli algoritmi di questo documento: se valutare fosse gratis, si campionerebbe a
tappeto e non ci sarebbe niente da discutere.

### Il budget

Ogni algoritmo riceve **120 valutazioni**, non una di più. Il confronto è a
parità di budget: quello che cambia è *come* le 120 valutazioni vengono spese.

### L'ODD: dove è possibile e quanto è probabile

**ODD** sta per *Operational Design Domain*, l'insieme delle condizioni in cui il
sistema dovrebbe funzionare. Nel progetto ha due componenti, e tenerle distinte è
essenziale:

- i **bounds**, cioè la scatola dei valori ammessi (per esempio angoli fra 0° e
  85°, `segment_length` fra 10 m e 40 m);
- le **marginali operative**, cioè con quale probabilità ciascun valore si
  presenta *dentro* la scatola. Le curve dolci sono molto più frequenti di quelle
  strette, le velocità intermedie più frequenti degli estremi.

La distinzione è quella fra «può capitare» e «capita spesso». Formalmente le
marginali definiscono una densità di probabilità **f(θ)** sullo spazio: alta dove
lo scenario è ordinario, bassa dove è inusuale.

### Fallimento raro

Un fallimento è **raro** quando avviene in un punto che l'ODD rende improbabile.
Operativamente: si estraggono 200 000 punti dalle marginali, se ne ordinano le
log-densità, e si prende il decile inferiore come soglia. Un fallimento il cui
`log f(θ)` sta sotto quella soglia è raro.

Questo definisce l'obiettivo vero. Non «trovare fallimenti» — sarebbe facile,
basta guidare a velocità assurde su curve impossibili — ma **trovare fallimenti
che il sistema può davvero incontrare, e che il campionamento cieco non
troverebbe quasi mai**.

---

## 2. Plain sampling: il pavimento

### Come funziona

L'algoritmo più semplice possibile, in tre righe:

1. estrai 120 punti dalle marginali operative dell'ODD;
2. simula ciascuno;
3. conta quanti hanno margine negativo.

Non c'è modello, non c'è adattamento, non c'è memoria: i 120 punti sono decisi
prima di vedere qualunque risultato. È il **metodo Monte Carlo** nella sua forma
elementare.

### Che cosa stima, e perché è corretto

La frazione di fallimenti osservata è una stima **non distorta** (unbiased) della
probabilità di fallimento reale `P(fail)`: se ripetessi l'esperimento infinite
volte, la media delle stime convergerebbe al valore vero. È la sua qualità
principale, e la ragione per cui è il termine di paragone: non fa assunzioni,
non può ingannarsi sistematicamente.

### Perché sugli eventi rari costa troppo

La sua debolezza è la varianza. Contando *n* estrazioni indipendenti di un evento
con probabilità *p*, il numero di successi ha deviazione standard `√(n·p·(1−p))`,
quindi l'incertezza **relativa** della stima è circa:

```
incertezza relativa ≈ 1 / √(numero di eventi osservati)
```

Nelle campagne del progetto `P(fail)` vale circa il 2.5%. Con 120 simulazioni si
osservano quindi **circa 3 fallimenti**, e `1/√3 ≈ 0.58`: la stima balla del
±58%. È esattamente ciò che si misura — il coefficiente di variazione del plain
sampling fra i 12 seed è 0.57–0.67.

E il problema peggiora quanto più l'evento è raro. Per stimare con incertezza
relativa del 10% servono ~100 eventi osservati; se `p = 10⁻⁴`, sono **un milione
di simulazioni**. A 30 secondi l'una fanno quasi un anno di calcolo.

> **Il punto in una frase.** Il campionamento cieco è corretto ma inefficiente:
> spende quasi tutto il budget in zone dove la risposta è già nota, e osserva
> l'evento che interessa troppe poche volte perché la stima sia utile.

### A cosa serve nel confronto

A due cose. È il **pavimento**: senza di lui, dire «ne trovo 41 ogni 100
simulazioni» non significherebbe nulla, perché mancherebbe il termine di
paragone. Ed è l'unica stima di `P(fail)` che non dipende da alcun modello,
quindi il metro con cui si giudicano le stime degli altri due.

---

## 3. Cross-entropy: cambiare la distribuzione da cui si estrae

Il metodo cross-entropy nasce per rispondere alla domanda «come stimo una
probabilità piccola senza spendere un milione di simulazioni?». Si costruisce in
tre passi.

### Passo 1 — Importance sampling

L'idea è di **estrarre da una distribuzione diversa** da quella vera, scegliendola
in modo da visitare più spesso la regione che interessa, e poi correggere il
conto per compensare l'imbroglio.

Sia `f` la densità vera (le marginali dell'ODD) e `q` una densità di nostra
scelta, detta **proposta**. Vale l'identità:

```
P(fallimento) = E_f[ 1(fallimento) ]  =  E_q[ 1(fallimento) · f(θ)/q(θ) ]
```

In parole: se estrai da `q` invece che da `f`, e pesi ogni punto con il rapporto
`w(θ) = f(θ)/q(θ)`, ottieni comunque una stima corretta della probabilità sotto
`f`. Il rapporto `w` si chiama **peso di importanza** e ha un'interpretazione
diretta: quanto quel punto è più (o meno) probabile nel mondo vero rispetto al
mondo da cui l'hai pescato.

Il guadagno è che con una `q` centrata sulla regione di fallimento si osservano
molti fallimenti anche con poche estrazioni, e la varianza della stima crolla.
Il rischio è simmetrico: con una `q` sbagliata i pesi diventano enormi e
sbilanciati, e la stima peggiora invece di migliorare.

### Passo 2 — La discesa cross-entropy: imparare la proposta

Resta il problema di scegliere `q`. Il metodo cross-entropy la **impara**,
avvicinandola per gradi alla regione di fallimento:

1. parti da una proposta `q₀` che assomiglia alla distribuzione vera (stessa
   media e stessa dispersione delle marginali);
2. estrai un blocco di punti da `q` e simulali;
3. ordina i margini ottenuti e tieni la frazione peggiore — per esempio il 20% —
   che si chiama **insieme elite**. La soglia che li separa si indica con **γ**
   (gamma);
4. ricalcola media e dispersione di `q` sui soli punti elite, pesandoli con i
   pesi di importanza;
5. torna al punto 2.

A ogni giro l'insieme elite è peggiore del precedente, quindi γ scende e `q` si
sposta verso la zona in cui il sistema fallisce. Ci si ferma quando γ raggiunge
la soglia di fallimento — la proposta ha centrato la regione — o quando restano
troppo pochi punti validi per stimare qualcosa.

Il nome viene dalla teoria dell'informazione: l'aggiornamento minimizza la
*cross-entropy* fra la proposta e la distribuzione ideale, che è la distribuzione
vera ristretta alla regione di fallimento.

### Passo 3 — La miscela difensiva

C'è un modo tipico in cui l'importance sampling si rompe. Se `q` si allontana
molto da `f`, in qualche punto `f(θ)/q(θ)` diventa enorme: un solo campione
domina la somma, e la stima dipende praticamente da lui. È la **degenerazione dei
pesi**.

La difesa è estrarre da una **miscela**:

```
d(θ) = α·f(θ) + (1−α)·q(θ)         con α = 0.2
```

cioè un quinto dei punti dalla distribuzione vera e quattro quinti dalla
proposta. Il peso diventa `f/d`, che è **limitato superiormente da 1/α = 5**
qualunque cosa faccia `q`: nessun campione può più dominare.

La quota estratta da `f` si chiama **campione difensivo**, e la sua numerosità
decide se lo stimatore è utilizzabile. Il codice controlla quel numero
(`MIN_DEFENSIVE_SAMPLES = 30`) e marca la stima come non utilizzabile quando è
troppo basso: con budget 120 i punti difensivi sono 9, quindi in queste campagne
la probabilità stimata dalla cross-entropy **non va letta**, mentre i fallimenti
che ha trovato restano validi.

### Una metrica di salute: l'ESS

L'**effective sample size** riassume quanto sono sbilanciati i pesi:

```
ESS = (Σ wᵢ)² / Σ wᵢ²
```

Vale *n* quando i pesi sono tutti uguali (campionamento perfettamente
equilibrato) e tende a 1 quando un solo peso domina. Si legge come «quanti
campioni indipendenti valgono davvero quelli che ho».

### Che cosa produce

Una stima di `P(fallimento)` con intervallo di confidenza, la proposta finale
(media e dispersione per ogni parametro: dove il metodo pensa che stiano i
fallimenti), e tutti i punti valutati lungo la discesa.

### Il suo limite strutturale

La cross-entropy deve restare su distribuzioni **assolutamente continue rispetto
a f**: se estraesse da regioni dove `f` è nulla, i pesi non sarebbero definiti e
lo stimatore perderebbe senso. Non può quindi spostarsi liberamente sulla
scatola, perché lo stimatore è la sua ragione d'essere. È una proprietà del
metodo, non una scelta implementativa, ed è ciò che lo distingue da active
boundary.

---

## 4. Active boundary: costruire un modello e interrogare quello

Il terzo algoritmo cambia strategia: invece di scegliere *da quale distribuzione*
estrarre, costruisce un **modello approssimato della funzione** e lo usa per
decidere, punto per punto, dove conviene simulare.

### Passo 1 — Il surrogato: un processo gaussiano

Un **modello surrogato** è un'approssimazione economica di una funzione cara. Lo
si addestra sui punti già valutati e poi lo si interroga liberamente, perché
interrogarlo non costa simulazioni.

Il surrogato usato è un **processo gaussiano** (GP). Detto senza formalismo: è un
modello di regressione che, dato un punto θ qualunque, non restituisce un solo
numero ma **due**:

- **μ(θ)** — il valore che il modello si aspetta per il margine;
- **σ(θ)** — quanto il modello è incerto su quel valore.

La seconda quantità è ciò che rende il GP adatto qui. Vicino ai punti già
simulati σ è piccola (il modello sa); lontano cresce (il modello sta
estrapolando). Un modello che dicesse solo μ non permetterebbe di distinguere «so
che qui va bene» da «non ne ho idea».

Il legame fra punti vicini è dato dal **kernel**, che formalizza l'assunzione «θ
simili producono margini simili». Il kernel usato è RBF con **ARD** (*automatic
relevance determination*): una lunghezza caratteristica per ogni parametro,
stimata dai dati. Un parametro con lunghezza corta è uno su cui il margine cambia
in fretta, cioè un parametro influente — da lì viene la classifica di importanza
dei parametri che il metodo produce gratis.

### Passo 2 — Dalla previsione alla probabilità di fallire

Poiché il GP restituisce una media e una deviazione, in ogni punto definisce una
distribuzione normale sul margine. La probabilità che il margine sia sotto la
soglia si legge quindi direttamente:

```
p(θ) = Φ( (soglia − μ(θ)) / σ(θ) )
```

dove Φ è la funzione di ripartizione della normale standard. È la stima del
modello che quello scenario fallisca. Vale ~0 dove il modello è sicuro che vada
bene, ~1 dove è sicuro che fallisca, e **~0.5 dove non sa**.

### Passo 3 — L'acquisizione: scegliere dove simulare

La **funzione di acquisizione** è il criterio con cui si sceglie il prossimo
punto da valutare. Qui è l'**entropia binaria** di quella probabilità:

```
H(p) = −[ p·log p + (1−p)·log(1−p) ]
```

`H` vale zero quando `p` è 0 o 1 e raggiunge il massimo a `p = 0.5`. Misura
**quanta incertezza c'è sull'esito**, non quanto l'esito è brutto: una certezza
non insegna nulla in nessuna delle due direzioni, mentre un punto a 50-50 dà il
massimo di informazione qualunque cosa succeda.

Poiché `p ≈ 0.5` accade dove il modello colloca il confine fra successo e
fallimento, massimizzare l'entropia significa in pratica **simulare lungo la
frontiera di fallimento**.

### Passo 4 — Diversità del batch

I punti si simulano a gruppi (16 alla volta), perché riaddestrare il GP dopo ogni
singola simulazione costerebbe troppo. Prendere semplicemente i 16 punteggi più
alti però non funziona: si ammasserebbero tutti nella stessa zona incerta, e
sedici copie della stessa domanda costano sedici simulazioni e rispondono a una.

La selezione è quindi **greedy con vincolo di distanza minima**: si prende il
punto migliore, si scartano i candidati troppo vicini, si prende il migliore fra
i rimasti, e così via.

### Il ciclo completo

```
40 punti iniziali, scelti senza modello        (design iniziale)
   ↓
ripeti 5 volte:
   addestra il GP su tutto ciò che hai
   genera 4000 candidati (non costano nulla: non vengono simulati)
   calcola μ, σ, p ed entropia per ciascuno
   scegli 16 punti ad alta entropia e distanti fra loro
   simulali e aggiungili al set
   ↓
addestramento finale
stima di P(fallimento) integrando il GP sull'ODD
```

Il conto del budget è `40 + 5 × 16 = 120`.

### La stima di P(fallimento) senza simulare

Alla fine si estraggono 8000 punti dalle marginali dell'ODD — che **non vengono
simulati**, solo dati in pasto al GP — e si media la probabilità puntuale:

```
P(fallimento) ≈ media su quegli 8000 punti di  Φ( (soglia − μ(θ)) / σ(θ) )
```

È una stima di natura diversa da quella del plain sampling: non una frequenza
osservata ma il valore atteso di una probabilità **modellata**. Le due rispondono
quasi alla stessa domanda e vanno riportate come grandezze distinte.

L'intervallo di credibilità si ottiene campionando funzioni dal posteriore del GP
e calcolando la frazione di fallimenti per ciascuna, il che propaga l'incertezza
del modello nella stima finale.

### Perché finisce nella coda dell'ODD

Il campionamento di active boundary è **uniforme sulla scatola**, non sull'ODD.
Nella scatola gli scenari improbabili occupano tanto volume quanto quelli
ordinari, quindi il metodo li visita spesso — e la frontiera che insegue passa
proprio per lì. È il motivo per cui i suoi fallimenti stanno intorno al 1.6°
percentile di verosimiglianza operativa, mentre quelli della cross-entropy stanno
al 21°.

---

## 5. Le due dimensioni ortogonali

Oltre alla famiglia di ricerca, il confronto varia due cose che agiscono
*dentro* ciascun algoritmo.

### Il disegno campionario: LHS contro random

È il modo in cui si estraggono i punti quando non c'è un modello a guidarli — il
design iniziale e i pool di candidati.

**Random** significa estrazioni indipendenti e uniformi. **LHS** (*Latin
Hypercube Sampling*) è una stratificazione: per estrarre *n* punti si divide ogni
asse in *n* fasce di uguale probabilità e si pesca esattamente un valore in
ciascuna, permutando poi le fasce in modo indipendente fra gli assi.

L'effetto è che nessuna fascia resta vuota. Il campionamento casuale, invece, ne
lascia scoperta circa il 37% — la probabilità che nessuno di *n* tiri cada in una
data fascia è `(1 − 1/n)ⁿ`, che tende a `1/e`.

La garanzia di LHS però è **su un asse alla volta**: assicura che qualche punto
abbia `angle_1` alto e qualche punto abbia `min_speed` alta, non che i due valori
capitino nello stesso punto. Su regioni di fallimento definite dalla
combinazione di più parametri il vantaggio svanisce, e c'è un secondo requisito:
gli strati sono larghi `1/n`, quindi una regione più sottile di uno strato è
invisibile alla stratificazione.

### L'acquisizione spenta: le configurazioni `+randacq`

Sono varianti di active boundary identiche in tutto — stesso design iniziale,
stesso pool, stesso budget, GP ancora addestrato a ogni iterazione — tranne che
il batch da simulare viene pescato **a caso dal pool** invece che per entropia
massima.

Servono a separare due contributi che il confronto diretto confonde: active
boundary campiona uniformemente sulla scatola mentre plain sampling estrae
dall'ODD, quindi parte del suo vantaggio potrebbe venire da *dove* guarda invece
che da *cosa* impara. Spegnendo la sola acquisizione si misura il secondo
contributo isolato.

---

## 6. I tre a confronto

| | plain sampling | cross-entropy | active boundary |
|---|---|---|---|
| **come sceglie i punti** | estrazione dall'ODD, decisa in anticipo | estrazione da una proposta che si sposta a ogni iterazione | il modello indica dove è più incerto |
| **usa i risultati precedenti?** | no | sì, per aggiornare la proposta | sì, per riaddestrare il surrogato |
| **dove campiona** | dalle marginali operative | da distribuzioni vicine alle marginali | uniformemente sulla scatola |
| **cosa stima** | una frequenza | una probabilità pesata per importanza | il valore atteso di una probabilità modellata |
| **prodotto principale** | il termine di paragone | `P(fallimento)` con intervallo | i punti sulla frontiera, e la sua forma |
| **si rompe quando** | l'evento è troppo raro per il budget | la proposta degenera e i pesi si sbilanciano | il surrogato non riesce ad approssimare `g` |
| **modulo** | `pipeline/model_comparison.py` | `pipeline/rare_event_random.py` | `pipeline/active_boundary_random.py` |

### Come leggerli insieme

I tre non ottimizzano lo stesso obiettivo, ed è la chiave per interpretare il
confronto. La cross-entropy punta a **stimare bene una probabilità**: concentra
gli sforzi dove la massa di probabilità è, e i suoi fallimenti stanno per
costruzione in scenari relativamente ordinari. Active boundary punta a **mappare
una frontiera**: non stima niente durante la ricerca, va dove è più incerto, e
finisce nella coda. Plain sampling non punta a nulla e per questo è il metro.

Una conseguenza pratica: se la domanda è «quanto spesso questo sistema fallisce
in esercizio», la risposta si legge dal campionamento cieco o dall'integrazione
del GP. Se la domanda è «quali scenari lo fanno fallire», serve un metodo che
vada a cercarli.

---

## 7. Dove sta il codice

| file | contenuto |
|---|---|
| `pipeline/samplers.py` | `LHSSampler` e `RandomSampler` dietro una sola interfaccia |
| `pipeline/active_boundary_random.py` | GP surrogato, acquisizione per entropia, selezione diversificata |
| `pipeline/rare_event_random.py` | discesa cross-entropy e stima con miscela difensiva |
| `pipeline/rare_event.py` | la stessa stima nella versione non parametrica sul campionatore |
| `pipeline/model_comparison.py` | l'orchestrazione delle campagne e `PlainSamplingBaseline` |
| `scenarios/common/driver.py` | il controllore che guida durante ogni simulazione |
| `scenarios/lane_keeping/qoi.py` | il margine di sicurezza `g(θ)` |
