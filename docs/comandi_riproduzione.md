# Riproducibilità

Materiale di servizio di `risultati_sperimentazione.md`: dove stanno i risultati,
quali comandi li generano, quale modulo fa cosa.


## File dei risultati

| file | contenuto |
|---|---|
| `results/cmp_rare.*`, `cmp_ab12.*` | campagne Udacity (§1) |
| `results/cmp_md12.*` | MetaDrive, ODD largo, 12 seed |
| `results/cmp_md12_narrow.*` | MetaDrive, ODD ristretto, controllore ideale |
| `results/cmp_md12_narrow_lag2.*` | MetaDrive, ODD ristretto, degradato (§2.2) |
| `results/cmp_md12_ctrl.*`, `cmp_md12_ctrl_rand.*` | controlli di acquisizione (§2.3) |
| `results/cmp_md12_2x2.*` | unione verificata delle due, base del disegno 2×2 |
| `results/cmp_md12_full.*` | unione verificata a 8 configurazioni — **la classifica di §2.1** |
| `results/rank_*.json/.txt` | classifiche + sequenze pre-registrate |
| `results/taratura_*.json` | punti operativi, con ODD e leva registrati |
| `results/confronto_cross.json` | cross-simulatore a design condiviso (§3.5) |
| `docs/preregistrazione_*.json` | i piani, committati prima dei dati |

`scripts/rank_arms.py` **non simula**: rilegge il `.npz` e ricalcola classifica,
test appaiati e sequenza pre-registrata in una decina di secondi. Ogni tabella di
`risultati_sperimentazione.md` è rigenerabile senza toccare un simulatore.

## Comandi

```powershell
# --- campagne MetaDrive ---
.venv310\Scripts\python.exe scripts\run_model_comparison_md.py --budget 120 ^
       --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --out results\cmp_md12

.venv310\Scripts\python.exe scripts\run_model_comparison_md.py --budget 120 ^
       --seeds 0 1 2 3 4 5 6 7 8 9 10 11 ^
       --from-calibration results\taratura_md_narrow_lag_odd.json ^
       --out results\cmp_md12_narrow_lag2

# --- controlli di acquisizione (§2.3) ---
.venv310\Scripts\python.exe scripts\run_model_comparison_md.py --arms ab_lhs ab_lhs_randacq plain_lhs --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --budget 120 --out results\cmp_md12_ctrl
.venv310\Scripts\python.exe scripts\run_model_comparison_md.py --arms ab_random ab_random_randacq plain_random --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --budget 120 --out results\cmp_md12_ctrl_rand

# --- unioni verificate ---
.venv310\Scripts\python.exe scripts\merge_campaigns.py results\cmp_md12_ctrl_raw.npz results\cmp_md12_ctrl_rand_raw.npz --out results\cmp_md12_2x2 --verify-against results\cmp_md12_raw.npz
.venv310\Scripts\python.exe scripts\merge_campaigns.py results\cmp_md12_raw.npz results\cmp_md12_ctrl_raw.npz results\cmp_md12_ctrl_rand_raw.npz --out results\cmp_md12_full --verify-against results\cmp_md12_raw.npz
.venv310\Scripts\python.exe scripts\rank_arms.py results\cmp_md12_full_raw.npz --out results\rank_md12_full

# --- campagne Udacity (i flag ODD sono ricostruiti dai theta salvati) ---
python scripts\run_model_comparison.py --budget 120 --seeds 0 1 2 3 4 5 ^
       --max-angle 8 --max-speed 9.6 --max-seg 12 --out results\cmp_rare
python scripts\run_model_comparison.py --arms ab_lhs ab_random --budget 120 ^
       --seeds 0 1 2 3 4 5 6 7 8 9 10 11 ^
       --max-angle 8 --max-speed 9.6 --max-seg 12 --out results\cmp_ab12

# --- taratura del punto operativo ---
.venv310\Scripts\python.exe scripts\calibrate_operating_point.py lane_keeping_md ^
       --lever obs_lag --sampling odd --max-angle 8 --max-speed 9.6 --max-seg 12 ^
       --n 120 --low 0.02 --high 0.10 --scale-min 0.15 --scale-max 0.32 ^
       --out results\taratura_md_narrow_lag_odd.json

# --- classifiche e sequenze pre-registrate (nessuna simulazione) ---
python scripts\rank_arms.py results\cmp_md12_raw.npz --regions ^
       --plan docs\preregistrazione_classifica.json --out results\rank_md12
python scripts\rank_arms.py results\cmp_md12_narrow_lag2_raw.npz --regions ^
       --plan docs\preregistrazione_replica_degradata.json --out results\rank_md12_narrow_lag2
python scripts\rank_arms.py results\cmp_rare_raw.npz --regions ^
       --max-angle 8 --max-speed 9.6 --max-seg 12 --out results\rank_rare

# --- LHS vs random su regioni analitiche (§4.2) ---
python scripts\validate_model_comparison.py --skip-part2 --reps 4000

# --- parita' geometrica e confronto cross-simulatore ---
python scripts\diag_road_parity.py --n 60 --seed 42
.venv310\Scripts\python.exe scripts\run_cross_simulator.py collect lane_keeping_md --n 60 --speed-scale 0.1766
python scripts\run_cross_simulator.py collect lane_keeping --n 60 --speed-scale 0.42
python scripts\run_cross_simulator.py compare results\cross_lane_keeping_md.json results\cross_lane_keeping.json

# --- test ---
python -m pytest tests\test_arm_ranking.py tests\test_random_search_comparison.py -q
```

## Codice

| modulo | ruolo |
|---|---|
| `pipeline/samplers.py` | `LHSSampler` / `RandomSampler` dietro una sola interfaccia |
| `pipeline/active_boundary*.py` | GP surrogato e acquisizione sulla frontiera, versione parametrica sul campionatore |
| `pipeline/rare_event*.py` | cross-entropy + importance sampling, requisito sul budget difensivo |
| `pipeline/model_comparison.py` | harness delle campagne, `PlainSamplingBaseline`, ordine mescolato, diagnostico di deriva |
| `pipeline/arm_ranking.py` | rarità, test appaiati fra tutte le coppie, sequenza pre-registrata |
| `pipeline/failure_regions.py` | clustering, `axis_spread`, `structure_score` |
| `pipeline/operating_point.py` | taratura per bisezione, due leve, ODD registrato |
| `pipeline/cross_simulator.py` | confronto Udacity ↔ MetaDrive su design condiviso |
| `scenarios/common/` | i quattro moduli condivisi fra i backend |
| `scenarios/lane_keeping_md/` | il backend MetaDrive |
| `scripts/rank_arms.py` | classifica ed esecuzione del piano, senza simulare |
| `scripts/merge_campaigns.py` | unione con verifica di riproduzione seed per seed |
| `tests/test_arm_ranking.py` | 33 test, incluse tutte le regressioni |
