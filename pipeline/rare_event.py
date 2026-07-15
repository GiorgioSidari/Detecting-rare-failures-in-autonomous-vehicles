from __future__ import annotations

"""
Stima efficiente di eventi rari (livello massimo, M4) — metodo Cross-Entropy.

Problema: stimare P(fallimento) = P(margine QoI < soglia) sotto la distribuzione
operativa realistica f, quando questa probabilita' e' BASSA. Il Monte Carlo ingenuo
e' inefficiente: per vedere anche solo pochi fallimenti servono moltissimi run (a
~10 s/run sul simulatore vero, ore). Vedi scripts/validate_rare_event.py per la
dimostrazione (a P~1e-4 l'MC ingenuo e' inutile a parita' di budget).

Idea (Cross-Entropy + Importance Sampling):
  1. Si parte con una proposta q = f.
  2. A ogni iterazione si campiona da q, si valutano i margini, si selezionano gli
     "elite" (i piu' vicini/dentro al fallimento — soglia gamma abbassata verso 0),
     e si RI-ADATTANO i parametri di q verso la regione di fallimento (aggiornamento
     cross-entropy = momenti pesati degli elite, pesi = f/q).
  3. Quando gamma raggiunge la soglia di fallimento, si stima P con importance
     sampling usando la proposta finale, in modo che i campioni cadano dove i
     fallimenti sono. Cosi' pochi run bastano.

Robustezza: la stima finale usa una DEFENSIVE MIXTURE  d = alpha*f + (1-alpha)*q,
cosi' i pesi f/d sono limitati (<= 1/alpha) e la copertura del supporto e' garantita
(evita la sottostima tipica di una proposta troppo stretta). Il CI e' bootstrap
(l'estimatore IS a P piccola e' asimmetrico: il CI normale sarebbe ottimista).

LIMITE NOTO: la proposta e' un prodotto di distribuzioni UNIMODALI indipendenti. Se
la regione di fallimento e' MULTIMODALE (es. "almeno uno di N parametri estremo"),
questa CE la copre male e sottostima. In quei casi si usa una proposta a MISCELA o
la Subset Simulation (vedi roadmap). Per regioni a modo singolo funziona bene.

Nota: NON valida il simulatore; qui si assume che il margine (via QoI) sia gia'
affidabile — la fedelta' di cadenza e' un prerequisito risolto altrove.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import stats


@dataclass
class RareEventResult:
    p_fail: float                       # stima di P(fallimento) sotto f
    ci: tuple                           # (lo, hi) intervallo bootstrap 95%
    n_evaluations: int                  # run totali usati (budget)
    iterations: int                     # iterazioni CE eseguite
    q_loc: np.ndarray                   # loc della proposta finale (per dimensione)
    q_scale: np.ndarray                 # scale della proposta finale
    gamma_history: list = field(default_factory=list)   # soglie gamma per iterazione
    n_fail_effective: int = 0           # n. campioni finali in fallimento (diagnostico)


def scenario_margin_fn(scenario):
    """
    Costruisce una funzione margine(params)->margini dallo scenario reale, chiamando
    il simulatore e la QoI (come fa l'orchestrator). params: (M, d) -> margini (M,).
    I run non validi tornano NaN (gia' gestito da compute_qoi) e vengono filtrati a valle.
    """
    def _margin(params: np.ndarray) -> np.ndarray:
        traj = scenario.run_simulation(params)
        return np.asarray(scenario.compute_qoi(traj, params), dtype=float)
    return _margin


def _build_q(lo, hi, loc, scale):
    """Proposta = prodotto di truncnorm su [lo, hi] con (loc, scale) per dimensione."""
    scale = np.maximum(scale, 1e-9)
    a = (lo - loc) / scale
    b = (hi - loc) / scale
    return [stats.truncnorm(a[j], b[j], loc=loc[j], scale=scale[j]) for j in range(len(lo))]


def _logpdf_product(dists, X):
    """Somma dei log-pdf per dimensione: log della densita' prodotto in X (M,d)->(M,)."""
    lp = np.zeros(X.shape[0])
    for j, d in enumerate(dists):
        lp += d.logpdf(X[:, j])
    return lp


def _sample_product(dists, n, rng, d):
    """Campiona n punti dal prodotto di distribuzioni (rng condiviso -> dim indipendenti)."""
    X = np.empty((n, d))
    for j in range(d):
        X[:, j] = dists[j].rvs(size=n, random_state=rng)
    return X


def _bootstrap_ci(h, rng, n_boot=4000):
    """CI percentile 95% via bootstrap sulla media di h (contributi IS per campione)."""
    n = len(h)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = h[idx].mean(axis=1)
    return (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def estimate_failure_probability(
    margin_fn,
    f_dists,
    lower,
    upper,
    threshold: float = 0.0,
    samples_per_iter: int = 500,
    rho: float = 0.2,
    max_iter: int = 30,
    final_samples: int = 8000,
    alpha: float = 0.2,
    scale_floor: float = 0.03,
    seed: int = 0,
    verbose: bool = False,
) -> RareEventResult:
    """
    Stima P(margine < threshold) sotto la distribuzione f (lista di distribuzioni
    scipy congelate, una per dimensione), con la Cross-Entropy + defensive-mixture IS.

    Parametri
    ---------
    margin_fn        : callable((M,d)) -> (M,) margini. NaN = run non valido (filtrato).
    f_dists          : distribuzione operativa reale (es. scenario.param_distributions()).
    lower, upper     : bound per la troncatura della proposta.
    threshold        : soglia di fallimento (default 0.0: margine < 0).
    samples_per_iter : campioni per iterazione CE.
    rho              : frazione elite (quantile di margine per abbassare gamma).
    final_samples    : campioni della stima finale (defensive mixture).
    alpha            : quota di mixture campionata da f (0<alpha<1): limita i pesi a 1/alpha.
    scale_floor      : scala minima della proposta (frazione del range) anti-collasso.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    d = len(lo)

    # Init proposta = momenti di f (q0 ~ f in forma)
    loc = np.array([float(fd.mean()) for fd in f_dists])
    scale = np.array([float(fd.std()) for fd in f_dists])
    scale_min = scale_floor * (hi - lo)

    n_eval = 0
    gamma_hist: list = []

    # ── Fase CE: sposta q verso la regione di fallimento ──
    for _ in range(max_iter):
        q = _build_q(lo, hi, loc, scale)
        X = _sample_product(q, samples_per_iter, rng, d)
        m = np.asarray(margin_fn(X), dtype=float)
        n_eval += samples_per_iter
        ok = np.isfinite(m)
        Xv, mv = X[ok], m[ok]
        if mv.size < 2:
            break
        gamma = max(float(np.quantile(mv, rho)), threshold)
        gamma_hist.append(gamma)
        Xe = Xv[mv <= gamma]
        if Xe.shape[0] < 2:
            break
        # pesi di importanza degli elite (per l'aggiornamento CE), stabilizzati
        logw = _logpdf_product(f_dists, Xe) - _logpdf_product(q, Xe)
        w = np.exp(logw - logw.max())
        w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        loc = np.clip((w[:, None] * Xe).sum(axis=0), lo, hi)
        scale = np.maximum(np.sqrt((w[:, None] * (Xe - loc) ** 2).sum(axis=0)), scale_min)
        if verbose:
            print(f"  [CE] gamma={gamma:+.4f}  elite={Xe.shape[0]}  eval={n_eval}", flush=True)
        if gamma <= threshold:
            break   # raggiunta la soglia di fallimento

    # ── Stima finale: importance sampling con defensive mixture d=alpha*f+(1-alpha)*q ──
    q = _build_q(lo, hi, loc, scale)
    n_f = int(alpha * final_samples)
    X = np.vstack([_sample_product(f_dists, n_f, rng, d),
                   _sample_product(q, final_samples - n_f, rng, d)])
    m = np.asarray(margin_fn(X), dtype=float)
    n_eval += final_samples
    ok = np.isfinite(m)
    Xv, mv = X[ok], m[ok]
    # peso = f/d = 1 / (alpha + (1-alpha) * q/f), stabile in log-spazio
    log_qf = _logpdf_product(q, Xv) - _logpdf_product(f_dists, Xv)
    w = 1.0 / (alpha + (1.0 - alpha) * np.exp(log_qf))
    fail = (mv < threshold).astype(float)
    h = fail * w
    p_hat = float(h.mean()) if h.size else 0.0
    ci = _bootstrap_ci(h, rng)

    return RareEventResult(
        p_fail=p_hat,
        ci=ci,
        n_evaluations=n_eval,
        iterations=len(gamma_hist),
        q_loc=loc,
        q_scale=scale,
        gamma_history=gamma_hist,
        n_fail_effective=int(fail.sum()),
    )
