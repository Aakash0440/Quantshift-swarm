import numpy as np
from scipy import stats
from dataclasses import dataclass
from enum import Enum

class RegimeTier(Enum):
    STABLE   = "STABLE"
    WATCH    = "WATCH"
    CRITICAL = "CRITICAL"

@dataclass
class DriftResult:
    tier: RegimeTier
    drift_score: float
    p_value: float
    triggered_tests: list
    details: dict

class DriftDetector:
    def __init__(self, stable_p: float = 0.3, watch_p: float = 0.1, volatility_z_critical: float = 3.0):
        self.stable_p = stable_p
        self.watch_p = watch_p
        self.volatility_z_critical = volatility_z_critical

    def _ks_test(self, ref, cur):
        _, p = stats.ks_2samp(ref, cur)
        return p

    def _mmd(self, ref, cur):
        def rbf(x, y, sigma=1.0):
            return np.exp(-np.sum((x-y)**2)/(2*sigma**2))
        n, m = len(ref), len(cur)
        if n == 0 or m == 0: return 0.0
        xx = np.mean([rbf(ref[i], ref[j]) for i in range(min(n,50)) for j in range(min(n,50))])
        yy = np.mean([rbf(cur[i], cur[j]) for i in range(min(m,50)) for j in range(min(m,50))])
        xy = np.mean([rbf(ref[i], cur[j]) for i in range(min(n,50)) for j in range(min(m,50))])
        return max(0.0, 1.0 - float(xx+yy-2*xy)*10)

    def _chi_squared_test(self, ref, cur, bins=10):
        ref_hist, edges = np.histogram(ref, bins=bins)
        cur_hist, _ = np.histogram(cur, bins=edges)
        ref_hist = ref_hist + 1; cur_hist = cur_hist + 1
        _, p = stats.chisquare(cur_hist, f_exp=ref_hist*(cur_hist.sum()/ref_hist.sum()))
        return p

    def _volatility_zscore(self, ref, cur):
        ref_vol = np.std(ref); cur_vol = np.std(cur)
        if ref_vol == 0: return 0.0
        return abs(cur_vol - ref_vol) / ref_vol

    def detect(self, reference_prices: np.ndarray, current_prices: np.ndarray) -> DriftResult:
        ref_returns = np.diff(reference_prices) / reference_prices[:-1]
        cur_returns = np.diff(current_prices) / current_prices[:-1]
        results = {}; p_values = []

        if len(ref_returns) > 5 and len(cur_returns) > 5:
            p_ks = self._ks_test(ref_returns, cur_returns)
            results["ks_test"] = p_ks; p_values.append(p_ks)

        p_mmd = self._mmd(ref_returns, cur_returns)
        results["mmd"] = p_mmd; p_values.append(p_mmd)

        if len(ref_returns) > 10 and len(cur_returns) > 10:
            p_chi = self._chi_squared_test(ref_returns, cur_returns)
            results["chi_squared"] = p_chi; p_values.append(p_chi)

        vol_z = self._volatility_zscore(ref_returns, cur_returns)
        results["volatility_z"] = vol_z

        combined_p = float(np.exp(np.mean(np.log([max(p,1e-10) for p in p_values])))) if p_values else 1.0
        drift_score = 1.0 - combined_p
        triggered = [k for k, v in results.items() if isinstance(v, float) and v < self.watch_p]

        if vol_z >= self.volatility_z_critical: tier = RegimeTier.CRITICAL
        elif combined_p < self.watch_p:         tier = RegimeTier.CRITICAL
        elif combined_p < self.stable_p:        tier = RegimeTier.WATCH
        else:                                   tier = RegimeTier.STABLE

        return DriftResult(tier=tier, drift_score=round(drift_score,3),
            p_value=round(combined_p,4), triggered_tests=triggered, details=results)
