import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class WindowResult:
    window: int
    train_start: str
    test_start: str
    sharpe: float
    win_rate: float
    max_drawdown: float
    total_return: float

class WalkForwardValidator:
    def __init__(self, n_windows: int = 12, train_months: int = 6, test_months: int = 1):
        self.n_windows = n_windows
        self.train_months = train_months
        self.test_months = test_months

    def run(self, df: pd.DataFrame, model_fn) -> list:
        results = []
        df = df.sort_index()
        total_months = self.n_windows + self.train_months
        for w in range(self.n_windows):
            train_end = w + self.train_months
            test_end  = train_end + self.test_months
            train_slice = df.iloc[:int(len(df)*train_end/total_months)]
            test_slice  = df.iloc[int(len(df)*train_end/total_months):int(len(df)*test_end/total_months)]
            if len(train_slice) < 100 or len(test_slice) < 10:
                continue
            predictor = model_fn(train_slice)
            returns = []
            for _, row in test_slice.iterrows():
                pred = predictor(row)
                actual_return = row.get("target", 0)
                if pred.get("direction") == "UP":   returns.append(actual_return)
                elif pred.get("direction") == "DOWN": returns.append(-actual_return)
            if not returns:
                continue
            returns_arr = np.array(returns)
            sharpe   = float(np.mean(returns_arr)/(np.std(returns_arr)+1e-9)*np.sqrt(252))
            win_rate = float(np.mean(returns_arr > 0))
            cumulative = np.cumprod(1 + returns_arr)
            max_dd = float(np.min(cumulative/np.maximum.accumulate(cumulative)) - 1)
            results.append(WindowResult(
                window=w, train_start=str(train_slice.index[0])[:10],
                test_start=str(test_slice.index[0])[:10],
                sharpe=round(sharpe,2), win_rate=round(win_rate,3),
                max_drawdown=round(max_dd,3), total_return=round(float(cumulative[-1]-1),3),
            ))
        return results

    def summary(self, results: list) -> dict:
        if not results: return {}
        sharpes = [r.sharpe for r in results]
        return {
            "mean_sharpe": round(float(np.mean(sharpes)),2),
            "std_sharpe":  round(float(np.std(sharpes)),2),
            "mean_win_rate": round(float(np.mean([r.win_rate for r in results])),3),
            "n_positive_windows": sum(1 for s in sharpes if s > 0),
            "n_windows": len(results),
        }
