"""
modulo_satelite/segunda_etapa.py - segunda etapa do quadro "Agora" com a
estacao do proprio CNP (T41).

Uma logistica por limiar do CORN (D, C, B, AI/A) com o logit do modelo do
"Agora" como offset: aprende so QUANTO a estacao corrige a chance, com poucos
parametros (3 coeficientes por limiar) e sinal forcado - base de nuvem e sol
maiores nunca fecham. Feito para pouca amostra: a estacao so tem historico
termico desde jan/2026. No T41: +0,060 de skill na validacao cruzada (IC
+0,026 a +0,096), mas ajustando com meses posteriores ao bloco avaliado; so com
o passado, como aqui, +0,01 a +0,02 (T53), crescendo com o historico.

Variaveis (modulo_rajada/estacao_cnp.variaveis): cnp_lcl_m (base estimada,
125 m x (T - Td)), o indicador "base >= 2.500 ft" (762 m, a regra de nuvem da
norma) e cnp_kt (radiacao medida / ceu limpo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

COLS = ["cnp_lcl_m", "cnp_kt"]
SINAIS = [+1, +1, +1]
L2 = 1.0


def matriz(V: pd.DataFrame) -> np.ndarray:
    lcl = np.clip(V.cnp_lcl_m.to_numpy(float), 0, 3000)
    return np.column_stack([lcl, (lcl >= 762).astype(float), np.clip(V.cnp_kt.to_numpy(float), 0, 1.3)])


def condicionais(p):
    S = np.column_stack([np.ones(len(p)), p[:, ::-1].cumsum(1)[:, ::-1][:, 1:]])
    return np.clip(S[:, 1:] / np.clip(S[:, :-1], 1e-9, None), 1e-4, 1 - 1e-4)


def de_condicionais(c):
    S = np.column_stack([np.ones(len(c)), np.cumprod(c, 1), np.zeros(len(c))])
    return S[:, :-1] - S[:, 1:]


class SegundaEtapa:
    def ajustar(self, V: pd.DataFrame, p: np.ndarray, y: np.ndarray):
        self.par = {}
        cond = condicionais(p)
        X0 = matriz(V)
        self.med = pd.DataFrame(X0).median().fillna(0).to_numpy()
        self.n = len(V)
        for k in range(1, 5):
            sel = y >= k - 1
            X = np.where(np.isnan(X0[sel]), self.med, X0[sel])
            mu, dp = X.mean(0), np.where(X.std(0) < 1e-9, 1, X.std(0))
            Z, off, alvo = (X - mu) / dp, logit(cond[sel, k - 1]), (y[sel] >= k).astype(float)

            def f(w):
                z = off + w[0] + Z @ w[1:]
                s = expit(z) - alvo
                return (np.logaddexp(0, z) - alvo * z).sum() + L2 * (w[1:] ** 2).sum(), \
                    np.concatenate([[s.sum()], Z.T @ s + 2 * L2 * w[1:]])
            lim = [(None, None)] + [(0, None) if s > 0 else (None, 0) for s in SINAIS]
            self.par[k] = (minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=lim).x, mu, dp)
        return self

    def prever(self, V: pd.DataFrame, p: np.ndarray) -> np.ndarray:
        cond = condicionais(p)
        X = matriz(V)
        X = np.where(np.isnan(X), self.med, X)
        out = np.empty_like(cond)
        for k in range(1, 5):
            w, mu, dp = self.par[k]
            out[:, k - 1] = expit(logit(cond[:, k - 1]) + w[0] + ((X - mu) / dp) @ w[1:])
        return de_condicionais(out)
