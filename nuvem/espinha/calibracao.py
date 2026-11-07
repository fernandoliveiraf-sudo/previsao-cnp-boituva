"""
calibracao.py - calibrador do modelo de decisao (Tentativa 14).

A isotonica pura (y_min=0), ajustada so em nov/2025, devolvia probabilidade
EXATAMENTE 0 pra escores abaixo do menor positivo visto; pela regra da
cadeia do CORN isso zerava todas as categorias numa hora (pagina de 06/10,
10h). A media isotonica + Platt nunca chega a 0 e, com 5 seeds, nao piora o
skill (0.263 +/- 0.015 contra 0.250 +/- 0.013; diferenca dentro do ruido) -
ver tentativas/14_calibracao_decisao/RESULTADO.md.

Fica num modulo proprio porque o pickle do modelo guarda a classe: quem
carrega modelo_reduzido.pkl precisa conseguir importar `calibracao`.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Platt:
    """Regressao logistica no logit do escore bruto (Platt 1999)."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e3)

    @staticmethod
    def _x(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        self.lr.fit(self._x(p), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(self._x(p))[:, 1]


class CalibradorMedia:
    """Media 50/50 de isotonica e Platt. Mesma interface da IsotonicRegression
    (fit/predict), entao prever() nao muda."""

    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        self.platt = Platt().fit(p, y)
        return self

    def predict(self, p):
        return 0.5 * self.iso.predict(p) + 0.5 * self.platt.predict(p)
