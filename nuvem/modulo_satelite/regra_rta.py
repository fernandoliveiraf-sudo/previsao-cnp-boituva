"""
modulo_satelite/regra_rta.py - a norma do CNP, como o RTAG a enunciou no
podcast "Deu Liga" (dados/podcast/ACHADOS_CRITERIOS.md), aplicada a uma
observacao: devolve o "nivel pela norma" (0 a 4) e as variaveis que o compoem.

  vento   fecha a partir de: AI/A 22 km/h (rodovia) / 26 (matos); B 26 / 29;
          C 31; D 35 - como o lado da rodovia nao foi dito, saem dois niveis:
          estrito (limites da rodovia) e brando (da mata)
  nuvem   mais de 5/8 abaixo de 2.500 ft fecha tudo; BKN/OVC em qualquer altura
          fecha AI/A (sem visual na reta)
  chuva, trovoada ou CB: fecha

Hoje a fonte e o METAR de Sorocaba (SDCO); a funcao recebe o formato de
redemet.metar_iem (station, valid, wxcodes, metar), entao a estacao do CNP
pode entrar no lugar quando tiver historico. No T39, estas variaveis somadas
ao modelo do quadro "Agora" deram +0,009 de skill na CV (IC +0,002 a +0,016).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ESTACAO = "SDCO"
ATRASO_MIN = 5
IDADE_MAX_MIN = 75
LIM_ESTRITO = [(35, 0), (31, 1), (26, 2), (22, 3)]      # km/h -> nivel (rodovia)
LIM_BRANDO = [(35, 0), (31, 1), (29, 2), (26, 3)]       # (matos)
OKTAS = {"FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV": 8}
COLS_REGRA = ["regra_estrita", "regra_branda", "regra_vel_kmh", "regra_dir_sin", "regra_dir_cos",
              "regra_chuva", "regra_trovoada", "regra_cob_abaixo_2500", "regra_cob_max", "regra_idade_min"]


def _nivel_vento(v, lim):
    for corte, nv in lim:
        if v >= corte:
            return nv
    return 4


def regras_metar(m: pd.DataFrame) -> pd.DataFrame:
    """Cada METAR -> variaveis e nivel pela norma do CNP."""
    m = m.reset_index(drop=True)
    txt = m.metar.fillna("")
    wx = m.wxcodes.fillna("")
    v = txt.str.extract(r"\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT\b")
    vel = np.fmax(pd.to_numeric(v[2], errors="coerce").fillna(0), pd.to_numeric(v[1], errors="coerce")) * 1.852
    dire = np.radians(pd.to_numeric(v[0], errors="coerce"))
    cob_baixa, cob_max = [], []
    for t in txt:
        camadas = [(OKTAS[c], int(h) * 100) for c, h in re.findall(r"\b(FEW|SCT|BKN|OVC|VV)(\d{3})", t)]
        cob_baixa.append(max([o for o, h in camadas if h < 2500], default=0))
        cob_max.append(max([o for o, _ in camadas], default=0))
    chuva = wx.str.contains("RA|DZ|SH").to_numpy()
    trov = (wx.str.contains("TS") | txt.str.contains(r"\bVCTS\b|CB\b")).to_numpy()
    out = pd.DataFrame({"valid": pd.to_datetime(m.valid, utc=True), "regra_vel_kmh": vel.to_numpy(float),
                        "regra_dir_sin": np.sin(dire).to_numpy(), "regra_dir_cos": np.cos(dire).to_numpy(),
                        "regra_chuva": chuva.astype(float), "regra_trovoada": trov.astype(float),
                        "regra_cob_abaixo_2500": cob_baixa, "regra_cob_max": cob_max})
    for nome, lim in (("regra_estrita", LIM_ESTRITO), ("regra_branda", LIM_BRANDO)):
        nv = np.array([_nivel_vento(x, lim) if np.isfinite(x) else 4 for x in out.regra_vel_kmh])
        nv = np.where(out.regra_cob_max >= 6, np.minimum(nv, 3), nv)
        nv = np.where(out.regra_cob_abaixo_2500 > 5, 0, nv)
        nv = np.where(chuva | trov, 0, nv)
        out[nome] = nv.astype(float)
    return out


def juntar(ts_utc: pd.Series, regras: pd.DataFrame) -> pd.DataFrame:
    """Para cada instante, a norma pela ultima observacao ate ATRASO_MIN antes
    (no maximo IDADE_MAX_MIN de idade)."""
    t = pd.to_datetime(pd.Series(ts_utc), utc=True).reset_index(drop=True) - pd.Timedelta(minutes=ATRASO_MIN)
    a = pd.DataFrame({"t": t, "pos": np.arange(len(t))}).sort_values("t")
    j = pd.merge_asof(a, regras.sort_values("valid"), left_on="t", right_on="valid", direction="backward",
                      tolerance=pd.Timedelta(minutes=IDADE_MAX_MIN)).sort_values("pos")
    j["regra_idade_min"] = (j.t - j.valid).dt.total_seconds() / 60 + ATRASO_MIN
    return j[COLS_REGRA].reset_index(drop=True)


def da_tabela_metar(tab: pd.DataFrame) -> pd.DataFrame:
    """Tabela de METAR (varias estacoes) -> regras da estacao de referencia."""
    return regras_metar(tab[tab.station == ESTACAO])
