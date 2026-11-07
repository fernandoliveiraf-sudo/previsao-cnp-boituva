"""
modulo_satelite/msg.py - irradiancia do satelite MSG (EUMETSAT) sobre Boituva,
pela Satellite Radiation API do Open-Meteo, a cada 15 min. A MESMA funcao
monta as variaveis no treino (dados/meteo_estendido/satelite.csv) e ao vivo
(baixar_recente), como em goes.py.

O MSG chega ao Open-Meteo com ~1-2 h de atraso, por isso a espinha usa o MSG
com idade de 2 h (T15: soma +0,016 de skill ao GOES de 5 min; IC 95% de
+0,007 a +0,028).

Variaveis (as do T14):
  kt        indice de claridade = GHI / radiacao no topo da atmosfera
  kt_1h     media de kt na ultima hora (4 carimbos)
  kt_3h     media nas ultimas 3 h
  kt_dif1h  variacao em 1 h (abrindo ou fechando)
  kt_dp1h   desvio em 1 h (nuvem quebrada)
  dkt_mod   kt_1h observado - kt do modelo (IFS) na mesma hora: o satelite
            viu mais ou menos nuvem que o modelo?
"""
from __future__ import annotations

import io
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

LAT, LON = -23.2983, -47.6924
VARIAVEIS = ["kt", "kt_1h", "kt_3h", "kt_dif1h", "kt_dp1h", "dkt_mod"]
URL_SATELITE = "https://satellite-api.open-meteo.com/v1/archive"
URL_FORECAST = "https://api.open-meteo.com/v1/forecast"
# o MSG do LSA SAF, a cada 15 min: e o mesmo dado do treino (baixado como
# "satellite_radiation_seamless", que ate out/2026 era este produto; conferido
# igual, diferenca 0 em 2025 e 2026). Em 03/11/2026 o "seamless" passou a vir
# vazio e em carimbos de 10 min - fica so como reserva.
MODELOS_MSG = ["eumetsat_lsa_saf_msg", "satellite_radiation_seamless"]


def _limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.split(" (")[0] for c in df.columns]
    df["time"] = pd.to_datetime(df.time, utc=True)
    return df.set_index("time").sort_index()


def variaveis_msg(sat: pd.DataFrame, ghi_modelo: pd.Series) -> pd.DataFrame:
    """sat: carimbos de 15 min (UTC) com shortwave_radiation e
    terrestrial_radiation; ghi_modelo: radiacao horaria do IFS (UTC, media da
    hora anterior, como no Open-Meteo). Mesma conta do T14."""
    toa = sat.terrestrial_radiation.where(sat.terrestrial_radiation > 50)   # sol baixo: kt instavel
    kt = (sat.shortwave_radiation / toa).clip(0, 1.2)
    ghi_mod = ghi_modelo.reindex(sat.index, method="bfill", limit=3)
    toa_h = toa.rolling(4, min_periods=2).mean()                            # topo da atmosfera na hora
    kt_mod = (ghi_mod / toa_h).clip(0, 1.2)
    f = pd.DataFrame({"kt": kt}, index=sat.index)
    f["kt_1h"] = kt.rolling(4, min_periods=2).mean()
    f["kt_3h"] = kt.rolling(12, min_periods=4).mean()
    f["kt_dif1h"] = kt - kt.shift(4)
    f["kt_dp1h"] = kt.rolling(4, min_periods=3).std()
    f["dkt_mod"] = f.kt_1h - kt_mod
    return f


def juntar(instantes: pd.Series, f: pd.DataFrame, idade_h: float) -> pd.DataFrame:
    """Para cada instante decidido (UTC), as variaveis como estavam em
    instante - idade (so carimbos <= esse instante, ate 30 min antes)."""
    ref = (pd.to_datetime(pd.Series(instantes), utc=True) - pd.Timedelta(hours=idade_h)).rename("t_ref")
    a = pd.DataFrame({"t_ref": ref.reset_index(drop=True), "pos": np.arange(len(ref))}).sort_values("t_ref")
    m = pd.merge_asof(a, f.reset_index().rename(columns={f.index.name or "index": "t_sat"}),
                      left_on="t_ref", right_on="t_sat", direction="backward",
                      tolerance=pd.Timedelta("30min")).sort_values("pos")
    out = m[VARIAVEIS].reset_index(drop=True)
    out["t_msg"] = m.t_sat.to_numpy()
    return out


# ------------------------------------------------------------- historico
def historico(pasta_meteo) -> pd.DataFrame:
    """Variaveis do treino, dos CSV baixados por pipeline/01_baixar_meteo.py."""
    sat = _limpar(pd.read_csv(f"{pasta_meteo}/satelite.csv"))
    h = _limpar(pd.read_csv(f"{pasta_meteo}/hist_forecast.csv"))
    return variaveis_msg(sat, h.shortwave_radiation)


# --------------------------------------------------------------- ao vivo
def _csv(url: str, params: dict) -> pd.DataFrame:
    for i in range(4):
        try:
            r = requests.get(url, params=dict(params, format="csv", timezone="UTC"), timeout=60)
            if r.status_code == 200:
                d = pd.read_csv(io.StringIO(r.text.split("\n", 2)[2]))
                return _limpar(d.rename(columns={d.columns[0]: "time"}))
        except requests.RequestException:
            pass
        time.sleep(3 * 2 ** i)
    raise RuntimeError(f"Open-Meteo sem resposta: {url}")


def baixar_recente(agora: pd.Timestamp | None = None) -> pd.DataFrame:
    """Ao vivo: MSG de ontem e hoje + a radiacao do IFS das mesmas horas ->
    as variaveis, com a mesma funcao do treino."""
    agora = agora or pd.Timestamp.now(tz="UTC")
    ini, fim = (agora - timedelta(days=1)).date(), agora.date()
    sat = None
    for modelo in MODELOS_MSG:            # o primeiro que trouxer algum valor
        sat = _csv(URL_SATELITE, {"latitude": LAT, "longitude": LON, "start_date": str(ini),
                                  "end_date": str(fim), "models": modelo,
                                  "temporal_resolution": "native",
                                  "hourly": "shortwave_radiation,terrestrial_radiation"})
        if sat.shortwave_radiation.notna().any():
            break
    # os carimbos que ainda nao chegaram vem vazios no fim: corta-os, para o
    # asof nao pegar um carimbo vazio no lugar do ultimo observado
    ok = sat.shortwave_radiation.notna()
    sat = sat.loc[:ok[ok].index.max()] if ok.any() else sat.iloc[:0]
    # past_days conta a partir de hoje; um `agora` no passado (conferencia de
    # paridade) precisa voltar ate o dia dele
    atras = (pd.Timestamp.now(tz="UTC").date() - ini).days + 1
    mod = _csv(URL_FORECAST, {"latitude": LAT, "longitude": LON, "models": "ecmwf_ifs025",
                              "past_days": min(max(atras, 2), 92), "forecast_days": 1,
                              "hourly": "shortwave_radiation"})
    return variaveis_msg(sat, mod.shortwave_radiation)
