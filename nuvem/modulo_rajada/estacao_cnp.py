"""
modulo_rajada/estacao_cnp.py - a estacao meteorologica automatica do proprio
CNP (Prefeitura de Boituva + Centro Paula Souza, instalada em dez/2025;
dados publicos no Klimadata, WeeWX 5.4 com o tema Belchertown):
https://klimadata.com/wx/boituva/  (-23,299121; -47,691638; 624 m)

COLETA. O site so deixa no ar:
  - json/day.json   so o dia corrente: vento medio e rajada a cada 5 min
                    (chart4 "haysChart": [t, vento, rajada], km/h), chuva
                    (taxa, 5 min), temperatura, ponto de orvalho, umidade,
                    radiacao e pressao (10 min);
  - json/week.json  vento, rajada e direcao por hora, ultimos 7 dias;
  - json/year.json  um valor por dia desde 01/01/2026 (o maximo do dia).
Entao a serie de 5 min so existe se for guardada todo dia: coletar() e
chamada pela atualizacao da pagina (de hora em hora, 06-17 h) e acrescenta o
que for novo, sem duplicar. Falha aqui nunca impede a pagina.

HISTORICO. Se o Klimadata mandar o arquivo da estacao (pedido por e-mail),
basta po-lo em dados/estacao_cnp/historico/: o banco do WeeWX (.sdb/.sqlite,
tabela archive) ou um CSV com as colunas do WeeWX (dateTime, usUnits,
windSpeed, windGust, ...) ou ja com as colunas de COLUNAS abaixo.
carregar_5min() junta historico e coleta (o historico vence onde se cruzam).

VARIAVEIS (T28). variaveis() monta, para cada intervalo, o que a estacao
dizia ate ATRASO_MIN antes do inicio dele: rajada maxima de 30 e 60 min,
vento medio de 30 min, tendencia da rajada, chuva nos ultimos 15 min, base
estimada das nuvens baixas (125 m x (T - Td)) e radiacao medida / radiacao de
ceu limpo (Haurwitz). Leitura com mais de IDADE_MAX_MIN fica vazia.

Saidas (dados/estacao_cnp/, fora do git):
  cnp_5min.parquet     ts (UTC) + COLUNAS
  cnp_horario.parquet  hora, ts (UTC, fim da hora agregada), vento_kmh, rajada_kmh, direcao
  cnp_diario.parquet   dia, vento_max_kmh, rajada_max_kmh, direcao (do year.json)

Uso:  python modulo_rajada/estacao_cnp.py          # coleta agora
"""
from __future__ import annotations

import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
import caminhos as C  # noqa: E402

URL = "https://klimadata.com/wx/boituva/json/{}.json"
# a coleta de hora em hora, 24 h, roda tambem no GitHub Actions (repositorio da
# pagina, branch estacao-cnp): independe deste computador estar ligado
URL_GITHUB = ("https://raw.githubusercontent.com/fernandoliveiraf-sudo/previsao-cnp-boituva/"
              "estacao-cnp/dados/{}")
PASTA = C.DADOS / "estacao_cnp"
PASTA_HIST = PASTA / "historico"
ARQ_5MIN = PASTA / "cnp_5min.parquet"
ARQ_HORA = PASTA / "cnp_horario.parquet"
ARQ_DIA = PASTA / "cnp_diario.parquet"
ARQ_TERMO_ANO = PASTA / "cnp_termo_ano.parquet"     # 10 min do year.json: temperatura, orvalho, umidade, radiacao
LAT, LON, ALT_M = -23.299121, -47.691638, 624
COLUNAS = ["vento_kmh", "rajada_kmh", "temp_c", "orvalho_c", "umidade", "chuva_mmh", "radiacao_wm2", "pressao_hpa"]

ATRASO_MIN = 5            # como o METAR no "Agora": so o que existia 5 min antes do intervalo
IDADE_MAX_MIN = 15        # leitura mais velha que isso nao vale
COLS_CNP_TERMO = ["cnp_lcl_m", "cnp_kt"]          # o que existe desde jan/2026 (sem vento)
COLS_CNP = ["cnp_raj_max_30", "cnp_raj_max_60", "cnp_vento_med_30", "cnp_tend_raj",
            "cnp_chovendo_15", "cnp_lcl_m", "cnp_kt"]


# ------------------------------------------------------------------ coleta
def _baixar(nome: str) -> dict:
    req = urllib.request.Request(URL.format(nome), headers={"User-Agent": "TCC-previsao-CNP (coleta horaria)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _serie(j: dict, grafico: str, nome: str, colunas: list[str]) -> pd.DataFrame:
    d = pd.DataFrame(j[grafico]["series"][nome]["data"], columns=["ts", *colunas])
    d["ts"] = pd.to_datetime(d.ts, unit="ms", utc=True)
    return d.dropna(subset=colunas, how="all")


def cinco_min() -> pd.DataFrame:
    j = _baixar("day")
    partes = [_serie(j, "chart4", "haysChart", ["vento_kmh", "rajada_kmh"]),
              _serie(j, "chart3", "rainRate", ["chuva_mmh"]),
              _serie(j, "chart1", "outTemp", ["temp_c"]),
              _serie(j, "chart1", "dewpoint", ["orvalho_c"]),
              _serie(j, "chart2", "outHumidity", ["umidade"]),
              _serie(j, "chart6", "radiation", ["radiacao_wm2"]),
              _serie(j, "chart7", "barometer", ["pressao_hpa"])]
    d = partes[0]
    for p in partes[1:]:
        d = d.merge(p, on="ts", how="outer")
    return d.sort_values("ts").reset_index(drop=True)


def horario() -> pd.DataFrame:
    j = _baixar("week")
    v = _serie(j, "chart5", "windSpeed", ["vento_kmh"])
    g = _serie(j, "chart5", "windGust", ["rajada_kmh"])
    dr = _serie(j, "chart5", "windDir", ["direcao"])
    d = v.merge(g, on="ts", how="outer").merge(dr, on="ts", how="outer")
    # o site agrega horas que terminam no minuto em que gerou o arquivo (:25, :35...):
    # uma linha por hora cheia, a da coleta mais recente
    d.insert(0, "hora", d.ts.dt.floor("h"))
    return d


def diario(j: dict | None = None) -> pd.DataFrame:
    j = j or _baixar("year")
    v = _serie(j, "chart5", "windSpeed", ["vento_max_kmh"])
    g = _serie(j, "chart5", "windGust", ["rajada_max_kmh"])
    dr = _serie(j, "chart5", "windDir", ["direcao"])
    d = v.merge(g, on="ts", how="outer").merge(dr, on="ts", how="outer")
    # o ponto do dia vem carimbado no meio do dia local
    d.insert(0, "dia", d.ts.dt.tz_convert("America/Sao_Paulo").dt.date.astype(str))
    return d.drop(columns="ts")


def _acrescentar(arq: Path, novo: pd.DataFrame, chave: str) -> int:
    if arq.exists():
        velho = pd.read_parquet(arq)
        antes = len(velho)
        tudo = pd.concat([velho, novo], ignore_index=True)
    else:
        antes, tudo = 0, novo
    # o valor mais recente de um instante vence (o da hora corrente pode ser parcial)
    tudo = tudo.drop_duplicates(chave, keep="last").sort_values(chave).reset_index(drop=True)
    tudo.to_parquet(arq, index=False)
    return len(tudo) - antes


def importar_github() -> dict:
    """Traz o que o GitHub Actions coletou (inclusive com este computador
    desligado) e junta aos arquivos locais. Falha silenciosa: e reforco."""
    out = {"5min": 0, "horario": 0}
    agora = pd.Timestamp.now(tz="UTC")
    for mes in pd.period_range("2026-10", agora.strftime("%Y-%m"), freq="M"):
        try:
            d = pd.read_csv(URL_GITHUB.format(f"cnp_5min_{mes}.csv"))
        except Exception:  # noqa: BLE001
            continue
        d["ts"] = pd.to_datetime(d.ts, utc=True)
        velho = pd.read_parquet(ARQ_5MIN) if ARQ_5MIN.exists() else pd.DataFrame(columns=["ts"])
        novos = d[~d.ts.isin(pd.to_datetime(velho.ts, utc=True))]           # a coleta local vence onde se cruza
        if len(novos):
            out["5min"] += _acrescentar(ARQ_5MIN, pd.concat([novos, velho], ignore_index=True), "ts")
    try:
        h = pd.read_csv(URL_GITHUB.format("cnp_horario.csv"))
        h["hora"], h["ts"] = pd.to_datetime(h.hora, utc=True), pd.to_datetime(h.ts, utc=True)
        velho = pd.read_parquet(ARQ_HORA) if ARQ_HORA.exists() else pd.DataFrame(columns=["hora"])
        novos = h[~h.hora.isin(pd.to_datetime(velho.hora, utc=True))]
        if len(novos):
            out["horario"] = _acrescentar(ARQ_HORA, pd.concat([novos, velho], ignore_index=True), "hora")
    except Exception:  # noqa: BLE001
        pass
    return out


def coletar() -> dict:
    """Acrescenta o que houver de novo; devolve quantas linhas novas por arquivo."""
    PASTA.mkdir(parents=True, exist_ok=True)
    try:
        gh = importar_github()
    except Exception:  # noqa: BLE001
        gh = {}
    out = {"github": gh, "5min": _acrescentar(ARQ_5MIN, cinco_min(), "ts"),
           "horario": _acrescentar(ARQ_HORA, horario(), "hora")}
    hoje = pd.Timestamp.now(tz="America/Sao_Paulo").date().isoformat()
    if not ARQ_DIA.exists() or pd.read_parquet(ARQ_DIA).dia.max() < hoje:
        j = _baixar("year")                                       # 36 MB: no maximo uma vez por dia
        out["diario"] = _acrescentar(ARQ_DIA, diario(j), "dia")
        out["termo_ano"] = len(historico_termo(j))                # preenche as horas que a coleta de 5 min perdeu
    return out


def historico_termo(j: dict | None = None) -> pd.DataFrame:
    """Temperatura, orvalho, umidade e radiacao a cada 10 min desde 01/01/2026
    (year.json: o site guarda o ano corrente nessa resolucao, ao contrario do
    vento; o Klimadata confirmou em 26/10/2026 que nao ha outro historico).
    Acrescenta em ARQ_TERMO_ANO sem apagar o que ja havia: na virada do ano o
    year.json deve recomecar do zero."""
    j = j or _baixar("year")
    partes = [_serie(j, "chart1", "outTemp", ["temp_c"]), _serie(j, "chart1", "dewpoint", ["orvalho_c"]),
              _serie(j, "chart2", "outHumidity", ["umidade"]), _serie(j, "chart6", "radiation", ["radiacao_wm2"])]
    d = partes[0]
    for x in partes[1:]:
        d = d.merge(x, on="ts", how="outer")
    PASTA.mkdir(parents=True, exist_ok=True)
    _acrescentar(ARQ_TERMO_ANO, d, "ts")
    return d


# ------------------------------------------------------------------ historico
_WEEWX = {"windSpeed": "vento_kmh", "windGust": "rajada_kmh", "outTemp": "temp_c", "dewpoint": "orvalho_c",
          "outHumidity": "umidade", "rainRate": "chuva_mmh", "radiation": "radiacao_wm2", "barometer": "pressao_hpa"}


def _de_weewx(a: pd.DataFrame) -> pd.DataFrame:
    """Tabela archive do WeeWX -> COLUNAS, convertendo pelo usUnits
    (1 = US, 16 = METRIC, 17 = METRICWX)."""
    out = pd.DataFrame({"ts": pd.to_datetime(a.dateTime.astype("int64"), unit="s", utc=True)})
    u = a.usUnits.to_numpy() if "usUnits" in a else np.full(len(a), 16)
    vel = np.select([u == 1, u == 17], [1.609344, 3.6], 1.0)            # -> km/h
    chuva = np.select([u == 1, u == 16], [25.4, 10.0], 1.0)              # -> mm/h
    pres = np.where(u == 1, 33.8639, 1.0)                                # -> hPa
    for orig, novo in _WEEWX.items():
        if orig not in a:
            out[novo] = np.nan
            continue
        v = pd.to_numeric(a[orig], errors="coerce").to_numpy(float)
        if novo in ("vento_kmh", "rajada_kmh"):
            v = v * vel
        elif novo in ("temp_c", "orvalho_c"):
            v = np.where(u == 1, (v - 32) * 5 / 9, v)
        elif novo == "chuva_mmh":
            v = v * chuva
        elif novo == "pressao_hpa":
            v = v * pres
        out[novo] = v
    return out


def carregar_historico() -> pd.DataFrame:
    partes = []
    for arq in sorted(PASTA_HIST.glob("*")) if PASTA_HIST.exists() else []:
        if arq.suffix.lower() in (".sdb", ".sqlite", ".db"):
            with sqlite3.connect(arq) as con:
                partes.append(_de_weewx(pd.read_sql("select * from archive", con)))
        elif arq.suffix.lower() == ".csv":
            a = pd.read_csv(arq)
            if "dateTime" in a:
                partes.append(_de_weewx(a))
            else:
                a["ts"] = pd.to_datetime(a.ts, utc=True)
                partes.append(a[["ts", *[c for c in COLUNAS if c in a]]])
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=["ts", *COLUNAS])


def carregar_5min() -> pd.DataFrame:
    """Historico (se houver) + coleta, um registro por instante."""
    col = pd.read_parquet(ARQ_5MIN) if ARQ_5MIN.exists() else pd.DataFrame(columns=["ts"])
    if ARQ_TERMO_ANO.exists():                 # o 10 min do ano entra por baixo da coleta
        col = pd.concat([pd.read_parquet(ARQ_TERMO_ANO), col], ignore_index=True)
    hist = carregar_historico()
    tudo = pd.concat([col, hist], ignore_index=True) if len(hist) else col.copy()
    tudo["ts"] = pd.to_datetime(tudo.ts, utc=True)
    for c in COLUNAS:
        if c not in tudo:
            tudo[c] = np.nan
    return tudo.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)[["ts", *COLUNAS]]


# ------------------------------------------------------------------ variaveis
def _ceu_limpo(ts: pd.DatetimeIndex) -> np.ndarray:
    """Radiacao global de ceu limpo (Haurwitz), W/m2."""
    sys.path.insert(0, str(C.PUBLICAR))
    from gerar_previsao_publica import elevacao_solar  # noqa: PLC0415
    sen = np.sin(np.radians(np.asarray(elevacao_solar(ts), float)))
    return np.where(sen > 0.17, 1098 * sen * np.exp(-0.057 / np.clip(sen, 1e-3, None)), np.nan)   # sol > 10 graus


def variaveis(ts_utc: pd.Series, tab: pd.DataFrame, atraso_min: int = ATRASO_MIN) -> pd.DataFrame:
    """Para cada instante t (inicio do intervalo), o que a estacao dizia ate
    t - atraso_min. Mesmo codigo no treino e ao vivo."""
    t = pd.to_datetime(pd.Series(ts_utc), utc=True).reset_index(drop=True) - pd.Timedelta(minutes=atraso_min)
    s = tab.dropna(subset=["rajada_kmh"]).set_index("ts").sort_index()
    if s.empty:                                # so termometria (historico sem vento)
        s = pd.DataFrame({"rajada_kmh": [np.nan], "vento_kmh": [np.nan], "chuva_mmh": [np.nan]},
                         index=pd.DatetimeIndex([pd.Timestamp("2000-01-01", tz="UTC")], name="ts"))
    roll = pd.DataFrame({
        "raj_max_30": s.rajada_kmh.rolling("30min").max(),
        "raj_max_60": s.rajada_kmh.rolling("60min").max(),
        "vento_med_30": s.vento_kmh.rolling("30min").mean(),
        "chov_15": (s.chuva_mmh.fillna(0) > 0).astype(float).rolling("15min").max(),
    }).reset_index().rename(columns={"ts": "t_obs"})
    m = tab.dropna(subset=["temp_c", "orvalho_c"]).set_index("ts").sort_index()
    termo = pd.DataFrame({"spread": m.temp_c - m.orvalho_c}).reset_index().rename(columns={"ts": "t_termo"})
    r = tab.dropna(subset=["radiacao_wm2"]).set_index("ts").sort_index()
    rad = pd.DataFrame({"rad_30": r.radiacao_wm2.rolling("30min").mean()}).reset_index().rename(columns={"ts": "t_rad"})

    a = pd.DataFrame({"t": t, "pos": np.arange(len(t))}).sort_values("t")
    tol = pd.Timedelta(minutes=IDADE_MAX_MIN)
    j = pd.merge_asof(a, roll, left_on="t", right_on="t_obs", direction="backward", tolerance=tol)
    a30 = a.assign(t=a.t - pd.Timedelta(minutes=30))
    j30 = pd.merge_asof(a30, roll[["t_obs", "raj_max_30"]], left_on="t", right_on="t_obs",
                        direction="backward", tolerance=tol)
    jt = pd.merge_asof(a, termo, left_on="t", right_on="t_termo", direction="backward",
                       tolerance=pd.Timedelta(minutes=25))
    jr = pd.merge_asof(a, rad, left_on="t", right_on="t_rad", direction="backward",
                       tolerance=pd.Timedelta(minutes=25))
    j, j30, jt, jr = (x.sort_values("pos").reset_index(drop=True) for x in (j, j30, jt, jr))
    ceu = _ceu_limpo(pd.DatetimeIndex(t - pd.Timedelta(minutes=15)))
    return pd.DataFrame({
        "cnp_raj_max_30": j.raj_max_30.to_numpy(float),
        "cnp_raj_max_60": j.raj_max_60.to_numpy(float),
        "cnp_vento_med_30": j.vento_med_30.to_numpy(float),
        "cnp_tend_raj": (j.raj_max_30 - j30.raj_max_30).to_numpy(float),
        "cnp_chovendo_15": j.chov_15.to_numpy(float),
        "cnp_lcl_m": (125 * jt.spread).clip(lower=0).to_numpy(float),
        "cnp_kt": np.clip(jr.rad_30.to_numpy(float) / ceu, 0, 1.5),
        "cnp_idade_min": ((j.t - j.t_obs).dt.total_seconds() / 60 + atraso_min).to_numpy(float),
    })


if __name__ == "__main__":
    print(coletar())
    for arq in (ARQ_5MIN, ARQ_HORA, ARQ_DIA):
        d = pd.read_parquet(arq)
        print(f"{arq.name}: {len(d)} linhas, {d.iloc[0, 0]} -> {d.iloc[-1, 0]}")
