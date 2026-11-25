"""
modulo_rajada/estacao_cnp.py - a estacao meteorologica automatica do proprio
CNP (Prefeitura de Boituva + Centro Paula Souza, instalada em dez/2025;
dados publicos no Klimadata): https://klimadata.com/pws/boituva
(-23,299121; -47,691638; 624 m)

COLETA (desde 24/11/2026). O Klimadata trocou o site antigo (WeeWX/Belchertown,
klimadata.com/wx/boituva/json/*.json, fora do ar desde 24/11 16h10) por uma
plataforma nova com API publica em JSON (klimadata.com/pub/estacoes/boituva/...):
  - serie?periodo=dia     ultimas 24 h (janela movel), blocos de 10 min;
  - serie?periodo=mes     ultimos 30 dias, por hora (periodo=semana: 7 dias, igual);
  - serie?periodo=ano     ultimo ano, por dia (carimbo: 00h local);
  - agora                 a ultima leitura (vai para a pagina pelo coletor da nuvem).
Vento e rajada em m/s, chuva em mm no bloco. Conferido contra os 5 min antigos em
24/11 (07h-16h10) e por hora de 26/10 a 24/11: e o mesmo dado bruto, reagregado.
  - O bloco de 10 min carimbado T junta os registros antigos T e T+5 (o carimbo
    antigo era o fim do intervalo): vento = media, rajada = maximo, chuva = soma
    (diferenca 0,01 km/h). Aqui ele vai para ts = T + 5 min, o fim do bloco, e a
    temperatura cai exatamente onde a antiga caia (:x5).
  - A radiacao nova e 1,050 x a antiga (55 pares, razao constante): ela e
    dividida por FATOR_RADIACAO para o cnp_kt seguir na escala em que a segunda
    etapa do Agora foi treinada.
  - A hora carimbada T e [T, T+1 h) dos registros antigos (vento |dif| 0,09 km/h,
    rajada 0,01, chuva corr 0,998).
So ha 10 min das ultimas 24 h: o que nao for guardado se perde (por hora, 30 dias).
Os 5 min vao ate 24/11 16h10; dali em diante a tabela e de 10 min (as janelas de
variaveis() sao por tempo e valem para os dois). coletar() e chamada pela
atualizacao da pagina e pela tarefa ColetaEstacao; falha aqui nunca impede a pagina.

HISTORIA DO SITE ANTIGO (troca de formato sem aviso, duas vezes antes desta):
em 07/11/2026 "barometer" virou "altimeter" e "rainRate" virou "rain", e a coleta
caiu de 07/11 16h20 a 15/11; em 17/11 o year.json deixou de ter o termo/radiacao
de 10 min do ano, e os pontos diarios das 12h entraram em cnp_termo_ano.parquet
como leituras (324 linhas, retiradas em 20/11; copia em
cnp_termo_ano_antes_limpeza_2026-11-20.parquet). O 10 min de 01/01 a 16/11/2026
so existe em cnp_termo_ano.parquet. Copias brutas da API em klimadata_pub/.

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
  cnp_5min.parquet       ts (UTC, fim do intervalo) + COLUNAS; 5 min ate 24/11/2026 16h10, depois 10 min
  cnp_horario.parquet    hora, ts (UTC, inicio da janela agregada [ts, ts + 1 h)), vento_kmh, rajada_kmh,
                         direcao, chuva_mm
  cnp_diario.parquet     dia + maximos/totais do dia (ate 23/11: vento_max_kmh, rajada_max_kmh, direcao,
                         chuva_mm do year.json; depois: rajada_max_kmh, vento_med_kmh, chuva_mm, temp_max_c,
                         temp_min_c da serie anual da API)
  cnp_termo_ano.parquet  ts (UTC), temp_c, orvalho_c, umidade, radiacao_wm2 de 10 min, 01/01 a 16/11/2026

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

URL_PUB = "https://klimadata.com/pub/estacoes/boituva/{}"
# a coleta de hora em hora, 24 h, roda tambem no GitHub Actions (repositorio da
# pagina, branch estacao-cnp): independe deste computador estar ligado
URL_GITHUB = ("https://raw.githubusercontent.com/fernandoliveiraf-sudo/previsao-cnp-boituva/"
              "estacao-cnp/dados/{}")
PASTA = C.DADOS / "estacao_cnp"
PASTA_HIST = PASTA / "historico"
ARQ_5MIN = PASTA / "cnp_5min.parquet"
ARQ_HORA = PASTA / "cnp_horario.parquet"
ARQ_DIA = PASTA / "cnp_diario.parquet"
ARQ_TERMO_ANO = PASTA / "cnp_termo_ano.parquet"     # 10 min do year.json ate 16/11/2026: temperatura, orvalho, umidade, radiacao
LAT, LON, ALT_M = -23.299121, -47.691638, 624
COLUNAS = ["vento_kmh", "rajada_kmh", "temp_c", "orvalho_c", "umidade", "chuva_mmh", "radiacao_wm2", "pressao_hpa",
           "qnh_hpa", "uv", "direcao"]
# API nova -> colunas: (campo da API, coluna, fator, essencial). Vento e rajada em m/s; a chuva (mm no
# bloco) vira taxa (mm/h) em cinco_min(). pressao_hpa (barometro, ao nivel do mar) nao existe mais: so QNH.
CAMPOS_PUB = [
    ("vento", "vento_kmh", 3.6, True), ("rajada", "rajada_kmh", 3.6, True),
    ("temperatura", "temp_c", 1, True), ("ponto_orvalho", "orvalho_c", 1, True),
    ("umidade", "umidade", 1, False), ("chuva", "chuva_mm", 1, True),
    ("radiacao", "radiacao_wm2", None, True), ("uv", "uv", 1, False),
    ("qnh", "qnh_hpa", 1, False), ("direcao", "direcao", 1, False),
]
FATOR_RADIACAO = 1.05                     # radiacao da API nova / da antiga (24/11/2026, 55 pares)
FIM_5MIN = pd.Timestamp("2026-11-24 19:10", tz="UTC")   # ultimo instante do site antigo (16h10 locais)

ATRASO_MIN = 5            # como o METAR no "Agora": so o que existia 5 min antes do intervalo
IDADE_MAX_MIN = 15        # leitura mais velha que isso nao vale
COLS_CNP_TERMO = ["cnp_lcl_m", "cnp_kt"]          # o que existe desde jan/2026 (sem vento)
COLS_CNP = ["cnp_raj_max_30", "cnp_raj_max_60", "cnp_vento_med_30", "cnp_tend_raj",
            "cnp_chovendo_15", "cnp_lcl_m", "cnp_kt"]


# ------------------------------------------------------------------ coleta
def _pub(caminho: str):
    req = urllib.request.Request(URL_PUB.format(caminho), headers={"User-Agent": "TCC-previsao-CNP (coleta horaria)",
                                                                   "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _serie_pub(periodo: str) -> tuple[pd.DataFrame, int, list[str]]:
    """serie?periodo=... da API: (tabela com ts = carimbo da API, UTC, e as colunas de CAMPOS_PUB nas
    unidades antigas; intervalo em segundos; campos essenciais que nao vieram). Campo que sumir fica vazio."""
    j = _pub(f"serie?periodo={periodo}")
    d = pd.DataFrame({"ts": pd.to_datetime(pd.Series(j["t"], dtype="int64"), unit="s", utc=True)})
    ausentes = []
    for campo, col, fator, essencial in CAMPOS_PUB:
        v = j.get(campo)
        if not isinstance(v, list) or len(v) != len(d):
            d[col] = np.nan
            if essencial:
                ausentes.append(campo)
            continue
        v = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float)
        d[col] = v / FATOR_RADIACAO if col == "radiacao_wm2" else v * fator
    return d, int(j.get("intervalo_s") or 0), ausentes


def cinco_min() -> pd.DataFrame:
    """As ultimas 24 h (serie?periodo=dia, blocos de 10 min) em ts + COLUNAS, na convencao da coleta
    antiga: ts = fim do bloco (carimbo da API + intervalo - 5 min), chuva em mm/h, radiacao na escala
    antiga. So instantes depois do fim do site antigo (FIM_5MIN): os 5 min de antes nao sao sobrescritos.
    d.attrs["ausentes"] lista os campos essenciais que nao vieram."""
    d, passo, ausentes = _serie_pub("dia")
    if passo <= 0:
        raise ValueError("serie do dia sem intervalo_s")
    d["ts"] = d.ts + pd.Timedelta(seconds=passo) - pd.Timedelta(minutes=5)
    d["chuva_mmh"] = d.chuva_mm * 3600 / passo
    d["pressao_hpa"] = np.nan
    d = d[d.ts > FIM_5MIN].dropna(subset=[c for c in COLUNAS if c != "pressao_hpa"], how="all")
    d = d[["ts", *COLUNAS]].reset_index(drop=True)
    d.attrs["ausentes"] = ausentes
    return d


def horario() -> pd.DataFrame:
    """Por hora, ultimos 30 dias (serie?periodo=mes): hora = ts = inicio da hora; vento medio, rajada
    maxima, direcao e chuva (mm) da hora. A hora corrente vem parcial e e regravada na coleta seguinte."""
    d, _, _ = _serie_pub("mes")
    return pd.DataFrame({"hora": d.ts, "ts": d.ts, "vento_kmh": d.vento_kmh, "rajada_kmh": d.rajada_kmh,
                         "direcao": d.direcao, "chuva_mm": d.chuva_mm})


def diario() -> pd.DataFrame:
    """Por dia local, ultimo ano ate ontem (serie?periodo=ano, carimbo 00h local)."""
    j = _pub("serie?periodo=ano")
    t = pd.to_datetime(pd.Series(j["t"], dtype="int64"), unit="s", utc=True).dt.tz_convert("America/Sao_Paulo")

    def num(campo, fator=1.0):
        v = j.get(campo)
        if not isinstance(v, list) or len(v) != len(t):
            return np.full(len(t), np.nan)
        return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float) * fator

    return pd.DataFrame({"dia": t.dt.strftime("%Y-%m-%d"), "rajada_max_kmh": num("rajada", 3.6),
                         "vento_med_kmh": num("vento", 3.6), "chuva_mm": num("chuva"),
                         "temp_max_c": num("temp_max"), "temp_min_c": num("temp_min")})


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


def _com_nuvem(velho: pd.DataFrame, nuvem: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Junta a coleta da nuvem a local. Instantes novos entram inteiros; nos que as duas tem, a nuvem
    preenche o que o local deixou vazio e corrige o que difere mais que o arredondamento dela (3 casas).
    Motivo: o PC coleta no minuto :00 e grava o ultimo instante ainda provisorio (termo e radiacao
    mudam depois; as vezes o vento ainda nem veio); a nuvem coleta em :01/:16/:31/:46 e regrava o
    instante a cada execucao. O year.json confirmou a nuvem em todos os casos de 02 a 19/11/2026.
    Devolve (tabela, instantes novos, valores corrigidos)."""
    v = velho.copy()
    v["ts"] = pd.to_datetime(v.ts, utc=True).astype("datetime64[ns, UTC]")
    n = nuvem.copy()
    n["ts"] = pd.to_datetime(n.ts, utc=True).astype("datetime64[ns, UTC]")
    v, n = v.set_index("ts"), n.drop_duplicates("ts", keep="last").set_index("ts")
    for c in COLUNAS:
        if c not in v:
            v[c] = np.nan
    cols = [c for c in COLUNAS if c in n]
    comum = v.index.intersection(n.index)
    corrigidos = 0
    if len(comum):
        a = v.loc[comum, cols].astype(float)
        b = n.loc[comum, cols].astype(float)
        troca = b.notna() & (a.isna() | (a - b).abs().gt(0.01))
        corrigidos = int(troca.to_numpy().sum())
        if corrigidos:
            v.loc[comum, cols] = a.mask(troca, b)
    novos = n.index.difference(v.index)
    tudo = pd.concat([v, n.loc[novos, cols]]).sort_index().reset_index()
    return tudo, len(novos), corrigidos


def importar_github() -> dict:
    """Traz o que o GitHub Actions coletou (inclusive com este computador
    desligado) e junta aos arquivos locais. Falha silenciosa: e reforco."""
    out = {"5min": 0, "5min_corrigidos": 0, "horario": 0}
    agora = pd.Timestamp.now(tz="UTC")
    for mes in pd.period_range("2026-10", agora.strftime("%Y-%m"), freq="M"):
        try:
            d = pd.read_csv(URL_GITHUB.format(f"cnp_5min_{mes}.csv"))
        except Exception:  # noqa: BLE001
            continue
        d["ts"] = pd.to_datetime(d.ts, utc=True)
        velho = pd.read_parquet(ARQ_5MIN) if ARQ_5MIN.exists() else pd.DataFrame(columns=["ts", *COLUNAS])
        tudo, novos, corrigidos = _com_nuvem(velho, d)
        if novos or corrigidos:
            tudo.to_parquet(ARQ_5MIN, index=False)
        out["5min"] += novos
        out["5min_corrigidos"] += corrigidos
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
    """Acrescenta o que houver de novo; devolve quantas linhas novas por arquivo.
    Cada parte e independente: a falha de uma (site fora, serie trocada) nao
    impede as outras de gravar."""
    PASTA.mkdir(parents=True, exist_ok=True)
    out = {}

    def parte(nome, f):
        try:
            out[nome] = f()
        except Exception as e:  # noqa: BLE001
            out[nome] = f"falhou: {type(e).__name__}: {e}"[:150]

    def _5min():
        d = cinco_min()
        if d.attrs.get("ausentes"):
            out["series_ausentes"] = d.attrs["ausentes"]
        return _acrescentar(ARQ_5MIN, d, "ts")

    parte("github", importar_github)
    parte("5min", _5min)
    parte("horario", lambda: _acrescentar(ARQ_HORA, horario(), "hora"))
    def _diario():
        # coluna a coluna: o valor novo vence, e as colunas que so o year.json tinha (vento maximo,
        # direcao) ficam nos dias antigos
        novo = diario().set_index("dia")
        velho = pd.read_parquet(ARQ_DIA).set_index("dia") if ARQ_DIA.exists() else novo.iloc[:0]
        tudo = novo.combine_first(velho).sort_index().reset_index()
        tudo.to_parquet(ARQ_DIA, index=False)
        return len(tudo) - len(velho)

    # a serie anual vai ate ontem: no maximo uma vez por dia
    ontem = (pd.Timestamp.now(tz="America/Sao_Paulo") - pd.Timedelta("1D")).date().isoformat()
    if not ARQ_DIA.exists() or pd.read_parquet(ARQ_DIA).dia.max() < ontem:
        parte("diario", _diario)
    return out


# ------------------------------------------------------------------ historico
_WEEWX = {"windSpeed": "vento_kmh", "windGust": "rajada_kmh", "outTemp": "temp_c", "dewpoint": "orvalho_c",
          "outHumidity": "umidade", "rainRate": "chuva_mmh", "radiation": "radiacao_wm2", "barometer": "pressao_hpa",
          "altimeter": "qnh_hpa", "UV": "uv", "windDir": "direcao"}


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
        elif novo in ("pressao_hpa", "qnh_hpa"):
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
