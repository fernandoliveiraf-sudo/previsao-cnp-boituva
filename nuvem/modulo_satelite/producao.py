"""
modulo_satelite/producao.py - a decisao do intervalo atual com o satelite
(T15): a espinha + a nuvem do GOES-East com 5 min de atraso + a irradiancia
do MSG com 2 h de idade.

Medido no T15 (validacao cruzada em 3 blocos, 197 dias): +0,059 de skill
sobre a espinha (IC 95% de +0,041 a +0,079); +0,055 nos dias de nuvem. Sem
MSG, o GOES sozinho dá +0,043 (IC +0,027 a +0,061) - e o modelo reserva.

T26: somar o METAR (REDEMET) de Sorocaba, Campinas e Jundiai, o ultimo com
horario ate 5 min antes do intervalo (no maximo 75 min), da mais +0,042 de
skill (IC +0,026 a +0,059). Sem METAR de Sorocaba nesse prazo, vale o modelo
sem METAR.

T39: a norma do CNP (regras do RTAG no podcast) aplicada ao METAR de Sorocaba
entra nos modelos com METAR (modulo_satelite/regra_rta.py): +0,009 de skill
na CV (IC +0,002 a +0,016).

T41: por cima do modelo com METAR, a SEGUNDA ETAPA da estacao do proprio CNP
(base de nuvem estimada e sol medido, a cada 10 min desde jan/2026;
modulo_satelite/segunda_etapa.py): +0,01 a +0,02 de skill na CV ajustando so com
o passado (T53; o +0,060 do T41 usava meses posteriores no ajuste).
Sem leitura da estacao nos ultimos 25 min, vale o modelo sem ela.

T57/T60: a chance publicada e a MEDIA do intervalo atual com os 2 anteriores
do mesmo dia (janela de 45 min), cada um com as entradas que existiam na sua
hora - imita a espera do RTA antes de mudar a decisao. +0,008 de skill na CV
(IC +0,004 a +0,012), empate no teste, +0,014 de skill e +1,4 ponto de acerto
nos anuncios do grupo (so conferencia).

Vale so para o intervalo de 15 min que comeca ate 5 min depois de agora: o
modelo foi treinado com a ultima imagem criada ate 5 min ANTES do inicio do
intervalo, e ao vivo isso so existe para esse intervalo. Os seguintes
precisariam de uma imagem que ainda nao foi feita.

Uso:
    python modulo_satelite/producao.py treinar       # gera modelo_satelite.pkl
    python modulo_satelite/producao.py agora         # o bloco da pagina, ao vivo
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(AQUI))
import caminhos as C  # noqa: E402
import goes  # noqa: E402
import msg  # noqa: E402
import redemet  # noqa: E402
import regra_rta  # noqa: E402
import segunda_etapa  # noqa: E402

sys.path.insert(0, str(RAIZ / "modulo_rajada"))
import estacao_cnp  # noqa: E402

sys.path.insert(0, str(C.ESPINHA))
import treinar_decisao_reduzida as T  # noqa: E402

MODELO = AQUI / "modelo_satelite.pkl"
ENTRADAS_HOJE = C.DADOS / "cache" / "entradas_espinha_hoje.parquet"   # gravado pela pagina
ARQ_GOES = C.DADOS / "satelite" / "goes_acha_boituva.parquet"
ATRASO_GOES_MIN = 5
IDADE_MSG_H = 2
INICIO = "2025-01-01"                   # antes disso nao ha satelite
SEMENTE = 42                            # como a espinha em producao
HORA_INI, HORA_FIM = 8, 17              # janela de operacao (local), a mesma da pagina
NIVEL_MINIMO = {"D": 1, "C": 2, "B": 3, "A": 4, "AI": 4}
TZ = "America/Sao_Paulo"
AEROPORTOS_METAR = ["SDCO", "SBKP", "SBJD"]     # Sorocaba, Campinas, Jundiai
ARQ_METAR = C.DADOS / "metar" / "metar_redemet.csv"   # testes/T21_radar/baixar.py
ATRASO_METAR_MIN = 5
IDADE_MAX_METAR_MIN = 75
VARS_METAR = ("chovendo", "trovoada", "teto_ft", "visib_m", "vento_kt", "rajada_kt")
COLS_METAR = [f"metar_{st}_{v}" for st in AEROPORTOS_METAR for v in VARS_METAR]


def _base() -> list[str]:
    with open(C.ESPINHA / "modelo_reduzido.pkl", "rb") as fh:
        return pickle.load(fh)["features"]


def metar_vars(m: pd.DataFrame) -> pd.DataFrame:
    """METAR no formato da IEM (redemet.metar_iem) -> as variaveis do T26."""
    txt = m.metar.fillna("").reset_index(drop=True)
    wx = m.wxcodes.fillna("").reset_index(drop=True)
    teto = []
    for t in txt:
        cam = [int(h) * 100 for _, h in re.findall(r"\b(BKN|OVC|VV)(\d{3})", t)]
        teto.append(min(cam) if cam else (20000 if ("CAVOK" in t or re.search(r"\b(FEW|SCT|NSC|SKC)\d{0,3}", t))
                                         else np.nan))
    vis = txt.str.extract(r"KT(?:\s+\d{3}V\d{3})?\s+(\d{4})\b")[0].astype(float)
    vis = np.where(txt.str.contains("CAVOK"), 9999, vis)
    sknt, gust = m.sknt.to_numpy(float), m.gust.to_numpy(float)
    return pd.DataFrame({"station": m.station.to_numpy(), "valid": pd.to_datetime(m.valid, utc=True).to_numpy(),
                         "chovendo": wx.str.contains("RA|DZ|SH").astype(float),
                         "trovoada": wx.str.contains("TS").astype(float),
                         "teto_ft": teto, "visib_m": vis, "vento_kt": sknt, "rajada_kt": np.fmax(gust, sknt)})


def juntar_metar(ts_utc: pd.Series, tab: pd.DataFrame) -> pd.DataFrame:
    """Para cada instante, o ultimo METAR de cada aeroporto com horario ate
    ATRASO_METAR_MIN antes (no maximo IDADE_MAX_METAR_MIN)."""
    t = pd.to_datetime(pd.Series(ts_utc), utc=True).reset_index(drop=True) - pd.Timedelta(minutes=ATRASO_METAR_MIN)
    a = pd.DataFrame({"t": t, "pos": np.arange(len(t))}).sort_values("t")
    tab = tab.assign(valid=pd.to_datetime(tab.valid, utc=True))
    col = {}
    for st in AEROPORTOS_METAR:
        g = tab[tab.station == st].sort_values("valid")
        r = pd.merge_asof(a, g, left_on="t", right_on="valid", direction="backward",
                          tolerance=pd.Timedelta(minutes=IDADE_MAX_METAR_MIN)).sort_values("pos")
        for v in VARS_METAR:
            col[f"metar_{st}_{v}"] = r[v].to_numpy(float)
        if st == AEROPORTOS_METAR[0]:
            col["t_metar"] = r["valid"].to_numpy()
    return pd.DataFrame(col)


def montar_treino(com_metar: bool = True) -> pd.DataFrame:
    """As linhas do treino com as variaveis dos dois satelites, como no T15
    (e o METAR, como no T26)."""
    df = T.carregar(str(C.achar(C.DATASET_ESTENDIDO)))
    df = df[df.ts >= INICIO].reset_index(drop=True)
    g = goes.variaveis(pd.read_parquet(ARQ_GOES), df.ts_utc, ATRASO_GOES_MIN)
    m = msg.juntar(df.ts_utc, msg.historico(C.achar(C.METEO_ESTENDIDO)), IDADE_MSG_H)
    partes = [df, g, m[msg.VARIAVEIS]]
    if com_metar:
        tab = pd.read_csv(ARQ_METAR)
        partes.append(juntar_metar(df.ts_utc, metar_vars(tab))[COLS_METAR])
        partes.append(regra_rta.juntar(df.ts_utc, regra_rta.da_tabela_metar(tab)))
    return pd.concat(partes, axis=1)


def treinar():
    base = _base()
    df = montar_treino()
    d = df.ts
    tr = df[d < C.CORTE_VAL].reset_index(drop=True)
    va = df[(d >= C.CORTE_VAL) & (d < C.CORTE_TESTE)].reset_index(drop=True)
    te = df[d >= C.CORTE_TESTE].reset_index(drop=True)
    clima, y = T.climatologia(tr, te), te.alvo.values
    pacote = {"K": 4, "atraso_goes_min": ATRASO_GOES_MIN, "idade_msg_h": IDADE_MSG_H,
              "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"), "skill_teste": {}}
    G, M = goes.VARIAVEIS, msg.VARIAVEIS
    for nome, cols in (("espinha", base), ("so_goes", base + G), ("goes_msg", base + G + M),
                       ("so_goes_metar", base + G + COLS_METAR + regra_rta.COLS_REGRA),
                       ("goes_msg_metar", base + G + M + COLS_METAR + regra_rta.COLS_REGRA)):
        mods, cals = T.treinar_gbm_calibrado(tr, va, cols, SEMENTE)
        p = T.prever(mods, cals, te, cols)
        sk = 1 - T.rps(p, y) / T.rps(clima, y)
        pacote["skill_teste"][nome] = round(float(sk), 4)
        print(f"  {nome:9s} {len(cols)} entradas | skill no teste set-out/2026 {sk:.3f}", flush=True)
        if nome != "espinha":            # a espinha so serve de referencia aqui
            pacote[nome] = {"modelos": mods, "calibradores": cals, "features": cols}
    # segunda etapa da estacao do CNP (T41): treinada nas linhas de 2026 com estacao,
    # sobre o modelo com METAR (o que vai para a pagina)
    V = estacao_cnp.variaveis(df.ts_utc, estacao_cnp.carregar_5min())[segunda_etapa.COLS]
    ok = (df.ts >= "2026-01-01").to_numpy() & V.notna().all(axis=1).to_numpy()
    mod = pacote["goes_msg_metar"]
    base_p = T.prever(mod["modelos"], mod["calibradores"], df[ok].reset_index(drop=True), mod["features"])
    se = segunda_etapa.SegundaEtapa().ajustar(V[ok].reset_index(drop=True), base_p, df.alvo.to_numpy()[ok])
    pacote["estacao_cnp"] = se
    print(f"  segunda etapa da estacao do CNP: {ok.sum()} linhas de 2026 "
          f"({df.ts[ok].min():%d/%m/%Y} a {df.ts[ok].max():%d/%m/%Y})")
    with open(MODELO, "wb") as fh:
        pickle.dump(pacote, fh)
    print(f"salvo: {MODELO}")


# ------------------------------------------------------------------ ao vivo
def gravar_entradas_hoje(feats: pd.DataFrame, brutas: pd.DataFrame):
    """Chamado pela pagina: guarda as entradas horarias da espinha das
    proximas 36 h, para a renovacao de hora em hora nao precisar refazer a
    previsao inteira."""
    cols = _base()
    X = feats.join(brutas[[c for c in cols if c not in feats.columns]])[cols]
    agora = pd.Timestamp.now(tz="UTC").floor("h")
    X = X[(X.index >= agora - pd.Timedelta(hours=3)) & (X.index <= agora + pd.Timedelta(hours=36))]
    ENTRADAS_HOJE.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(ENTRADAS_HOJE)


def _tempo_no_instante(linha: pd.DataFrame, ts_utc: pd.Timestamp) -> pd.DataFrame:
    """Como no dataset: a meteorologia e a da hora cheia anterior, mas sol,
    hora e epoca sao os do proprio intervalo de 15 min."""
    sys.path.insert(0, str(C.PUBLICAR))
    from gerar_previsao_publica import elevacao_solar  # noqa: PLC0415
    x = linha.copy()
    loc = ts_utc.tz_convert(TZ)
    x["elev_solar"] = elevacao_solar(pd.DatetimeIndex([ts_utc]))
    h = loc.hour + loc.minute / 60
    x["sin_hora"], x["cos_hora"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    x["sin_ano"] = np.sin(2 * np.pi * loc.dayofyear / 365.25)
    x["cos_ano"] = np.cos(2 * np.pi * loc.dayofyear / 365.25)
    return x


def _por_categoria(p: np.ndarray) -> dict:
    return {c: float(p[n:].sum()) for c, n in NIVEL_MINIMO.items()}


SUAVIZAR_MIN = 45                       # T57/T60: media dos intervalos do dia em (t - 45 min, t]


def prever_agora(agora: pd.Timestamp | None = None, entradas: pd.DataFrame | None = None) -> dict:
    """O bloco "agora, com satelite" da pagina. Nunca levanta excecao para
    fora: qualquer falta vira status 'indisponivel' com o motivo."""
    try:
        agora = agora or pd.Timestamp.now(tz="UTC")
        fontes = {}                      # GOES, MSG, METAR e estacao baixados uma vez so
        atual = _prever_agora(agora, entradas, fontes)
        if atual.get("status") == "ok":
            atual = _suavizar(atual, agora, entradas, fontes)
        return atual
    except Exception as e:  # noqa: BLE001
        return {"status": "indisponivel", "motivo": f"{type(e).__name__}: {e}"[:200]}


def _suavizar(atual: dict, agora: pd.Timestamp, entradas, fontes: dict) -> dict:
    """Media das probabilidades do intervalo atual com as dos anteriores do
    mesmo dia na janela de SUAVIZAR_MIN, como no T57 (cada intervalo com o que
    se sabia na hora dele). Intervalo anterior indisponivel fica de fora."""
    ts = pd.Timestamp(atual["ts_utc"])
    ps, usados = [np.array(atual["prob_niveis"])], [atual["ts_utc"]]
    for k in range(1, SUAVIZAR_MIN // 15 + 1):
        if 15 * k >= SUAVIZAR_MIN:
            break
        try:
            r = _prever_agora(agora - pd.Timedelta(minutes=15 * k), entradas, fontes)
        except Exception:  # noqa: BLE001
            continue
        if r.get("status") != "ok" or pd.Timestamp(r["ts_utc"]).tz_convert(TZ).date() != ts.tz_convert(TZ).date():
            continue
        ps.append(np.array(r["prob_niveis"]))
        usados.append(r["ts_utc"])
    p = np.mean(ps, axis=0)
    return {**atual,
            "prob_niveis": [float(v) for v in p],
            "nivel_mais_provavel": int(p.argmax()),
            "prob_por_categoria": _por_categoria(p),
            "sem_suavizar": {"prob_niveis": atual["prob_niveis"], "prob_por_categoria": atual["prob_por_categoria"]},
            "suavizacao": {"janela_min": SUAVIZAR_MIN, "intervalos_utc": usados}}


def _pkl(caminho):
    with open(caminho, "rb") as fh:
        return pickle.load(fh)


def _fonte(fontes: dict | None, chave: str, fn):
    """Busca uma fonte uma vez so por rodada; um erro tambem fica guardado."""
    if fontes is None:
        return fn()
    if chave not in fontes:
        try:
            fontes[chave] = fn()
        except Exception as e:  # noqa: BLE001
            fontes[chave] = e
    if isinstance(fontes[chave], Exception):
        raise fontes[chave]
    return fontes[chave]


def _prever_agora(agora, entradas, fontes: dict | None = None) -> dict:
    """O Agora sem suavizar de um intervalo. Com `fontes`, as series sao
    baixadas na primeira chamada (a do intervalo atual, que cobre os
    anteriores) e reaproveitadas; todas as juncoes sao 'ultimo dado ate o
    instante', entao a serie maior nao deixa o intervalo anterior ver o futuro."""
    agora = agora or pd.Timestamp.now(tz="UTC")
    agora_fontes = fontes.setdefault("_agora", agora) if fontes is not None else agora
    ts = (agora + pd.Timedelta(minutes=ATRASO_GOES_MIN)).floor("15min")
    loc = ts.tz_convert(TZ)
    base = {"ts_utc": ts.isoformat(), "fim_utc": (ts + pd.Timedelta(minutes=15)).isoformat(),
            "gerado_utc": agora.isoformat(timespec="seconds")}
    if not HORA_INI <= loc.hour <= HORA_FIM:
        return {"status": "fora_do_horario", **base}
    pac = _fonte(fontes, "pac", lambda: _pkl(MODELO))
    if entradas is None:
        entradas = _fonte(fontes, "entradas", lambda: pd.read_parquet(ENTRADAS_HOJE))
    hora = ts.floor("h")
    if hora not in entradas.index or entradas.loc[[hora]].isna().any(axis=None):
        return {"status": "indisponivel", "motivo": "sem as entradas da espinha para esta hora", **base}
    linha = _tempo_no_instante(entradas.loc[[hora]].reset_index(drop=True), ts)

    serie = _fonte(fontes, "goes", lambda: goes.recentes(horas=3 if fontes is not None else 2, agora=agora_fontes))
    if not len(serie):
        return {"status": "indisponivel", "motivo": "nenhuma imagem do GOES nas ultimas 2 h", **base}
    g = goes.variaveis(serie, pd.Series([ts]), ATRASO_GOES_MIN)
    if g[["frac_nuvem_1", "frac_nuvem_3"]].isna().any(axis=None):
        return {"status": "indisponivel", "motivo": "imagem do GOES velha demais", **base}
    ultima = serie.loc[pd.to_datetime(serie.criacao, utc=True) <= ts - pd.Timedelta(minutes=ATRASO_GOES_MIN)]
    ultima = ultima.sort_values("criacao").iloc[-1]
    try:
        m = msg.juntar(pd.Series([ts]), _fonte(fontes, "msg", lambda: msg.baixar_recente(agora_fontes)), IDADE_MSG_H)
    except Exception:  # noqa: BLE001 - sem MSG, vale o modelo so com o GOES
        m = pd.DataFrame({c: [np.nan] for c in msg.VARIAVEIS + ["t_msg"]})
    # sol baixo (inicio da manha) deixa o kt vazio tambem no treino; ja o MSG
    # que nao chegou a tempo e falta que o modelo nao viu - entao vai o reserva
    com_msg = bool(m["t_msg"].notna().iloc[0])
    try:
        bruto = _fonte(fontes, "metar", lambda: redemet.metar_iem(redemet.metar(
            AEROPORTOS_METAR, (agora_fontes - pd.Timedelta(hours=4)).floor("h"), agora_fontes.ceil("h"))))
        mt = juntar_metar(pd.Series([ts]), metar_vars(bruto))
        rg = regra_rta.juntar(pd.Series([ts]), regra_rta.da_tabela_metar(bruto))
    except Exception:  # noqa: BLE001 - sem REDEMET, vale o modelo sem METAR
        bruto, mt = pd.DataFrame(), pd.DataFrame({c: [np.nan] for c in COLS_METAR + ["t_metar"]})
        rg = pd.DataFrame({c: [np.nan] for c in regra_rta.COLS_REGRA})
    com_metar = bool(pd.notna(mt["t_metar"].iloc[0])) and "goes_msg_metar" in pac
    nome = ("goes_msg" if com_msg else "so_goes") + ("_metar" if com_metar else "")
    X = pd.concat([linha, g, m[msg.VARIAVEIS], mt[COLS_METAR], rg], axis=1)
    mod = pac[nome]
    p = T.prever(mod["modelos"], mod["calibradores"], X, mod["features"])[0]
    est = None
    if nome == "goes_msg_metar" and "estacao_cnp" in pac:
        try:
            v = estacao_cnp.variaveis(pd.Series([ts]), _fonte(fontes, "estacao", estacao_cnp.carregar_5min))[segunda_etapa.COLS]
            if v.notna().all(axis=1).iloc[0]:
                p_sem = p
                p = pac["estacao_cnp"].prever(v, p[None, :])[0]
                est = {"base_estimada_m": float(v.cnp_lcl_m.iloc[0]), "sol_vs_ceu_limpo": float(v.cnp_kt.iloc[0]),
                       "sem_estacao": _por_categoria(p_sem)}
        except Exception:  # noqa: BLE001 - sem a estacao, vale o modelo sem ela
            est = None
    esp = _fonte(fontes, "espinha", lambda: _pkl(C.ESPINHA / "modelo_reduzido.pkl"))
    p0 = T.prever(esp["modelos"], esp["calibradores"], X, esp["features"])[0]
    return {
        "status": "ok", **base,
        "modelo": nome,
        "prob_niveis": [float(v) for v in p],
        "nivel_mais_provavel": int(p.argmax()),
        "prob_por_categoria": _por_categoria(p),
        "sem_satelite": _por_categoria(p0),
        "goes": {"satelite": str(ultima.satelite),
                 "fim_varredura_utc": pd.Timestamp(ultima.fim_varredura).isoformat(),
                 "idade_min": float(g.idade_goes_min.iloc[0] + ATRASO_GOES_MIN),
                 "frac_nuvem_10km": float(g.frac_nuvem_1.iloc[0]),
                 "frac_baixa_10km": float(g.frac_baixa_1.iloc[0])},
        "msg": {"carimbo_utc": pd.Timestamp(m.t_msg.iloc[0]).isoformat()} if com_msg else None,
        "metar": _resumo_metar(bruto, mt) if com_metar else None,
        "estacao_cnp": est,
        "norma": ({"estrita": int(rg.regra_estrita.iloc[0]), "branda": int(rg.regra_branda.iloc[0])}
                  if com_metar and pd.notna(rg.regra_estrita.iloc[0]) else None),
        # ganho sobre a espinha: T15; com METAR, mais os +0,042 do T26 (so_goes_metar nao foi medido a parte)
        "ganho_medido": {"goes_msg": 0.059, "so_goes": 0.043, "goes_msg_metar": 0.101,
                         "so_goes_metar": None}[nome],
    }


def _resumo_metar(bruto: pd.DataFrame, mt: pd.DataFrame) -> dict:
    """O METAR de Sorocaba que entrou, para a pagina mostrar."""
    t = pd.Timestamp(mt.t_metar.iloc[0])
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    st = AEROPORTOS_METAR[0]
    linha = bruto[(bruto.station == st) & (pd.to_datetime(bruto.valid, utc=True) == t)]
    v = {k: mt[f"metar_{st}_{k}"].iloc[0] for k in ("chovendo", "trovoada", "teto_ft")}
    return {"estacao": st, "hora_utc": t.isoformat(),
            "texto": str(linha.metar.iloc[-1]) if len(linha) else None,
            "chovendo": bool(v["chovendo"] == 1), "trovoada": bool(v["trovoada"] == 1),
            "teto_ft": None if pd.isna(v["teto_ft"]) else float(v["teto_ft"])}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "treinar":
        treinar()
    else:
        print(json.dumps(prever_agora(), ensure_ascii=False, indent=2, default=str))
