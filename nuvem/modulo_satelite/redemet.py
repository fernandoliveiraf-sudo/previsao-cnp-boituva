"""
modulo_satelite/redemet.py - METAR e radar meteorologico da REDEMET (DECEA),
para o historico (treino) e ao vivo (pagina), com as MESMAS funcoes nos dois.

Chave: variavel de ambiente REDEMET_KEY (no Windows, a do usuario; nunca em
arquivo do projeto). Limites da chave: 900 consultas/min, 10 mil/h, 100 mil/dia.

METAR/SPECI (mensagens/metar, desde 2003): a fonte oficial brasileira. Tem
Sorocaba (SDCO, ~20 km do CNP) e Jundiai (SBJD), que a distribuicao
internacional (aviationweather, IEM) nao traz, e nao sofre o atraso de
distribuicao visto em 14/10/2026. metar_iem() converte para o formato da IEM
usado em modulo_rajada/sem_inmet.py.

Radar (produtos/radar, desde 2006): imagem MaxCAPPI de Sao Roque/SP (centro a
~65 km do CNP, raio de 400 km), 400 x 400 pixels (~2 km), a cada 10-20 min. A
cor vai do azul (eco fraco) ao verde, amarelo, vermelho e magenta (tempestade).
intensidade() converte a cor numa escala ORDINAL de 1 a 6 pelo matiz (a legenda
em dBZ nao vem na API); para o modelo basta a ordem.
"""
from __future__ import annotations

import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

BASE = "https://api-redemet.decea.mil.br"
LAT, LON = -23.2983, -47.6924
_S = requests.Session()
_S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=32))


def chave() -> str:
    k = os.environ.get("REDEMET_KEY")
    if not k and os.name == "nt":                     # variavel do usuario criada depois do terminal abrir
        import winreg
        try:
            k = winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), "REDEMET_KEY")[0]
        except OSError:
            k = None
    if not k:
        raise RuntimeError("sem a chave da REDEMET (variavel REDEMET_KEY)")
    return k


def _get(caminho: str, params: dict, timeout: int = 60) -> dict:
    for t in range(5):
        try:
            r = _S.get(f"{BASE}/{caminho}", headers={"X-Api-Key": chave()}, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 30)))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(3 * 2 ** t)
    raise RuntimeError(f"REDEMET sem resposta: {caminho}")


# =================================================================== METAR
def metar(localidades: list[str], ini: pd.Timestamp, fim: pd.Timestamp) -> pd.DataFrame:
    """Mensagens METAR/SPECI entre ini e fim (UTC), todas as paginas."""
    linhas, pag = [], 1
    while True:
        j = _get(f"mensagens/metar/{','.join(localidades)}",
                 {"data_ini": ini.strftime("%Y%m%d%H"), "data_fim": fim.strftime("%Y%m%d%H"),
                  "page_tam": 150, "page": pag})["data"]
        linhas += j["data"]
        if j.get("next_page_url") in (None, "") or pag >= j.get("last_page", 1):
            break
        pag += 1
    return pd.DataFrame(linhas, columns=["id_localidade", "validade_inicial", "mens", "recebimento"])


_VENTO = re.compile(r"\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT\b")
_TEMP = re.compile(r"\b(M?\d{2})/(M?\d{2})?\b")
_Q = re.compile(r"\bQ(\d{4})\b")
_HORA = re.compile(r"\b(\d{2})(\d{2})(\d{2})Z\b")
_WX = re.compile(r"(?<!\S)([-+]|VC)?(MI|PR|BC|DR|BL|SH|TS|FZ)?(DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)+(?!\S)"
                 r"|(?<!\S)([-+]|VC)?TS(?!\S)"
                 r"|(?<!\S)VC(SH|FG|PO|SS|DS|FC|BL)(?!\S)")


def metar_iem(m: pd.DataFrame) -> pd.DataFrame:
    """Mensagens da REDEMET -> formato da IEM (station, valid, tmpf, sknt,
    gust, alti, wxcodes, metar), o que modulo_rajada/sem_inmet.py le."""
    out = []
    for _, r in m.iterrows():
        txt = re.sub(r"=\s*$", "", str(r.mens)).strip()
        corpo = re.sub(r"^(METAR|SPECI)\s+(COR\s+)?", "", txt)
        corpo = corpo.split(" RMK ")[0]
        h = _HORA.search(corpo)
        if not h:
            continue
        base = pd.Timestamp(r.validade_inicial, tz="UTC")
        valid = base.replace(hour=int(h.group(2)), minute=int(h.group(3)))
        if valid - base > pd.Timedelta(hours=12):        # virada de dia
            valid -= pd.Timedelta(days=1)
        v = _VENTO.search(corpo)
        t = _TEMP.search(corpo.split("Q")[0] if "Q" in corpo else corpo)
        q = _Q.search(corpo)
        temp = (-1 if t and t.group(1).startswith("M") else 1) * float(t.group(1).lstrip("M")) if t else np.nan
        wx = " ".join(g.group(0) for g in _WX.finditer(corpo.split(" Q")[0]))
        out.append({"station": r.id_localidade, "valid": valid,
                    "tmpf": round(temp * 9 / 5 + 32, 2) if np.isfinite(temp) else np.nan,
                    "drct": float(v.group(1)) if v and v.group(1) != "VRB" else np.nan,
                    "sknt": float(v.group(2)) if v else np.nan,
                    "gust": float(v.group(3)) if v and v.group(3) else np.nan,
                    "alti": round(float(q.group(1)) / 33.8639, 2) if q else np.nan,
                    "wxcodes": wx or None, "metar": corpo, "recebido": pd.Timestamp(r.recebimento, tz="UTC")})
    return pd.DataFrame(out)


# =================================================================== radar
SAO_ROQUE = {"lat_min": -27.1905651987, "lat_max": -19.9677734375,
             "lon_min": -51.1249313354, "lon_max": -43.0637242496}


def radar_quadros(fim_hora: pd.Timestamp, n: int = 15, area: str = "sr", tipo: str = "maxcappi") -> pd.DataFrame:
    """Os ultimos n quadros ate a hora fim_hora (UTC): horario e endereco da imagem."""
    j = _get(f"produtos/radar/{tipo}", {"area": area, "data": fim_hora.strftime("%Y%m%d%H"), "anima": n})
    linhas = []
    for quadro in j["data"]["radar"]:
        for x in quadro:
            if x and x.get("localidade") == area and x.get("path"):
                linhas.append({"tempo": pd.Timestamp(x["data"], tz="UTC"), "path": x["path"],
                               **{k: float(x[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")}})
    return pd.DataFrame(linhas).drop_duplicates("path") if linhas else pd.DataFrame()


def intensidade(img_rgba: np.ndarray) -> np.ndarray:
    """Cor -> intensidade ordinal: 0 sem eco; 1 azul ... 3 verde, 4 amarelo,
    5 vermelho, 6 magenta. Pelo matiz (HSV)."""
    rgb = img_rgba[..., :3].astype(float) / 255
    alfa = img_rgba[..., 3]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.where(mx - mn == 0, 1, mx - mn)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    idx = np.where(h <= 240, (240 - h) / 60 + 1, 5 + (360 - h) / 120)    # azul 1, verde 3, amarelo 4, vermelho 5
    idx = np.where(h > 240, np.clip(6 - (h - 240) / 60, 5, 6), idx)      # roxo/magenta acima do vermelho
    return np.where((alfa > 0) & (mx - mn > 0.05), idx, 0.0).astype(np.float32)


def baixar_imagem(path: str) -> np.ndarray | None:
    from PIL import Image
    for t in range(4):
        try:
            r = _S.get(path, timeout=60)
            if r.status_code == 200:
                return intensidade(np.array(Image.open(io.BytesIO(r.content)).convert("RGBA")))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 ** t)
    return None


ANEIS_KM = [(0, 10), (10, 25), (25, 50), (50, 100)]


def _grade_km(caixa: dict, forma, lat0, lon0) -> np.ndarray:
    ny, nx = forma
    lats = caixa["lat_max"] - (np.arange(ny) + 0.5) * (caixa["lat_max"] - caixa["lat_min"]) / ny
    lons = caixa["lon_min"] + (np.arange(nx) + 0.5) * (caixa["lon_max"] - caixa["lon_min"]) / nx
    la, lo = np.meshgrid(np.radians(lats), np.radians(lons), indexing="ij")
    p0, l0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((la - p0) / 2) ** 2 + np.cos(p0) * np.cos(la) * np.sin((lo - l0) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def resumo(I: np.ndarray, km: np.ndarray) -> dict:
    """Uma imagem de intensidade -> numeros em volta de um ponto."""
    out = {}
    for a, b in ANEIS_KM:
        m = (km >= a) & (km < b)
        v = I[m]
        out[f"rad_max_{a}_{b}km"] = float(v.max()) if v.size else np.nan
        out[f"rad_frac_{a}_{b}km"] = float((v > 0).mean()) if v.size else np.nan
        out[f"rad_forte_{a}_{b}km"] = float((v >= 4).mean()) if v.size else np.nan
    forte = (I >= 4) & (km <= 150)
    out["rad_km_forte"] = float(km[forte].min()) if forte.any() else 150.0
    eco = (I > 0) & (km <= 150)
    out["rad_km_eco"] = float(km[eco].min()) if eco.any() else 150.0
    return out


# radar -> variaveis numa hora t: so quadros com horario <= t - ATRASO_RADAR_MIN
# (ao vivo, em 14/10/2026 as 21:56 o ultimo quadro disponivel era o das 21:36;
# os quadros saem a cada 20 min)
ATRASO_RADAR_MIN = 20
_TENDENCIA = ["rad_max_0_10km", "rad_max_10_25km", "rad_frac_10_25km", "rad_frac_25_50km",
              "rad_forte_25_50km", "rad_km_forte", "rad_km_eco"]


def resumir_quadros(quadros: pd.DataFrame, imagens, lat0: float, lon0: float) -> pd.DataFrame:
    """quadros: tempo e caixa de cada imagem; imagens: as matrizes de
    intensidade na mesma ordem. Devolve uma linha por quadro com resumo()."""
    linhas, cache_km = [], {}
    for (_, q), I in zip(quadros.iterrows(), imagens):
        if I is None:
            continue
        chave_km = (q.lat_min, q.lat_max, q.lon_min, q.lon_max, I.shape)
        if chave_km not in cache_km:
            cache_km[chave_km] = _grade_km(q, I.shape, lat0, lon0)
        linhas.append({"tempo": q.tempo, **resumo(I, cache_km[chave_km])})
    return pd.DataFrame(linhas)


def variaveis_radar(resumos: pd.DataFrame, instantes: pd.Series) -> pd.DataFrame:
    """Para cada t: o ultimo quadro utilizavel, a idade dele e a variacao em
    ~40 min (quadro anterior utilizavel mais proximo de 40 min antes)."""
    s = resumos.copy()
    s["tempo"] = pd.to_datetime(s.tempo, utc=True)
    s = s.sort_values("tempo").reset_index(drop=True)
    t = pd.to_datetime(pd.Series(instantes), utc=True).reset_index(drop=True)
    ref = t - pd.Timedelta(minutes=ATRASO_RADAR_MIN)
    a = pd.DataFrame({"t": ref, "pos": np.arange(len(t))}).sort_values("t")
    m = pd.merge_asof(a, s, left_on="t", right_on="tempo", direction="backward",
                      tolerance=pd.Timedelta("40min")).sort_values("pos")
    b = pd.DataFrame({"t": m.tempo - pd.Timedelta(minutes=40), "pos": np.arange(len(t))}).sort_values("t")
    b = b.dropna(subset=["t"])
    mb = pd.merge_asof(b, s, left_on="t", right_on="tempo", direction="nearest",
                       tolerance=pd.Timedelta("15min")).set_index("pos").reindex(range(len(t)))
    cols = [c for c in s.columns if c != "tempo"]
    out = m[cols].reset_index(drop=True)
    for c in _TENDENCIA:
        out[f"{c}_d40"] = out[c].to_numpy() - mb[c].to_numpy()
    out["rad_idade_min"] = ((t - m.tempo.reset_index(drop=True)).dt.total_seconds() / 60).to_numpy()
    return out


# =================================================================== STSC
# Ocorrencia de trovoada (produtos/stsc): pontos numa grade de 0,1 grau
# (~11 km), um quadro a cada ~6 min. Uma consulta com anima=60 devolve a hora
# pedida e as 3 anteriores. Guarda so a caixa de ~400 km em volta do CNP.
CAIXA_STSC = (-27.0, -19.6, -51.7, -43.7)        # lat min, lat max, lon min, lon max


def stsc(data_hora: pd.Timestamp, anima: int = 60) -> pd.DataFrame:
    j = _get("produtos/stsc", {"data": data_hora.strftime("%Y%m%d%H"), "anima": anima}, timeout=150)["data"]
    linhas = []
    base = data_hora.floor("D")
    for hhmm, pts in zip(j.get("anima", []), j.get("stsc", [])):
        h, m = map(int, hhmm.split(":"))
        t = base + pd.Timedelta(hours=h, minutes=m)
        if t > data_hora + pd.Timedelta(hours=1):          # virada de dia
            t -= pd.Timedelta(days=1)
        for p in pts or []:
            la, lo = float(p["la"]), float(p["lo"])
            if CAIXA_STSC[0] <= la <= CAIXA_STSC[1] and CAIXA_STSC[2] <= lo <= CAIXA_STSC[3]:
                linhas.append((t, la, lo))
        if not pts:
            linhas.append((t, np.nan, np.nan))              # quadro sem trovoada na America do Sul
    d = pd.DataFrame(linhas, columns=["tempo", "lat", "lon"])
    d.attrs["quadros"] = [str(base + pd.Timedelta(hours=int(x[:2]), minutes=int(x[3:]))) for x in j.get("anima", [])]
    return d


# ============================================================ TAF e TEMP
def mensagens(tipo: str, ini: pd.Timestamp, fim: pd.Timestamp, localidades: list[str] | None = None,
              estacao: str | None = None) -> pd.DataFrame:
    """TAF (mensagens/taf/{loc}) ou TEMP (mensagens/temp?estacao=), todas as paginas."""
    caminho = f"mensagens/{tipo}/{','.join(localidades)}" if localidades else f"mensagens/{tipo}"
    linhas, pag = [], 1
    while True:
        p = {"data_ini": ini.strftime("%Y%m%d%H"), "data_fim": fim.strftime("%Y%m%d%H"), "page_tam": 150, "page": pag}
        if estacao:
            p["estacao"] = estacao
        j = _get(caminho, p)["data"]
        linhas += j["data"]
        if not j.get("next_page_url") or pag >= j.get("last_page", 1):
            break
        pag += 1
    return pd.DataFrame(linhas)


def _temp_ttt(s: str):
    """TTTDD do TEMP -> (temperatura, ponto de orvalho) em graus C."""
    if "/" in s[:3]:
        return np.nan, np.nan
    t = int(s[:3]) / 10
    t = -t if int(s[2]) % 2 else t                         # decimo impar = negativo
    if "/" in s[3:5]:
        return t, np.nan
    dd = int(s[3:5])
    dep = dd / 10 if dd <= 50 else dd - 50 if dd >= 56 else np.nan
    return t, t - dep


def _vento(s: str):
    if "/" in s:
        return np.nan, np.nan
    d, f = int(s[:3]), int(s[3:])
    if d % 5:                                              # vento >= 100 kt: a unidade da direcao vai para a velocidade
        f += 100 * (d % 5)
        d -= d % 5
    return float(d), float(f)


def temp_indices(mens: str) -> dict:
    """Parte A do TEMP (TTAA) -> niveis padrao e indices de instabilidade."""
    g = mens.replace("=", "").split()
    if not g or g[0] != "TTAA":
        return {}
    niveis = {"92": 925, "85": 850, "70": 700, "50": 500}
    out = {}
    i = 3
    while i + 2 < len(g):
        cod = g[i][:2]
        if cod in niveis:
            t, td = _temp_ttt(g[i + 1])
            d, f = _vento(g[i + 2])
            n = niveis[cod]
            out.update({f"t{n}": t, f"td{n}": td, f"dir{n}": d, f"vel{n}": f})
        if cod in ("88", "77", "31"):
            break
        i += 3
    if all(k in out for k in ("t850", "td850", "t700", "td700", "t500")):
        out["indice_k"] = out["t850"] - out["t500"] + out["td850"] - (out["t700"] - out["td700"])
        out["totais_totais"] = out["t850"] + out["td850"] - 2 * out["t500"]
    if all(k in out for k in ("dir850", "vel850", "dir500", "vel500")):
        u = lambda d, f: (-f * np.sin(np.radians(d)), -f * np.cos(np.radians(d)))  # noqa: E731
        (u8, v8), (u5, v5) = u(out["dir850"], out["vel850"]), u(out["dir500"], out["vel500"])
        out["cisalhamento_850_500_kt"] = float(np.hypot(u5 - u8, v5 - v8))
    return out
