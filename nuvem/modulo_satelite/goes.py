"""
modulo_satelite/goes.py - nuvem observada pelo satelite GOES-East sobre Boituva,
a cada 10 min, para o historico (treino) e ao vivo (pagina), com a MESMA
funcao de extracao nos dois (paridade treino/producao).

Por que: o T14 mostrou que a nebulosidade observada ha 15 min melhora a
decisao em +0,07 de skill (+0,10 nos dias de nuvem), mas o satelite que o
projeto tinha (MSG via Open-Meteo) chega com ~2 h de atraso. O GOES-East cobre
o Brasil a cada 10 min e fica no AWS em poucos minutos.

Produto: ABI-L2-ACHAF (altura do topo da nuvem, disco inteiro, 10 km, 10 min;
~1,5 MB por arquivo, baixado inteiro). Conta como nuvem o pixel com altura e
bandeira de qualidade boa, marginal ou opaca (DQF 0, 1, 4); sem nuvem, o
produto grava preenchimento com DQF 3. Satelites: GOES-16 ate a troca de
07/04/2025, GOES-19 depois. A posicao do CNP na grade e calculada pela
longitude de cada arquivo (o GOES-19 esteve em teste a 89,5 W antes da troca).
Conferido contra o MSG: ceu claro -> sem nuvem; ceu fechado -> topos de
9-14 km; nuvem quebrada de tarde -> topos de 1,5-2,5 km.
Fonte: NOAA Open Data Dissemination, buckets publicos noaa-goes16/19.

Variaveis por imagem, em duas caixas centradas no CNP (raio de 1 pixel ~30 km
e de 3 pixels ~70 km):
  frac_nuvem_r  fracao de pixels com nuvem
  frac_baixa_r  fracao com topo abaixo de 3.000 m (nuvem baixa)
  ht_min_r      menor topo de nuvem (m)
  ht_med_r      mediana do topo (m)

Tempo de cada imagem: fim da varredura (e...) e criacao do produto (c...),
tirados do nome do arquivo. No treino, so entra imagem com CRIACAO <=
instante decidido - atraso (sem vazamento).

Uso (historico; feito para rodar no Colab, rede rapida):
    import goes
    goes.baixar_periodo("2025-01-01", "2026-10-12", "goes_acha_boituva.parquet")
Uso (ao vivo):
    goes.recentes(horas=3)   # DataFrame com as imagens das ultimas horas
"""
from __future__ import annotations

import io
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

LAT, LON = -23.2983, -47.6924                  # SDOI (o mesmo de modulo_rajada/base.py)
PRODUTO = "ABI-L2-ACHAF"
# GOES-East: o 16 ate a troca (07/04/2025), o 19 depois. Antes da troca o 19
# tambem publica, mas em teste a 89,5 W - por isso a ordem depende da data.
TROCA_GOES19 = pd.Timestamp("2025-04-07", tz="UTC")
DQF_NUVEM = (0, 1, 4)                          # boa, marginal, nuvem opaca (2 = tentou e falhou; 3 = sem nuvem)
RAIOS = (1, 3)                                 # pixels de 10 km
LIMIAR_BAIXA_M = 3000.0
HORAS_UTC = range(8, 22)                       # 05-18h locais: janela de operacao + 3 h antes
_SESS = requests.Session()
# um pool do tamanho das threads: com o padrao (10), as conexoes a mais sao
# descartadas e reabertas, e o urllib3 avisa a cada vez ("Connection pool is full")
_SESS.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=64))
_INDICE: dict[tuple, tuple[int, int]] = {}     # (satelite, longitude, forma) -> (linha, coluna) do CNP


def _tempo(nome: str, letra: str) -> pd.Timestamp:
    m = re.search(rf"_{letra}(\d{{4}})(\d{{3}})(\d{{2}})(\d{{2}})(\d{{2}})", nome)
    a, doy, h, mi, s = map(int, m.groups())
    return pd.Timestamp(datetime(a, 1, 1, h, mi, s, tzinfo=timezone.utc) + timedelta(days=doy - 1))


def listar_hora(t: pd.Timestamp) -> list[tuple[str, str]]:
    """Arquivos (bucket, chave) de uma hora UTC, do primeiro bucket que tiver."""
    pref = f"{PRODUTO}/{t:%Y}/{t.dayofyear:03d}/{t:%H}/"
    buckets = ("noaa-goes19", "noaa-goes16") if t >= TROCA_GOES19 else ("noaa-goes16", "noaa-goes19")
    for b in buckets:
        for tentativa in range(4):
            try:
                r = _SESS.get(f"https://{b}.s3.amazonaws.com/", params={"list-type": "2", "prefix": pref},
                              timeout=30)
                ks = re.findall(r"<Key>([^<]+)</Key>", r.text)
                break
            except requests.RequestException:
                time.sleep(2 ** tentativa)
        else:
            ks = []
        if ks:
            return [(b, k) for k in ks if k.endswith(".nc")]
    return []


def _xy_fixo(g) -> tuple[float, float]:
    """Latitude/longitude do CNP -> angulos de varredura (x, y) da grade fixa
    do ABI (GOES-R Product User Guide, vol. 4, secao 4.2.8)."""
    a = lambda n: float(np.ravel(g.attrs[n])[0])  # noqa: E731
    req, rpol = a("semi_major_axis"), a("semi_minor_axis")
    H = a("perspective_point_height") + req
    lam0 = math.radians(a("longitude_of_projection_origin"))
    phi, lam = math.radians(LAT), math.radians(LON)
    e2 = (req ** 2 - rpol ** 2) / req ** 2
    phic = math.atan(rpol ** 2 / req ** 2 * math.tan(phi))
    rc = rpol / math.sqrt(1 - e2 * math.cos(phic) ** 2)
    sx = H - rc * math.cos(phic) * math.cos(lam - lam0)
    sy = -rc * math.cos(phic) * math.sin(lam - lam0)
    sz = rc * math.sin(phic)
    return math.asin(-sy / math.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), math.atan(sz / sx)


def extrair(conteudo: bytes, nome: str) -> dict:
    """Uma imagem ACHA -> as variaveis do CNP. A MESMA funcao no treino e ao vivo."""
    import h5py
    sat = re.search(r"_(G\d\d)_", nome).group(1)
    with h5py.File(io.BytesIO(conteudo), "r") as f:
        d = f["HT"]
        lon0 = round(float(np.ravel(f["goes_imager_projection"].attrs["longitude_of_projection_origin"])[0]), 2)
        chave = (sat, lon0, d.shape)
        if chave not in _INDICE:
            x0, y0 = _xy_fixo(f["goes_imager_projection"])
            xv = f["x"][:] * f["x"].attrs["scale_factor"] + f["x"].attrs["add_offset"]
            yv = f["y"][:] * f["y"].attrs["scale_factor"] + f["y"].attrs["add_offset"]
            _INDICE[chave] = (int(np.abs(yv - y0).argmin()), int(np.abs(xv - x0).argmin()))
        i, j = _INDICE[chave]
        r = max(RAIOS)
        bruto = d[i - r:i + r + 1, j - r:j + r + 1]
        dqf = f["DQF"][i - r:i + r + 1, j - r:j + r + 1]
        fill = int(np.ravel(d.attrs["_FillValue"])[0])
        esc = float(np.ravel(d.attrs.get("scale_factor", 1.0))[0])
        off = float(np.ravel(d.attrs.get("add_offset", 0.0))[0])
    valido = (bruto != fill) & np.isin(dqf, DQF_NUVEM)
    ht = np.where(valido, bruto * esc + off, np.nan)
    out = {"satelite": sat, "lon_satelite": lon0, "fim_varredura": _tempo(nome, "e"),
           "criacao": _tempo(nome, "c")}
    for rr in RAIOS:
        c = ht[r - rr:r + rr + 1, r - rr:r + rr + 1]
        nuv = np.isfinite(c)
        out[f"frac_nuvem_{rr}"] = float(nuv.mean())
        out[f"frac_baixa_{rr}"] = float((c < LIMIAR_BAIXA_M).sum() / c.size)
        out[f"ht_min_{rr}"] = float(np.nanmin(c)) if nuv.any() else np.nan
        out[f"ht_med_{rr}"] = float(np.nanmedian(c)) if nuv.any() else np.nan
    return out


def _baixar_um(bk: tuple[str, str]) -> dict | None:
    b, k = bk
    for tentativa in range(4):
        try:
            r = _SESS.get(f"https://{b}.s3.amazonaws.com/{k}", timeout=60)
            if r.status_code == 200:
                return extrair(r.content, k.rsplit("/", 1)[-1])
        except Exception:  # noqa: BLE001 - arquivo corrompido ou rede: tenta de novo
            pass
        time.sleep(2 ** tentativa)
    return None


def baixar_periodo(inicio: str, fim: str, saida: str, threads: int = 32,
                   horas_utc=HORAS_UTC, dias_por_lote: int = 5) -> pd.DataFrame:
    """Historico, em lotes de dias gravados a cada lote (retoma de onde parou)."""
    feito = pd.read_parquet(saida) if os.path.exists(saida) else pd.DataFrame()
    dias_feitos = set(pd.to_datetime(feito.fim_varredura).dt.date) if len(feito) else set()
    dias = [d for d in pd.date_range(inicio, fim, freq="D", tz="UTC") if d.date() not in dias_feitos]
    print(f"{len(dias)} dias a baixar ({len(dias_feitos)} ja feitos)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for n in range(0, len(dias), dias_por_lote):
            lote = dias[n:n + dias_por_lote]
            horas = [d + pd.Timedelta(hours=h) for d in lote for h in horas_utc]
            arquivos = [bk for lst in ex.map(listar_hora, horas) for bk in lst]
            linhas = [x for x in ex.map(_baixar_um, arquivos) if x is not None]
            if linhas:
                feito = pd.concat([feito, pd.DataFrame(linhas)], ignore_index=True)
                feito.to_parquet(saida, index=False)
            print(f"  {lote[0]:%Y-%m-%d}..{lote[-1]:%Y-%m-%d}: {len(linhas)}/{len(arquivos)} imagens "
                  f"| total {len(feito):,} | {(time.time() - t0) / 60:.0f} min".replace(",", "."), flush=True)
    return feito


def recentes(horas: int = 3, agora: pd.Timestamp | None = None, threads: int = 8) -> pd.DataFrame:
    """Ao vivo: as imagens das ultimas `horas` horas (mesma extracao do treino)."""
    agora = agora or pd.Timestamp.now(tz="UTC")
    hs = [(agora - pd.Timedelta(hours=k)).floor("h") for k in range(horas + 1)]
    with ThreadPoolExecutor(threads) as ex:
        arquivos = [bk for lst in ex.map(listar_hora, hs) for bk in lst]
        linhas = [x for x in ex.map(_baixar_um, arquivos) if x is not None]
    d = pd.DataFrame(linhas)
    return d.sort_values("fim_varredura").reset_index(drop=True) if len(d) else d


# ------------------------------------------------- variaveis para a espinha
VARIAVEIS = ["frac_nuvem_1", "frac_baixa_1", "ht_min_1", "frac_nuvem_3", "frac_baixa_3",
             "frac_nuvem_1_1h", "frac_baixa_1_1h", "dfrac_nuvem_1_1h", "idade_goes_min"]


def variaveis(serie: pd.DataFrame, instantes: pd.Series, atraso_min: float = 0.0) -> pd.DataFrame:
    """Para cada instante decidido (UTC), o que o satelite mostrava com
    criacao <= instante - atraso: a ultima imagem, a media da ultima hora e a
    variacao em 1 h. `idade_goes_min` = minutos entre o fim da varredura e o
    instante (o modelo sabe quao velho e o dado)."""
    s = serie.copy()
    s["criacao"] = pd.to_datetime(s.criacao, utc=True)
    s["fim_varredura"] = pd.to_datetime(s.fim_varredura, utc=True)
    s = s.drop_duplicates("fim_varredura").sort_values("fim_varredura").reset_index(drop=True)
    idx = s.set_index("fim_varredura")
    for c in ("frac_nuvem_1", "frac_baixa_1"):
        s[f"{c}_1h"] = idx[c].rolling("60min", min_periods=2).mean().to_numpy()
    # a imagem de ~1 h antes (a mais proxima, ate 15 min de folga)
    antes = pd.merge_asof(pd.DataFrame({"t": s.fim_varredura - pd.Timedelta("60min")}),
                          s[["fim_varredura", "frac_nuvem_1"]], left_on="t", right_on="fim_varredura",
                          direction="nearest", tolerance=pd.Timedelta("15min"))
    s["dfrac_nuvem_1_1h"] = s.frac_nuvem_1.to_numpy() - antes.frac_nuvem_1.to_numpy()
    s = s.sort_values("criacao").reset_index(drop=True)
    ref = pd.to_datetime(instantes, utc=True) - pd.Timedelta(minutes=atraso_min)
    a = pd.DataFrame({"t_ref": ref.reset_index(drop=True), "pos": np.arange(len(ref))}).sort_values("t_ref")
    m = pd.merge_asof(a, s, left_on="t_ref", right_on="criacao", direction="backward",
                      tolerance=pd.Timedelta("90min")).sort_values("pos")
    out = m[[c for c in VARIAVEIS if c != "idade_goes_min"]].reset_index(drop=True)
    out["idade_goes_min"] = ((m.t_ref - m.fim_varredura).dt.total_seconds() / 60).to_numpy()
    return out
